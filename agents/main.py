"""
NNP-AI Agents Service - LangGraph workflow orchestration.

Features:
- Configurable checkpointing (memory/postgres)
- Human-in-the-loop with resume capability
- Thread-based workflow tracking
"""
import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load .env file for local development
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    load_dotenv(env_file)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from config import get_config
from models import HealthResponse, AgentState

try:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    _SQLITE_AVAILABLE = True
except ImportError:
    AsyncSqliteSaver = None
    _SQLITE_AVAILABLE = False

from graph.workflow import create_workflow, run_workflow, resume_workflow, get_workflow_state, get_checkpoint_db_path
from mcp_client import get_mcp_client, call_tool_on_session


# ============================================
# Request/Response Models
# ============================================
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))


class RunWorkflowRequest(BaseModel):
    document_id: str
    document_path: str = f"{Path(DEFAULT_DATA_DIR).as_posix()}/sample_document.pdf"
    run_extraction: bool = True
    run_signature_verification: bool = True


class ResumeWorkflowRequest(BaseModel):
    """
    Resume a paused workflow after a HITL checkpoint.

    thread_id       : same as the one returned by /run  (= "doc_<document_id>")
    human_role      : which role is clicking Proceed — "keyer" | "authenticator"
    modifications   : state updates to merge before resuming
                      For keyer:        {"extracted_data": {...}, "feedback": {...}}
                      For authenticator: {"auth_feedback": {...}}
    """
    thread_id: str
    human_role: str = "keyer"  # "keyer" | "authenticator"
    modifications: Dict[str, Any] = Field(default_factory=dict)


class VerifySignatureRequest(BaseModel):
    """Manual signature authentication request from the Authenticator."""
    document_id: str
    signature_index: int = 0
    signature_blob: str               # base64 detected signature (resolved by api-service from DB)
    signature_mime_type: str = "image/png"
    reference_blob: str               # base64 ISV reference image
    reference_id: str                 # ISV sigId
    reference_mime_type: str = "image/gif"


class VerifySignatureResponse(BaseModel):
    document_id: str
    signature_index: int
    match: bool
    confidence: float
    reasoning: str
    recommendation: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    scoring_details: Dict[str, Any] = Field(default_factory=dict)
    risk_indicators: List[str] = Field(default_factory=list)
    signature_blob: Optional[str] = None
    reference_blob: Optional[str] = None
    blob_mime_type: str = "image/png"


class WorkflowResult(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    document_id: str
    thread_id: str
    status: str
    is_paused: bool = False
    current_step: str = ""
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    signature_result: Dict[str, Any] = Field(default_factory=dict)
    thinking_traces: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: int = 0
    errors: List[str] = Field(default_factory=list)


class WorkflowStateResponse(BaseModel):
    """Response for workflow state query."""
    thread_id: str
    exists: bool
    is_paused: bool = False
    current_step: str = ""
    state: Dict[str, Any] = Field(default_factory=dict)


def _safe_serialize(obj):
    """
    Recursively convert any remaining non-JSON-safe objects (datetimes, Pydantic models)
    to JSON-safe primitives, preventing circular reference errors during FastAPI serialization.
    """
    import json
    from datetime import datetime, date

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(i) for i in obj]
    if hasattr(obj, 'model_dump'):
        return _safe_serialize(obj.model_dump(mode='json', exclude_none=True))
    try:
        return str(obj)
    except Exception:
        return None


def _collect_thinking_traces(result: AgentState) -> List[Dict[str, Any]]:
    """Collect thinking-trace entries from all attempt lists."""
    traces = []
    for step, attempts in [
        ("extraction",          result.extraction_attempts),
        ("signature_detection", result.detection_attempts),
        ("verification",        result.verification_attempts),
    ]:
        for attempt in attempts:
            if attempt.thoughts or attempt.thoughts_token_count:
                traces.append({
                    "step": step,
                    "attempt": attempt.attempt_number,
                    "thoughts": attempt.thoughts,
                    "thoughts_token_count": attempt.thoughts_token_count,
                    "thinking_budget_used": attempt.thinking_budget_used,
                    "timestamp": attempt.timestamp.isoformat() if hasattr(attempt, "timestamp") else None,
                })
    return traces


def _build_signature_payload(result: AgentState) -> Dict[str, Any]:
    payload = result.verification_result.model_dump(mode='json', exclude_none=True) if result.verification_result else {}

    latest_detection_attempt = result.detection_attempts[-1] if result.detection_attempts else None
    if latest_detection_attempt and latest_detection_attempt.detections:
        payload["detections"] = [d.model_dump(mode='json', exclude_none=True) for d in latest_detection_attempt.detections]

    latest_attempt = result.verification_attempts[-1] if result.verification_attempts else None
    if latest_attempt and latest_attempt.results:
        payload["all_verifications"] = [v.model_dump(mode='json', exclude_none=True) for v in latest_attempt.results]
    elif payload:
        # Copy payload to avoid circular self-reference
        payload["all_verifications"] = [dict(payload)]

    # Include manual per-tile authentication results stored during auth_review interrupt
    if result.manual_verification_results:
        payload["manual_verification_results"] = result.manual_verification_results

    return payload


# ============================================
# Lifespan
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    app.state.config = config

    db_path = get_checkpoint_db_path()
    hitl_info = "enabled" if config.features.human_in_loop else "disabled"

    if _SQLITE_AVAILABLE:
        # AsyncSqliteSaver must stay open for the full app lifetime
        async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
            workflow, _ = create_workflow(config, checkpointer=checkpointer)
            app.state.workflow = workflow
            app.state.checkpointer = checkpointer
            print(f"🤖 Agents Service started [LLM: {config.llm.provider}, Checkpointing: sqlite ({db_path}), HITL: {hitl_info}]")
            yield
    else:
        # Fallback: MemorySaver (state lost on restart, but service still works)
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        workflow, _ = create_workflow(config, checkpointer=checkpointer)
        app.state.workflow = workflow
        app.state.checkpointer = checkpointer
        print(f"🤖 Agents Service started [LLM: {config.llm.provider}, Checkpointing: memory (install langgraph-checkpoint-sqlite for persistence), HITL: {hitl_info}]")
        yield
    print("👋 Agents Service shutdown complete")


# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="NNP-AI Agents Service",
    description="LangGraph-based agent orchestration for document processing",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Health & Info Endpoints
# ============================================
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", service="agents-service")


@app.get("/")
async def root():
    config = get_config()
    return {
        "service": "NNP-AI Agents Service",
        "version": "1.0.0",
        "llm_provider": config.llm.provider,
        "enabled_agents": config.agents.enabled,
        "checkpointing": {
            "enabled": config.agents.checkpointing.enabled,
            "backend": config.agents.checkpointing.backend
        },
        "human_in_loop": config.features.human_in_loop
    }


# ============================================
# Workflow Endpoints
# ============================================
@app.post("/run", response_model=WorkflowResult)
async def run_agent_workflow(request: RunWorkflowRequest):
    """
    Execute the full agent workflow for a document.
    
    If Human-in-the-Loop is enabled, the workflow will pause at
    the human_review step and return with is_paused=True.
    Use /resume to continue after approval.
    
    Flow:
    1. Extraction Agent → Extract payment fields
    2. Human Review → PAUSE (if HITL enabled)
    3. Signature Detection Agent → Find signature regions
    4. Signature Verification Agent → Verify signatures
    """
    config = app.state.config
    workflow = app.state.workflow

    # thread_id is stable for the lifetime of the document
    thread_id = f"doc_{request.document_id}"

    # Create initial state (target_phase removed — workflow controls its own phases via interrupts)
    initial_state = AgentState(
        document_id=request.document_id,
        document_path=request.document_path,
    )

    try:
        result = await run_workflow(
            workflow,
            initial_state,
            thread_id=thread_id,
        )

        # After /run the graph always pauses at keyer_review
        is_paused = result.current_step != "complete"

        return WorkflowResult(
            document_id=request.document_id,
            thread_id=thread_id,
            status="PENDING_KEYER",   # paused at keyer_review — awaiting Keyer action
            is_paused=is_paused,
            current_step=result.current_step,
            extracted_data=result.extracted_payment.model_dump(mode='json', exclude_none=True) if result.extracted_payment else {},
            signature_result=_safe_serialize(_build_signature_payload(result)),
            processing_time_ms=100,
            errors=result.extraction_errors + result.detection_errors + result.verification_errors,
            thinking_traces=_collect_thinking_traces(result)
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume", response_model=WorkflowResult)
async def resume_agent_workflow(request: ResumeWorkflowRequest):
    """
    Resume a paused workflow after a HITL checkpoint.

    - human_role="keyer"         → Graph pauses at auth_review        → status AUTHENTICATED
    - human_role="authenticator" → Graph pauses at verifier_review    → status VERIFIED
    - human_role="verifier"      → Graph runs completion → END
                                   modifications.decision="accept" → status CONFIRMED
                                   modifications.decision="reject" → status EXTRACTED (back to Keyer)
    """
    workflow = app.state.workflow

    try:
        # Pass human changes to the graph via human_modifications
        updated_state: Optional[dict] = None
        if request.modifications:
            updated_state = {
                "human_modifications": request.modifications,
                "awaiting_approval": False,
            }

        result = await resume_workflow(
            workflow,
            thread_id=request.thread_id,
            updated_state=updated_state
        )

        # Terminal states: "complete" (verifier accepted) or "rejected" (verifier rejected)
        terminal = result.current_step in ("complete", "rejected")
        is_paused = not terminal

        if request.human_role == "keyer":
            status = "PENDING_AUTH"       # graph paused at auth_review
        elif request.human_role == "authenticator":
            status = "PENDING_VERIFIER"   # graph paused at verifier_review
        elif request.human_role == "verifier":
            decision = request.modifications.get("decision", "accept")
            status = "CONFIRMED" if decision == "accept" else "PENDING_KEYER"  # reject → back to keyer
        else:
            status = "VERIFIED"  # fallback

        thinking_traces = _collect_thinking_traces(result)

        response = WorkflowResult(
            document_id=result.document_id,
            thread_id=request.thread_id,
            status=status,
            is_paused=is_paused,
            current_step=result.current_step,
            extracted_data=result.extracted_payment.model_dump(mode='json', exclude_none=True) if result.extracted_payment else {},
            signature_result=_safe_serialize(_build_signature_payload(result)),
            processing_time_ms=100,
            errors=result.extraction_errors + result.detection_errors + result.verification_errors,
            thinking_traces=thinking_traces
        )

        print(f"\n📤 RESUME RESPONSE | role={request.human_role} status={status} paused={is_paused} step={result.current_step} terminal={terminal}\n")
        return response

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{thread_id}", response_model=WorkflowStateResponse)
async def get_workflow_status(thread_id: str):
    """
    Get the current state of a workflow by thread_id.
    
    Useful for:
    - Checking if workflow is paused
    - Getting intermediate results
    - Debugging workflow execution
    """
    workflow = app.state.workflow
    
    state = await get_workflow_state(workflow, thread_id)
    
    if state is None:
        return WorkflowStateResponse(
            thread_id=thread_id,
            exists=False
        )
    
    return WorkflowStateResponse(
        thread_id=thread_id,
        exists=True,
        is_paused=state.get("awaiting_approval", False),
        current_step=state.get("current_step", "unknown"),
        state=state
    )


# ============================================
# Individual Agent Endpoints
# ============================================
@app.post("/run/extraction")
async def run_extraction_only(request: RunWorkflowRequest):
    """Run only the extraction agent."""
    config = app.state.config
    
    initial_state = AgentState(
        document_id=request.document_id,
        document_path=request.document_path
    )
    
    from graph.nodes.extraction import extraction_node
    result = await extraction_node(initial_state, config)
    
    return {
        "document_id": request.document_id,
        "extracted_data": result.extracted_payment.model_dump(exclude_none=True) if result.extracted_payment else {},
        "errors": result.extraction_errors
    }


@app.post("/run/signature")
async def run_signature_only(request: RunWorkflowRequest):
    """Run signature detection and verification."""
    config = app.state.config
    
    initial_state = AgentState(
        document_id=request.document_id,
        document_path=request.document_path
    )
    
    from graph.nodes.signature_detection import signature_detection_node
    from graph.nodes.verification import verification_node
    
    state = await signature_detection_node(initial_state, config)
    state = await verification_node(state, config)
    
    return {
        "document_id": request.document_id,
        "signature_detections": [d.model_dump() for d in state.signature_detections],
        "verification_result": state.verification_result.model_dump() if state.verification_result else {},
        "errors": state.detection_errors + state.verification_errors
    }


@app.post("/run/verify", response_model=VerifySignatureResponse)
async def verify_signature_against_isv(request: VerifySignatureRequest):
    """
    Manually authenticate a single detected signature against an ISV reference blob.

    Called by the Authenticator per-tile DURING the auth_review interrupt.
    Results are persisted back into the LangGraph checkpoint via aupdate_state so the
    graph remains the single source of truth.

    Steps:
    1. Get the detected signature blob from the checkpoint state
    2. Call MCP verify_signature with the detected blob + ISV reference blob
    3. Persist result into state.manual_verification_results[sig_index]
    4. Return M1-M7 metrics + FIV scoring details to the frontend
    """
    workflow = app.state.workflow
    thread_id = f"doc_{request.document_id}"

    # signature_blob is passed directly by the api-service (read from DB).
    # No checkpoint read needed — same pattern as /resume.
    signature_blob = request.signature_blob
    sig_mime_type = request.signature_mime_type

    # Call MCP verify_signature with the extracted blob + ISV reference blob
    verify_args = {
        "signature_blob": signature_blob,
        "signature_mime_type": sig_mime_type,
        "reference_blob": request.reference_blob,
        "reference_mime_type": request.reference_mime_type,
    }

    try:
        async with get_mcp_client() as mcp:
            verify_result = await call_tool_on_session(mcp, "verify_signature", verify_args)

        if not verify_result.get("success"):
            raise HTTPException(
                status_code=500,
                detail=f"Signature authentication failed: {verify_result.get('error', 'unknown error')}"
            )

        verification_data = verify_result.get("verification", {})
        metrics_score = verify_result.get("metrics_score", {})

        # Build metrics and scoring details (same structure as verification_node)
        metrics_dict = metrics_score.get("metrics", {}) if metrics_score else {}
        scoring_details: Dict[str, Any] = {}
        if metrics_score:
            veto_info = metrics_score.get("veto", {})
            scoring_details = {
                "vetoed": veto_info.get("vetoed", False),
                "veto_reason": veto_info.get("veto_reason", ""),
                "veto_metric": veto_info.get("veto_metric", ""),
                "base_score": metrics_score.get("base_score", 100.0),
                "penalties": metrics_score.get("penalties", []),
                "bonuses": metrics_score.get("bonuses", []),
                "penalties_applied": sum(p.get("amount", 0) for p in metrics_score.get("penalties", [])),
                "bonuses_applied": sum(b.get("amount", 0) for b in metrics_score.get("bonuses", [])),
                "final_score": metrics_score.get("final_confidence", 0.0),
                "fiv_version": metrics_score.get("fiv_version", "FIV-1.0"),
                "confidence_band": metrics_score.get("confidence_band", "UNKNOWN"),
                "decision": metrics_score.get("decision", "UNKNOWN"),
                "decision_reason": metrics_score.get("decision_reason", ""),
                "llm_model": metrics_score.get("llm_model", "unknown"),
                "processing_time_ms": metrics_score.get("processing_time_ms", 0),
                "audit_summary": metrics_score.get("audit_summary", {}),
            }

        result_dict = {
            "document_id": request.document_id,
            "signature_index": request.signature_index,
            "reference_id": request.reference_id,
            "match": verification_data.get("match", False),
            "confidence": float(verification_data.get("confidence", 0.0)),
            "reasoning": verification_data.get("reasoning", ""),
            "recommendation": verification_data.get("recommendation", "MANUAL_REVIEW"),
            "metrics": _safe_serialize(metrics_dict),
            "scoring_details": _safe_serialize(scoring_details),
            "risk_indicators": verification_data.get("risk_indicators", []),
            "blob_mime_type": sig_mime_type,
        }

        # Persist the result into LangGraph state so the graph owns the authentication data.
        # Read current manual_verification_results first (async, safe with sqlite checkpointer).
        lg_config = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await workflow.aget_state(lg_config)
            current_mvr = dict(snapshot.values.get("manual_verification_results") or {}) if snapshot else {}
        except Exception:
            current_mvr = {}
        current_mvr[str(request.signature_index)] = result_dict
        await workflow.aupdate_state(lg_config, {"manual_verification_results": current_mvr})

        print(f"\n✅ Manual signature authentication: sig_idx={request.signature_index} "
              f"ref_id={request.reference_id} "
              f"match={result_dict['match']} "
              f"confidence={result_dict['confidence']:.2f} "
              f"→ persisted to LangGraph checkpoint\n")

        return VerifySignatureResponse(
            **result_dict,
            signature_blob=signature_blob,
            reference_blob=request.reference_blob,
        )

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path
    
    # Only watch agents/ directory (absolute path)
    service_dir = Path(__file__).parent.resolve()
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[str(service_dir)],
        reload_excludes=["api-service/**", "mcp-tools/**", "frontend/**", "shared/**", "scripts/**"]
    )
