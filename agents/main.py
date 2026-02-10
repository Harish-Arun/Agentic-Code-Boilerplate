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
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from config import get_config
from models import HealthResponse, AgentState

from graph.workflow import create_workflow, run_workflow, resume_workflow, get_workflow_state


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
    """Request to resume a paused workflow."""
    thread_id: str
    approved: bool = True
    modifications: Dict[str, Any] = {}


class WorkflowResult(BaseModel):
    document_id: str
    thread_id: str
    status: str
    is_paused: bool = False
    current_step: str = ""
    extracted_data: Dict[str, Any] = {}
    signature_result: Dict[str, Any] = {}
    processing_time_ms: int = 0
    errors: List[str] = []


class WorkflowStateResponse(BaseModel):
    """Response for workflow state query."""
    thread_id: str
    exists: bool
    is_paused: bool = False
    current_step: str = ""
    state: Dict[str, Any] = {}


# ============================================
# Lifespan
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    config = get_config()
    app.state.config = config
    
    # create_workflow now returns (workflow, checkpointer)
    workflow, checkpointer = create_workflow(config)
    app.state.workflow = workflow
    app.state.checkpointer = checkpointer
    
    checkpoint_info = "enabled" if checkpointer else "disabled"
    hitl_info = "enabled" if config.features.human_in_loop else "disabled"
    
    print(f"🤖 Agents Service started [LLM: {config.llm.provider}, Checkpointing: {checkpoint_info}, HITL: {hitl_info}]")
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
    
    # Use document_id as thread_id for tracking
    thread_id = f"doc_{request.document_id}"
    
    # Create initial state
    initial_state = AgentState(
        document_id=request.document_id,
        document_path=request.document_path
    )
    
    try:
        result = await run_workflow(
            workflow, 
            initial_state,
            thread_id=thread_id,
            run_extraction=request.run_extraction,
            run_signature=request.run_signature_verification
        )
        
        # Check if workflow is paused (HITL)
        is_paused = result.current_step == "human_review" or getattr(result, 'awaiting_approval', False)
        
        return WorkflowResult(
            document_id=request.document_id,
            thread_id=thread_id,
            status="AWAITING_APPROVAL" if is_paused else ("VERIFIED" if result.verification_result else "EXTRACTED"),
            is_paused=is_paused,
            current_step=result.current_step,
            extracted_data=result.extracted_payment.model_dump(exclude_none=True) if result.extracted_payment else {},
            signature_result=result.verification_result.model_dump(exclude_none=True) if result.verification_result else {},
            processing_time_ms=100,
            errors=result.extraction_errors + result.detection_errors + result.verification_errors
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/resume", response_model=WorkflowResult)
async def resume_agent_workflow(request: ResumeWorkflowRequest):
    """
    Resume a paused workflow after human review.
    
    Use this after /run returns is_paused=True.
    You can optionally modify the extracted data before resuming.
    """
    workflow = app.state.workflow
    
    try:
        # Build updated state if modifications provided
        updated_state = None
        if request.modifications:
            updated_state = request.modifications
            updated_state["awaiting_approval"] = False
        
        result = await resume_workflow(
            workflow,
            thread_id=request.thread_id,
            updated_state=updated_state
        )
        
        response = WorkflowResult(
            document_id=result.document_id,
            thread_id=request.thread_id,
            status="VERIFIED" if result.verification_result else "EXTRACTED",
            is_paused=False,
            current_step=result.current_step,
            extracted_data=result.extracted_payment.model_dump(exclude_none=True) if result.extracted_payment else {},
            signature_result=result.verification_result.model_dump(exclude_none=True) if result.verification_result else {},
            processing_time_ms=100,
            errors=result.extraction_errors + result.detection_errors + result.verification_errors
        )
        
        # DEBUG: Log complete response before sending
        print("\n" + "="*80)
        print("📤 AGENTS RESPONSE TO API SERVICE")
        print("="*80)
        print(f"Status: {response.status}")
        print(f"Current Step: {response.current_step}")
        print(f"\nExtracted Data ({len(response.extracted_data)} fields):")
        import json
        print(json.dumps(response.extracted_data, indent=2))
        print(f"\nSignature Result:")
        print(json.dumps(response.signature_result, indent=2))
        print(f"\nErrors: {response.errors}")
        print("="*80 + "\n")
        
        return response
    except Exception as e:
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
    
    state = get_workflow_state(workflow, thread_id)
    
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
