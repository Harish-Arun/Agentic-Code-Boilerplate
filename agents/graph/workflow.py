import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

from models.schemas import AgentState
from config.loader import AppConfig

from graph.nodes.extraction import extraction_node
from graph.nodes.signature_detection import signature_detection_node


# ============================================
# Conditional Edge Functions
# ============================================

def should_continue_after_extraction(state) -> str:
    """Stop the graph if extraction failed; otherwise continue to detection."""
    if isinstance(state, dict):
        extraction_attempts = state.get("extraction_attempts", [])
        extraction_errors = state.get("extraction_errors", [])
        extracted_payment = state.get("extracted_payment")
    else:
        extraction_attempts = state.extraction_attempts
        extraction_errors = state.extraction_errors
        extracted_payment = state.extracted_payment

    if extraction_errors:
        return "end"

    if extraction_attempts:
        latest = extraction_attempts[-1]
        succeeded = latest.success if hasattr(latest, "success") else latest.get("success", False)
        if succeeded:
            return "continue"

    return "continue" if extracted_payment else "end"


def should_continue_after_detection(state) -> str:
    """
    After detection, always proceed to keyer_review (interrupt point).
    Only hard-stop on detection errors — a doc with no sigs is still valid.
    """
    if isinstance(state, dict):
        detection_errors = state.get("detection_errors", [])
    else:
        detection_errors = state.detection_errors

    return "end" if detection_errors else "continue"


# ============================================
# HITL Node Bodies
# (LangGraph interrupts BEFORE these nodes;
#  each body runs AFTER the human resumes)
# ============================================

def keyer_review_node(state) -> dict:
    """
    Keyer HITL checkpoint body — runs after Keyer clicks Proceed.

    The interrupt fires BEFORE this node. The /resume call passes
    field corrections and feedback via human_modifications.
    Sets current_step so api-service transitions doc to AUTHENTICATED.
    """
    # LangGraph passes an AgentState Pydantic model, not a plain dict
    s = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    s["current_step"] = "keyer_review_done"
    s["awaiting_approval"] = False
    s.setdefault("history", []).append({
        "timestamp": datetime.utcnow().isoformat(),
        "step": "keyer_review",
        "action": "keyer_approved",
        "data": {"edited_fields": list(s.get("human_modifications", {}).keys())},
        "agent": "keyer",
        "notes": "Keyer reviewed extraction and clicked Proceed"
    })
    return s


def auth_review_node(state) -> dict:
    """
    Authenticator HITL checkpoint body — runs after Authenticator clicks Proceed.

    Sets current_step so api-service transitions doc to VERIFIED.
    """
    s = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    s["current_step"] = "auth_review_done"
    s["awaiting_approval"] = False
    s.setdefault("history", []).append({
        "timestamp": datetime.utcnow().isoformat(),
        "step": "auth_review",
        "action": "authenticator_approved",
        "data": s.get("human_modifications", {}),
        "agent": "authenticator",
        "notes": "Authenticator completed validation and clicked Proceed"
    })
    return s


def verifier_review_node(state) -> dict:
    """
    Verifier HITL checkpoint body — runs after Verifier clicks Accept or Reject.

    Reads human_modifications.decision ("accept" | "reject").
    Sets current_step so the completion_node and api-service know the final decision.
    """
    s = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    human_mods = s.get("human_modifications", {})
    decision = human_mods.get("decision", "accept")
    s["current_step"] = f"verifier_{decision}"   # "verifier_accept" or "verifier_reject"
    s["awaiting_approval"] = False
    s.setdefault("history", []).append({
        "timestamp": datetime.utcnow().isoformat(),
        "step": "verifier_review",
        "action": f"verifier_{decision}",
        "data": {"decision": decision, "feedback": human_mods.get("feedback", "")},
        "agent": "verifier",
        "notes": f"Verifier {'accepted' if decision == 'accept' else 'rejected'} the document"
    })
    return s


def completion_node(state) -> dict:
    """
    Final node — marks workflow complete (accepted) or rejected.
    Runs synchronously after verifier resumes.
    """
    s = state.model_dump() if hasattr(state, "model_dump") else dict(state)
    # Inherit decision from verifier_review_node
    prev_step = s.get("current_step", "")
    if "reject" in prev_step:
        s["current_step"] = "rejected"
        notes = "Verifier rejected — document returned to Keyer queue"
    else:
        s["current_step"] = "complete"
        notes = "Workflow complete — document confirmed and dispatched"
    s.setdefault("history", []).append({
        "timestamp": datetime.utcnow().isoformat(),
        "step": "completion",
        "action": s["current_step"],
        "data": {},
        "agent": "system",
        "notes": notes
    })
    return s


# ============================================
# Checkpointer DB Path Helper
# ============================================

def get_checkpoint_db_path() -> str:
    """Return the absolute path to the SQLite checkpoints DB file."""
    default_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = Path(os.environ.get("DATA_DIR", str(default_data_dir)))
    data_dir.mkdir(parents=True, exist_ok=True)
    return str(data_dir / "checkpoints.db")


# ============================================
# Graph Construction
# ============================================

def create_workflow(
    config: AppConfig,
    checkpointer: Optional[BaseCheckpointSaver] = None,
) -> Tuple[StateGraph, BaseCheckpointSaver]:
    """
    Build and compile the LangGraph workflow.

    checkpointer should be an AsyncSqliteSaver created and managed
    by the caller's async lifespan context. Falls back to MemorySaver
    if none is provided (loses state on restart).

    Graph:
      extraction -> detection
        -> [INTERRUPT keyer_review]     (Keyer edits fields, clicks Proceed)
        -> [INTERRUPT auth_review]      (Authenticator authenticates per tile, clicks Proceed)
        -> [INTERRUPT verifier_review]  (Verifier reads all, clicks Accept/Reject)
        -> completion -> END
    Signature authentication is manual: Authenticator calls /run/verify per tile,
    which stores results back into the checkpoint via aupdate_state.
    """
    if checkpointer is None:
        print("⚠️  No checkpointer provided — falling back to MemorySaver (state lost on restart)")
        checkpointer = MemorySaver()

    async def run_extraction_node(state):
        return await extraction_node_wrapper(state, config)

    async def run_detection_node(state):
        return await signature_detection_wrapper(state, config)

    graph = StateGraph(AgentState)

    graph.add_node("extraction",          run_extraction_node)
    graph.add_node("signature_detection", run_detection_node)
    graph.add_node("keyer_review",        keyer_review_node)
    graph.add_node("auth_review",         auth_review_node)
    graph.add_node("verifier_review",     verifier_review_node)
    graph.add_node("completion",          completion_node)

    graph.set_entry_point("extraction")

    graph.add_conditional_edges(
        "extraction",
        should_continue_after_extraction,
        {"continue": "signature_detection", "end": END}
    )
    graph.add_conditional_edges(
        "signature_detection",
        should_continue_after_detection,
        {"continue": "keyer_review", "end": END}
    )

    graph.add_edge("keyer_review",    "auth_review")
    graph.add_edge("auth_review",     "verifier_review")
    graph.add_edge("verifier_review", "completion")
    graph.add_edge("completion",      END)

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["keyer_review", "auth_review", "verifier_review"]
    )

    return compiled, checkpointer


# ============================================
# Node Wrappers  (dict <-> AgentState bridge)
# ============================================

def _ensure_agent_state(state: dict) -> AgentState:
    """Safely coerce a raw LangGraph state dict into an AgentState."""
    state.setdefault("extraction_attempts", [])
    state.setdefault("detection_attempts", [])
    state.setdefault("verification_attempts", [])
    state.setdefault("history", [])
    state.setdefault("human_modifications", {})
    return AgentState(**state)


async def extraction_node_wrapper(state: dict, config: AppConfig) -> dict:
    agent_state = _ensure_agent_state(state) if isinstance(state, dict) else state
    result = await extraction_node(agent_state, config)
    return result.model_dump()


async def signature_detection_wrapper(state: dict, config: AppConfig) -> dict:
    agent_state = _ensure_agent_state(state) if isinstance(state, dict) else state
    result = await signature_detection_node(agent_state, config)
    return result.model_dump()


# ============================================
# Workflow Execution Helpers
# ============================================

async def run_workflow(
    workflow,
    initial_state: AgentState,
    thread_id: Optional[str] = None,
    **kwargs  # absorb legacy run_extraction / run_signature flags
) -> AgentState:
    """
    Start a new workflow run from the beginning.
    Returns after the first HITL interrupt (keyer_review) with awaiting_approval=True.
    """
    initial_state.started_at = datetime.utcnow()
    initial_state.awaiting_approval = True
    lg_config = {"configurable": {"thread_id": thread_id}} if thread_id else {}

    result = await workflow.ainvoke(initial_state.model_dump(), lg_config)

    final_state = AgentState(**result)
    if final_state.current_step == "complete":
        final_state.completed_at = datetime.utcnow()

    return final_state


async def resume_workflow(
    workflow,
    thread_id: str,
    updated_state: Optional[dict] = None
) -> AgentState:
    """
    Resume a paused workflow after a HITL checkpoint.

    updated_state carries the human changes (edited fields, feedback).
    The graph runs until the next interrupt or END.

    Keyer Proceed  -> resumes past keyer_review, pauses at auth_review
    Auth Proceed   -> resumes past auth_review, runs completion, reaches END
    """
    lg_config = {"configurable": {"thread_id": thread_id}}

    if updated_state:
        await workflow.aupdate_state(lg_config, updated_state)

    result = await workflow.ainvoke(None, lg_config)

    final_state = AgentState(**result)
    if final_state.current_step == "complete":
        final_state.completed_at = datetime.utcnow()

    return final_state


async def get_workflow_state(workflow, thread_id: str) -> Optional[dict]:
    """Return the current checkpointed state dict for a thread, or None."""
    lg_config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = await workflow.aget_state(lg_config)
        return snapshot.values if snapshot else None
    except Exception:
        return None
