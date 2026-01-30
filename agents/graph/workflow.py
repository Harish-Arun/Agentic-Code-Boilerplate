"""
LangGraph Workflow Definition - Main processing pipeline.

This defines the state graph that orchestrates all agents:
Start → Extraction → Signature Detection → Verification → End

Features:
- Configurable checkpointing (memory, postgres)
- Human-in-the-loop interrupt points
- Resume workflow capability
"""
import os
from typing import Optional, Tuple
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from models import AgentState
from config import AppConfig

from .nodes.extraction import extraction_node
from .nodes.signature_detection import signature_detection_node
from .nodes.verification import verification_node


def create_checkpointer(config: AppConfig) -> Optional[BaseCheckpointSaver]:
    """
    Create a checkpointer based on configuration.
    
    Supported backends:
    - memory: In-memory (development only, lost on restart)
    - postgres: PostgreSQL (production, persistent)
    """
    if not config.agents.checkpointing.enabled:
        return None
    
    backend = config.agents.checkpointing.backend
    
    if backend == "memory":
        return MemorySaver()
    
    elif backend == "postgres":
        # Production: Use PostgreSQL for persistent checkpoints
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            conn_string = os.environ.get(
                "CHECKPOINT_POSTGRES_CONN", 
                "postgresql://postgres:postgres@localhost:5432/nnp_ai"
            )
            return PostgresSaver.from_conn_string(conn_string)
        except ImportError:
            print("⚠️ langgraph-checkpoint-postgres not installed, falling back to memory")
            return MemorySaver()
    
    # Default to memory
    return MemorySaver()


def create_workflow(config: AppConfig) -> Tuple[StateGraph, Optional[BaseCheckpointSaver]]:
    """
    Create the LangGraph workflow for document processing.
    
    Returns:
        Tuple of (compiled_graph, checkpointer)
    
    The graph structure:
    
    [Start]
       ↓
    [Extraction Agent] ──→ Extract payment fields
       ↓
    [Human Review] ──→ INTERRUPT (if HITL enabled)
       ↓
    [Signature Detection] ──→ Find signature regions
       ↓
    [Verification Agent] ──→ Verify signatures
       ↓
    [End]
    """
    
    # Create the state graph
    graph = StateGraph(AgentState)
    
    # Add nodes based on enabled agents
    enabled_agents = config.agents.enabled
    
    # Always start with extraction if enabled
    if "extraction" in enabled_agents:
        graph.add_node("extraction", lambda state: extraction_node_wrapper(state, config))
    
    # Add human review node if HITL enabled
    if config.features.human_in_loop:
        graph.add_node("human_review", human_review_node)
    
    if "signature_detection" in enabled_agents:
        graph.add_node("signature_detection", lambda state: signature_detection_wrapper(state, config))
    
    if "verification" in enabled_agents:
        graph.add_node("verification", lambda state: verification_wrapper(state, config))
    
    # Define edges based on enabled agents
    if "extraction" in enabled_agents:
        graph.set_entry_point("extraction")
        
        if config.features.human_in_loop:
            # Extraction → Human Review → Signature Detection
            graph.add_edge("extraction", "human_review")
            
            if "signature_detection" in enabled_agents:
                graph.add_edge("human_review", "signature_detection")
                
                if "verification" in enabled_agents:
                    graph.add_edge("signature_detection", "verification")
                    graph.add_edge("verification", END)
                else:
                    graph.add_edge("signature_detection", END)
            else:
                graph.add_edge("human_review", END)
        else:
            # No HITL - direct flow
            if "signature_detection" in enabled_agents:
                graph.add_edge("extraction", "signature_detection")
                
                if "verification" in enabled_agents:
                    graph.add_edge("signature_detection", "verification")
                    graph.add_edge("verification", END)
                else:
                    graph.add_edge("signature_detection", END)
            else:
                graph.add_edge("extraction", END)
                
    elif "signature_detection" in enabled_agents:
        graph.set_entry_point("signature_detection")
        
        if "verification" in enabled_agents:
            graph.add_edge("signature_detection", "verification")
            graph.add_edge("verification", END)
        else:
            graph.add_edge("signature_detection", END)
            
    elif "verification" in enabled_agents:
        graph.set_entry_point("verification")
        graph.add_edge("verification", END)
    
    # Create checkpointer
    checkpointer = create_checkpointer(config)
    
    # Compile with checkpointer and interrupt points
    interrupt_before = []
    if config.features.human_in_loop:
        interrupt_before = ["human_review"]  # Pause before human review
    
    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before
    )
    
    return compiled, checkpointer


def human_review_node(state: dict) -> dict:
    """
    Human review node - this is where the workflow pauses for approval.
    
    When resumed, the workflow continues with the (potentially modified) state.
    """
    state["current_step"] = "human_review"
    state["awaiting_approval"] = True
    return state


# ============================================
# Node Wrappers (sync/async handling)
# ============================================
def extraction_node_wrapper(state: dict, config: AppConfig) -> dict:
    """Sync wrapper for extraction node."""
    import asyncio
    
    # Convert dict to AgentState if needed
    if isinstance(state, dict):
        agent_state = AgentState(**state)
    else:
        agent_state = state
    
    # Run async node
    result = asyncio.get_event_loop().run_until_complete(
        extraction_node(agent_state, config)
    )
    return result.model_dump()


def signature_detection_wrapper(state: dict, config: AppConfig) -> dict:
    """Sync wrapper for signature detection node."""
    import asyncio
    
    if isinstance(state, dict):
        agent_state = AgentState(**state)
    else:
        agent_state = state
    
    result = asyncio.get_event_loop().run_until_complete(
        signature_detection_node(agent_state, config)
    )
    return result.model_dump()


def verification_wrapper(state: dict, config: AppConfig) -> dict:
    """Sync wrapper for verification node."""
    import asyncio
    
    if isinstance(state, dict):
        agent_state = AgentState(**state)
    else:
        agent_state = state
    
    result = asyncio.get_event_loop().run_until_complete(
        verification_node(agent_state, config)
    )
    return result.model_dump()


# ============================================
# Workflow Execution Functions
# ============================================
async def run_workflow(
    workflow, 
    initial_state: AgentState,
    thread_id: Optional[str] = None,
    run_extraction: bool = True,
    run_signature: bool = True
) -> AgentState:
    """
    Execute the workflow with the given initial state.
    
    Args:
        workflow: Compiled LangGraph workflow
        initial_state: Starting state with document info
        thread_id: Unique ID for checkpointing (use document_id)
        run_extraction: Whether to run extraction agent
        run_signature: Whether to run signature agents
    
    Returns:
        Final AgentState with all results
    """
    # Set timestamps
    initial_state.started_at = datetime.utcnow()
    
    # Build config with thread_id for checkpointing
    config = {}
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph
    result = workflow.invoke(initial_state.model_dump(), config)
    
    # Convert back to AgentState
    final_state = AgentState(**result)
    final_state.completed_at = datetime.utcnow()
    
    return final_state


async def resume_workflow(
    workflow,
    thread_id: str,
    updated_state: Optional[dict] = None
) -> AgentState:
    """
    Resume a paused workflow from its last checkpoint.
    
    Args:
        workflow: Compiled LangGraph workflow
        thread_id: The thread_id used when workflow was started
        updated_state: Optional state updates (e.g., after human review)
    
    Returns:
        Final AgentState after resumption
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    # Resume with optional state updates
    input_state = updated_state if updated_state else None
    result = workflow.invoke(input_state, config)
    
    final_state = AgentState(**result)
    final_state.completed_at = datetime.utcnow()
    
    return final_state


def get_workflow_state(workflow, thread_id: str) -> Optional[dict]:
    """
    Get the current state of a workflow by thread_id.
    
    Useful for checking if workflow is paused, getting intermediate results.
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        state = workflow.get_state(config)
        return state.values if state else None
    except Exception:
        return None
