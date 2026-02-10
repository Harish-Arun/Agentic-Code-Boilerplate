"""
Processing Router - Trigger agent pipelines for document processing.
"""
import httpx
import os
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from models import ProcessingRequest, ProcessingResult, DocumentStatus
from config import get_config


router = APIRouter()


# ============================================
# Request/Response Models
# ============================================
class ProcessDocumentRequest(BaseModel):
    document_id: str
    run_extraction: bool = True
    run_signature_verification: bool = True


class ProcessingStatusResponse(BaseModel):
    document_id: str
    status: str
    current_step: Optional[str] = None
    progress_percent: int = 0
    message: str = ""


# ============================================
# Background Processing
# ============================================
async def process_document_async(
    document_id: str,
    document_path: str,
    db,
    run_extraction: bool = True,
    run_signature: bool = True
):
    """
    Background task to process a document through the agent pipeline.
    
    Calls the Agents service which runs the LangGraph workflow.
    """
    try:
        # Update status to PROCESSING
        await db.update_document(document_id, {"status": "PROCESSING"})
        
        # Get agents service URL from environment
        import os
        agents_url = os.environ.get("AGENTS_SERVICE_URL", "http://localhost:8001")
        
        # Try to call agents service (may not be running in dev)
        # Increased timeout to 300s (5 min) for signature verification with Gemini
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{agents_url}/run",
                    json={
                        "document_id": document_id,
                        "document_path": document_path,
                        "run_extraction": run_extraction,
                        "run_signature_verification": run_signature
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # DEBUG: Log response from agents
                    print("\n" + "="*80)
                    print("📥 API SERVICE - Received from Agents")
                    print("="*80)
                    import json
                    print(json.dumps(result, indent=2, default=str))
                    print("="*80 + "\n")
                    
                    await db.update_document(document_id, {
                        "status": result.get("status", "EXTRACTED"),
                        "extracted_data": result.get("extracted_data", {}),
                        "signature_result": result.get("signature_result", {})
                    })
                    return
        except httpx.RequestError as conn_err:
            # Agents service not available — fail with clear error
            await db.update_document(document_id, {
                "status": "INGESTED",  # Reset to allow retry
            })
            raise RuntimeError(
                f"Agents service unavailable at {agents_url}: {conn_err}"
            )
        
    except Exception as e:
        await db.update_document(document_id, {
            "status": "INGESTED",  # Reset to allow retry
        })
        raise


# ============================================
# Endpoints
# ============================================
@router.post("/document", response_model=ProcessingStatusResponse)
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    process_request: ProcessDocumentRequest
):
    """
    Trigger document processing through the agent pipeline.
    
    This is the main entry point for the agentic flow:
    1. Extraction Agent (OCR + field extraction)
    2. Signature Detection Agent
    3. Signature Verification Agent
    
    Processing runs in the background - poll /process/status/{id} for updates.
    """
    db = request.app.state.db
    config = request.app.state.config
    
    # Verify document exists
    doc = await db.get_document(process_request.document_id)
    if not doc:
        raise HTTPException(
            status_code=404, 
            detail=f"Document {process_request.document_id} not found"
        )
    
    # Check if already processing
    if doc.get("status") == "PROCESSING":
        return ProcessingStatusResponse(
            document_id=process_request.document_id,
            status="PROCESSING",
            current_step="in_progress",
            progress_percent=50,
            message="Document is already being processed"
        )
    
    # Start background processing
    default_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = os.environ.get("DATA_DIR", str(default_data_dir))
    document_path = doc.get("raw_file_path") or f"{Path(data_dir).as_posix()}/uploads/sample_document.pdf"
    background_tasks.add_task(
        process_document_async,
        process_request.document_id,
        document_path,
        db,
        process_request.run_extraction,
        process_request.run_signature_verification
    )
    
    return ProcessingStatusResponse(
        document_id=process_request.document_id,
        status="PROCESSING",
        current_step="started",
        progress_percent=0,
        message="Processing started"
    )


@router.get("/status/{document_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(request: Request, document_id: str):
    """
    Get the current processing status of a document.
    
    Poll this endpoint to track progress of document processing.
    """
    db = request.app.state.db
    
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    status = doc.get("status", "UNKNOWN")
    
    # Map status to progress
    progress_map = {
        "INGESTED": 0,
        "PROCESSING": 50,
        "EXTRACTED": 70,
        "VERIFIED": 90,
        "REVIEWED": 95,
        "CONFIRMED": 100,
        "REJECTED": 100
    }
    
    return ProcessingStatusResponse(
        document_id=document_id,
        status=status,
        current_step=status.lower(),
        progress_percent=progress_map.get(status, 0),
        message=f"Document is in {status} state"
    )


@router.post("/rerun/{document_id}")
async def rerun_processing(
    request: Request,
    background_tasks: BackgroundTasks,
    document_id: str,
    step: str = "all"
):
    """
    Re-run processing for a document.
    
    - **step**: Which step to re-run: 'extraction', 'signature', or 'all'
    """
    db = request.app.state.db
    
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    run_extraction = step in ["all", "extraction"]
    run_signature = step in ["all", "signature"]

    default_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = os.environ.get("DATA_DIR", str(default_data_dir))
    document_path = doc.get("raw_file_path") or f"{Path(data_dir).as_posix()}/uploads/sample_document.pdf"
    
    background_tasks.add_task(
        process_document_async,
        document_id,
        document_path,
        db,
        run_extraction,
        run_signature
    )
    
    return {
        "document_id": document_id,
        "message": f"Re-running {step} processing",
        "status": "PROCESSING"
    }


@router.get("/history/{document_id}")
async def get_processing_history(request: Request, document_id: str):
    """
    Get the workflow history for a document.
    
    Returns the full processing history including:
    - All extraction attempts
    - All signature detection attempts  
    - All verification attempts
    - State transitions
    """
    db = request.app.state.db
    
    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    # Try to get workflow state from agents service
    import os
    agents_url = os.environ.get("AGENTS_SERVICE_URL", "http://localhost:8001")
    thread_id = f"doc_{document_id}"
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{agents_url}/status/{thread_id}")
            
            if response.status_code == 200:
                state_data = response.json()
                return {
                    "document_id": document_id,
                    "thread_id": thread_id,
                    "workflow_state": state_data.get("state", {}),
                    "is_paused": state_data.get("is_paused", False),
                    "current_step": state_data.get("current_step", "unknown")
                }
    except httpx.RequestError:
        pass
    
    # Fallback - return document state
    return {
        "document_id": document_id,
        "thread_id": thread_id,
        "workflow_state": {
            "status": doc.get("status"),
            "extracted_data": doc.get("extracted_data", {}),
            "signature_result": doc.get("signature_result", {})
        },
        "is_paused": False,
        "current_step": doc.get("status", "unknown").lower()
    }

