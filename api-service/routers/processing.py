"""
Processing Router - Trigger and resume agent workflows.
"""
import httpx
import json
import os
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field

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


class ResumeDocumentRequest(BaseModel):
    """
    Resume a paused workflow at the next HITL checkpoint.

    human_role    : "keyer" | "authenticator"
    modifications : changes the human made (edited fields, feedback, etc.)
    """
    document_id: str
    human_role: str = "keyer"
    modifications: Dict[str, Any] = Field(default_factory=dict)


class ProcessingStatusResponse(BaseModel):
    document_id: str
    status: str
    current_step: Optional[str] = None
    progress_percent: int = 0
    message: str = ""


class VerifySignatureRequest(BaseModel):
    """Manual signature authentication request from the Authenticator UI."""
    document_id: str
    signature_index: int = 0
    reference_blob: str          # base64 ISV reference image
    reference_id: str            # ISV sigId
    reference_mime_type: str = "image/png"
    reference_signer_name: str = ""  # Human-readable name from ISV (for verifier display)


# ============================================
# Background Processing Helpers
# ============================================

async def process_document_async(document_id: str, document_path: str, db):
    """
    Background task: call agents /run to start the workflow.
    Workflow pauses at keyer_review — we store status as EXTRACTED.
    """
    try:
        await db.update_document(document_id, {"status": "PROCESSING"})

        agents_url = os.environ.get("AGENTS_SERVICE_URL", "http://localhost:8001")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{agents_url}/run",
                    json={
                        "document_id": document_id,
                        "document_path": document_path,
                    }
                )

                if response.status_code != 200:
                    await db.update_document(document_id, {"status": "INGESTED"})
                    raise RuntimeError(
                        f"Agents service returned {response.status_code}: {response.text[:1000]}"
                    )

                result = response.json()
                print("\n" + "=" * 80)
                print("📥 API SERVICE - /run result")
                print(json.dumps(result, indent=2, default=str))
                print("=" * 80 + "\n")

                await db.update_document(document_id, {
                    "status": result.get("status", "PENDING_KEYER"),
                    "extracted_data": result.get("extracted_data", {}),
                    "signature_result": result.get("signature_result", {}),
                    "thinking_traces": result.get("thinking_traces", []),
                })

        except httpx.RequestError as conn_err:
            await db.update_document(document_id, {"status": "INGESTED"})
            raise RuntimeError(f"Agents service unavailable at {agents_url}: {conn_err}")

    except Exception:
        await db.update_document(document_id, {"status": "INGESTED"})
        raise


async def resume_document_async(
    document_id: str,
    human_role: str,
    modifications: Dict[str, Any],
    db,
):
    """
    Background task: call agents /resume to pass a HITL checkpoint.

    keyer resume         → pauses at auth_review        → PENDING_AUTH
    authenticator resume → pauses at verifier_review    → PENDING_VERIFIER
    verifier resume      → completion → END
        decision=accept  → CONFIRMED
        decision=reject  → PENDING_KEYER (back to Keyer queue with "Returned" badge)
    """
    try:
        thread_id = f"doc_{document_id}"
        agents_url = os.environ.get("AGENTS_SERVICE_URL", "http://localhost:8001")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{agents_url}/resume",
                    json={
                        "thread_id": thread_id,
                        "human_role": human_role,
                        "modifications": modifications,
                    }
                )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"Agents /resume returned {response.status_code}: {response.text[:1000]}"
                    )

                result = response.json()
                print(f"\n📥 API SERVICE - /resume result | role={human_role} status={result.get('status')}\n")

                # Determine target status
                if human_role == "keyer":
                    target_status = "PENDING_AUTH"
                elif human_role == "authenticator":
                    target_status = "PENDING_VERIFIER"
                elif human_role == "verifier":
                    decision = modifications.get("decision", "accept")
                    target_status = "CONFIRMED" if decision == "accept" else "PENDING_KEYER"
                else:
                    target_status = result.get("status", "PENDING_VERIFIER")

                # Build DB update
                new_extracted = result.get("extracted_data")
                update: Dict[str, Any] = {
                    "status": target_status,
                    "thinking_traces": result.get("thinking_traces", []),
                }
                if new_extracted:
                    update["extracted_data"] = new_extracted

                # IMPORTANT: For authenticator resume, do NOT overwrite signature_result.
                # The manual per-tile results were already written to DB by /process/verify.
                # The signature_result coming back from /resume is built from the LangGraph
                # checkpoint (old auto-verification data) and would stomp on the manual results.
                # For keyer and verifier resumes it is safe to update signature_result since
                # verifier may change the outcome, and keyer doesn't have sig results yet.
                if human_role != "authenticator":
                    update["signature_result"] = result.get("signature_result", {})

                # Preserve "returned from verifier" badge for rejected docs
                if human_role == "verifier" and target_status == "PENDING_KEYER":
                    update["feedback"] = {
                        **(result.get("feedback") or {}),
                        "returned_from": "verifier",
                        "verifier_feedback": modifications.get("feedback", ""),
                    }

                await db.update_document(document_id, update)

        except httpx.RequestError as conn_err:
            raise RuntimeError(f"Agents service unavailable at {agents_url}: {conn_err}")

    except Exception:
        raise


# ============================================
# Endpoints
# ============================================

@router.post("/document", response_model=ProcessingStatusResponse)
async def process_document(
    request: Request,
    background_tasks: BackgroundTasks,
    process_request: ProcessDocumentRequest,
):
    """Start a new workflow run (upload → EXTRACTED, paused at keyer_review)."""
    db = request.app.state.db

    doc = await db.get_document(process_request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {process_request.document_id} not found")

    if doc.get("status") == "PROCESSING":
        return ProcessingStatusResponse(
            document_id=process_request.document_id,
            status="PROCESSING",
            current_step="in_progress",
            progress_percent=50,
            message="Document is already being processed"
        )

    default_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = os.environ.get("DATA_DIR", str(default_data_dir))
    document_path = doc.get("raw_file_path") or f"{Path(data_dir).as_posix()}/uploads/sample_document.pdf"

    background_tasks.add_task(process_document_async, process_request.document_id, document_path, db)

    return ProcessingStatusResponse(
        document_id=process_request.document_id,
        status="PROCESSING",
        current_step="started",
        progress_percent=0,
        message="Processing started"
    )


@router.post("/resume", response_model=ProcessingStatusResponse)
async def resume_document(
    request: Request,
    background_tasks: BackgroundTasks,
    resume_request: ResumeDocumentRequest,
):
    """
    Resume a paused workflow after a human HITL checkpoint.

    human_role="keyer"         → doc moves to AUTHENTICATED (pauses at auth_review)
    human_role="authenticator" → doc moves to VERIFIED      (pauses at verifier_review)
    human_role="verifier"      → doc moves to CONFIRMED or EXTRACTED (based on decision)
    """
    db = request.app.state.db

    doc = await db.get_document(resume_request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {resume_request.document_id} not found")

    background_tasks.add_task(
        resume_document_async,
        resume_request.document_id,
        resume_request.human_role,
        resume_request.modifications,
        db,
    )

    role = resume_request.human_role
    if role == "keyer":
        next_status = "PENDING_AUTH"
    elif role == "authenticator":
        next_status = "PENDING_VERIFIER"
    elif role == "verifier":
        decision = resume_request.modifications.get("decision", "accept")
        next_status = "CONFIRMED" if decision == "accept" else "PENDING_KEYER"
    else:
        next_status = "PENDING_VERIFIER"

    return ProcessingStatusResponse(
        document_id=resume_request.document_id,
        status="PROCESSING",
        current_step=f"resuming_{role}",
        progress_percent=50,
        message=f"Resuming workflow after {role} → {next_status}"
    )


@router.get("/status/{document_id}", response_model=ProcessingStatusResponse)
async def get_processing_status(request: Request, document_id: str):
    """Get current processing status for a document."""
    db = request.app.state.db

    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    status = doc.get("status", "UNKNOWN")
    progress_map = {
        "INGESTED": 0, "PROCESSING": 50,
        "PENDING_KEYER": 60, "PENDING_AUTH": 75, "PENDING_VERIFIER": 90,
        "CONFIRMED": 100, "REJECTED": 100,
        # legacy
        "EXTRACTED": 60, "AUTHENTICATED": 75, "VERIFIED": 90,
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
):
    """Re-start the full workflow from scratch (resets thread state)."""
    db = request.app.state.db

    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    default_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = os.environ.get("DATA_DIR", str(default_data_dir))
    document_path = doc.get("raw_file_path") or f"{Path(data_dir).as_posix()}/uploads/sample_document.pdf"

    background_tasks.add_task(process_document_async, document_id, document_path, db)

    return {"document_id": document_id, "message": "Re-running full workflow", "status": "PROCESSING"}


@router.post("/verify")
async def verify_signature(
    request: Request,
    verify_request: VerifySignatureRequest,
):
    """
    Manually authenticate a single detected signature against a selected ISV reference.

    Called by the Authenticator phase after picking a reference signatory from ISV.
    Forwards the request to the agents service which runs the AI pipeline
    (Gemini M1-M7 + FIV 1.0 scoring) and returns the full result.

    The document's signature_result in the DB is updated with the new authentication data.
    """
    db = request.app.state.db
    agents_url = os.environ.get("AGENTS_SERVICE_URL", "http://localhost:8001")

    # Resolve the detected signature blob from the DB — same as how /resume gets
    # extracted_data without reading the checkpoint. Agents only needs to run the AI.
    doc = await db.get_document(verify_request.document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {verify_request.document_id} not found")

    sig_result = doc.get("signature_result", {})
    detections = sig_result.get("detections", [])
    idx = verify_request.signature_index

    if idx >= len(detections):
        raise HTTPException(
            status_code=400,
            detail=f"Signature index {idx} out of range — only {len(detections)} signature(s) detected"
        )

    det = detections[idx]
    signature_blob = det.get("image_blob") or det.get("signature_blob")
    sig_mime_type = det.get("blob_mime_type") or "image/png"

    if not signature_blob:
        raise HTTPException(
            status_code=400,
            detail=f"No image blob for signature {idx}. Re-process the document first."
        )

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{agents_url}/run/verify",
                json={
                    "document_id": verify_request.document_id,
                    "signature_index": idx,
                    "signature_blob": signature_blob,
                    "signature_mime_type": sig_mime_type,
                    "reference_blob": verify_request.reference_blob,
                    "reference_id": verify_request.reference_id,
                    "reference_mime_type": verify_request.reference_mime_type,
                }
            )

            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Agents service error: {response.text[:500]}"
                )

            result = response.json()
            print(f"\n📥 API SERVICE - /verify result | sig={idx} match={result.get('match')}\n")

            # Attach the signer name (only known on frontend/api-service side, not agents)
            if verify_request.reference_signer_name:
                result["reference_signer_name"] = verify_request.reference_signer_name

            # Merge result into the document's signature_result in DB.
            # Store in BOTH all_verifications (positional, for display) AND
            # manual_verification_results (keyed by index, for verifier read-back).
            existing_sig_result = doc.get("signature_result", {})

            all_verifications = existing_sig_result.get("all_verifications", [])
            while len(all_verifications) <= idx:
                all_verifications.append({})
            all_verifications[idx] = result
            existing_sig_result["all_verifications"] = all_verifications

            manual_vr = existing_sig_result.get("manual_verification_results", {})
            manual_vr[str(idx)] = result
            existing_sig_result["manual_verification_results"] = manual_vr

            await db.update_document(verify_request.document_id, {
                "signature_result": existing_sig_result
            })

            return result

    except HTTPException:
        raise
    except httpx.RequestError as conn_err:
        raise HTTPException(
            status_code=503,
            detail=f"Agents service unavailable: {conn_err}"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{document_id}")
async def get_processing_history(request: Request, document_id: str):
    """Return workflow state from agents service (or DB fallback)."""
    db = request.app.state.db

    doc = await db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

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
                    "current_step": state_data.get("current_step", "unknown"),
                }
    except httpx.RequestError:
        pass

    return {
        "document_id": document_id,
        "thread_id": thread_id,
        "workflow_state": {
            "status": doc.get("status"),
            "extracted_data": doc.get("extracted_data", {}),
            "signature_result": doc.get("signature_result", {}),
        },
        "is_paused": False,
        "current_step": doc.get("status", "unknown").lower(),
    }


