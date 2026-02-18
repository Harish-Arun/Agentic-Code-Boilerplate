"""
Documents Router - CRUD operations for documents.
"""
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from models import Document, DocumentCreate, DocumentUpdate, DocumentStatus, OperationEntry, ISVLookupRequest, ISVLookupResponse, ISVSignatory


router = APIRouter()


ALLOWED_STATUS_TRANSITIONS = {
    "INGESTED": {"PROCESSING", "REJECTED"},
    "PROCESSING": {"EXTRACTED", "REJECTED", "INGESTED"},
    "EXTRACTED": {"PROCESSING", "AUTHENTICATED", "REJECTED"},       # Keyer -> Authenticator
    "AUTHENTICATED": {"VERIFIED", "EXTRACTED", "REJECTED"},          # Authenticator -> Verifier (or back to Keyer)
    "VERIFIED": {"CONFIRMED", "EXTRACTED", "REJECTED"},              # Verifier -> Done (or back to Keyer)
    "CONFIRMED": {"DISPATCHED"},
    "APPROVED": {"DISPATCHED"},
    "DISPATCHED": set(),
    "REJECTED": {"EXTRACTED"},                                       # Allow re-opening rejected docs
    # Legacy states kept for backward compatibility
    "AWAITING_APPROVAL": {"REVIEW_PENDING", "REVIEWED", "CONFIRMED", "APPROVED", "REJECTED"},
    "REVIEW_PENDING": {"REVIEWED", "CONFIRMED", "APPROVED", "REJECTED"},
    "REVIEWED": {"CONFIRMED", "APPROVED", "REJECTED"},
}


def _validate_status_transition(current_status: str, next_status: str) -> None:
    if current_status == next_status:
        return

    allowed_next = ALLOWED_STATUS_TRANSITIONS.get(current_status, set())
    if next_status not in allowed_next:
        allowed_str = ", ".join(sorted(allowed_next)) if allowed_next else "<terminal>"
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid status transition: {current_status} -> {next_status}. "
                f"Allowed next states: {allowed_str}"
            )
        )


# ============================================
# Response Models
# ============================================
class DocumentListResponse(BaseModel):
    documents: List[Document]
    total: int
    limit: int
    offset: int


class StatusHistoryEntry(BaseModel):
    document_id: str
    from_status: Optional[str] = None
    to_status: str
    changed_at: datetime
    changed_by: Optional[str] = None
    reason: Optional[str] = None


class StatusHistoryResponse(BaseModel):
    document_id: str
    history: List[StatusHistoryEntry]
    total: int
    limit: int
    offset: int


class OperationResponse(BaseModel):
    operation: Optional[OperationEntry]


class OperationListResponse(BaseModel):
    operations: List[OperationEntry]
    total: int
    limit: int
    offset: int


# ============================================
# Endpoints
# ============================================
@router.get("", response_model=DocumentListResponse)
async def list_documents(
    request: Request,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    List all documents with optional status filtering.
    
    - **status**: Filter by document status (INGESTED, PROCESSING, etc.)
    - **limit**: Maximum documents to return (default 100)
    - **offset**: Pagination offset
    """
    db = request.app.state.db
    docs = await db.list_documents(status=status, limit=limit, offset=offset)
    total = await db.count_documents(status=status)
    
    return DocumentListResponse(
        documents=[Document(**doc) for doc in docs],
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/{document_id}", response_model=Document)
async def get_document(request: Request, document_id: str):
    """Get a single document by ID."""
    db = request.app.state.db
    doc = await db.get_document(document_id)
    
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    return Document(**doc)


@router.post("", response_model=Document, status_code=201)
async def create_document(
    request: Request,
    doc_create: DocumentCreate
):
    """
    Create a new document record.
    
    This is the ingestion endpoint - call this when a new document enters the system.
    """
    db = request.app.state.db
    
    doc_data = {
        "source": doc_create.source,
        "uploaded_by": doc_create.uploaded_by,
        "status": "INGESTED",
        "raw_file_path": doc_create.raw_file_path or "",
        "extracted_data": {},
        "signature_result": {}
    }
    
    doc_id = await db.create_document(doc_data)
    created_doc = await db.get_document(doc_id)
    
    return Document(**created_doc)


@router.post("/upload", response_model=Document, status_code=201)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    source: str = "manual",
    uploaded_by: str = "system"
):
    """
    Upload a document file.
    
    Saves the file to local storage. In production, this would save to S3/GCS/Azure Blob.
    """
    import aiofiles
    
    db = request.app.state.db
    
    # Ensure uploads directory exists
    default_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = Path(os.environ.get("DATA_DIR", str(default_data_dir)))
    upload_dir = data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the uploaded file
    file_path = upload_dir / file.filename
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Store as a stable path within the mounted /data volume
    saved_file_path = f"{data_dir.as_posix()}/uploads/{file.filename}"
    
    doc_data = {
        "source": source,
        "uploaded_by": uploaded_by,
        "status": "INGESTED",
        "raw_file_path": saved_file_path,
        "extracted_data": {},
        "signature_result": {}
    }
    
    doc_id = await db.create_document(doc_data)
    created_doc = await db.get_document(doc_id)
    
    return Document(**created_doc)


@router.patch("/{document_id}", response_model=Document)
async def update_document(
    request: Request,
    document_id: str,
    doc_update: DocumentUpdate
):
    """
    Update a document's status or data.
    
    Use this for:
    - Transitioning document states
    - Saving extracted data
    - Recording signature verification results
    """
    db = request.app.state.db
    
    # Verify document exists
    existing = await db.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    # Build updates
    updates = {}
    if doc_update.status:
        _validate_status_transition(existing.get("status", "INGESTED"), doc_update.status.value)
        updates["status"] = doc_update.status.value
    if doc_update.extracted_data:
        updates["extracted_data"] = doc_update.extracted_data
    if doc_update.signature_result:
        updates["signature_result"] = doc_update.signature_result
    if doc_update.thinking_traces is not None:
        updates["thinking_traces"] = doc_update.thinking_traces
    if doc_update.authentication_result is not None:
        updates["authentication_result"] = doc_update.authentication_result
    if doc_update.feedback is not None:
        updates["feedback"] = doc_update.feedback

    if updates:
        await db.update_document(document_id, updates)
    
    updated_doc = await db.get_document(document_id)
    return Document(**updated_doc)


@router.patch("/{document_id}/status")
async def update_document_status(
    request: Request,
    document_id: str,
    status: DocumentStatus
):
    """
    Quick endpoint to update just the document status.
    
    Valid transitions:
    - INGESTED → PROCESSING
    - PROCESSING → EXTRACTED
    - EXTRACTED → VERIFIED
    - VERIFIED → REVIEWED
    - REVIEWED → CONFIRMED or REJECTED
    """
    db = request.app.state.db
    
    existing = await db.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    _validate_status_transition(existing.get("status", "INGESTED"), status.value)
    
    await db.update_document(document_id, {"status": status.value})
    
    return {"document_id": document_id, "status": status.value, "message": "Status updated"}


@router.delete("/{document_id}", status_code=204)
async def delete_document(request: Request, document_id: str):
    """Delete a document."""
    db = request.app.state.db
    
    existing = await db.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    
    await db.delete_document(document_id)
    return None


@router.get("/{document_id}/status-history", response_model=StatusHistoryResponse)
async def get_document_status_history(
    request: Request,
    document_id: str,
    limit: int = 100,
    offset: int = 0
):
    """Get status transition history for a document."""
    db = request.app.state.db

    existing = await db.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    history = await db.get_document_status_history(document_id, limit=limit, offset=offset)
    total = await db.count_document_status_history(document_id)
    return StatusHistoryResponse(
        document_id=document_id,
        history=[StatusHistoryEntry(**entry) for entry in history],
        total=total,
        limit=limit,
        offset=offset
    )


@router.get("/{document_id}/operation", response_model=OperationResponse)
async def get_document_operation(request: Request, document_id: str):
    """Get latest software operation/audit event for a document."""
    db = request.app.state.db

    existing = await db.get_document(document_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    operation = await db.get_operation(document_id)
    if not operation:
        return OperationResponse(operation=None)
    return OperationResponse(operation=OperationEntry(**operation))


@router.get("/operations/list", response_model=OperationListResponse)
async def list_operations(
    request: Request,
    status: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """List software operation/audit events with optional to-status filter."""
    db = request.app.state.db

    operations = await db.list_operations(status=status, limit=limit, offset=offset)
    total = await db.count_operations(status=status)
    return OperationListResponse(
        operations=[OperationEntry(**op) for op in operations],
        total=total,
        limit=limit,
        offset=offset
    )


# ============================================
# ISV Stub Endpoint
# ============================================

# Small base64-encoded 100x40 PNG signature stubs (simple squiggle images)
_STUB_SIGNATURE_1 = (
    "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"
)

# _STUB_SIGNATURE_2 = (
#     "iVBORw0KGgoAAAANSUhEUgAAAGQAAAAoCAYAAAAIeF9DAAAA1klEQVRoge3XMQ7CMABEUZ/k/n"
#     "emooACUVAgiYQTx/a+P5VVuVr9DQAAAAAAAAAAAOBCy7per497X+O47y7Xb/e6dr/e03PY+z7n"
#     "8/Se016/F/uce5/HefjAiIgIESIiESJERCJEiIhEiBARiRAhIhIhQkQkQoSISIQIEZEIESIiES"
#     "JERCJEiIhEiBARiRAhIhIhQkQkQoSISIQIEZEIESIiESJERCJEiIhEiBARiRAhIhIhQkQkQoSI"
#     "SIQIEZEIESIiESJERCJEiIhEiBARiRAhIvIPfQAAAP//AwBYjBmJMzIZnAAAAABJRU5ErkJggg=="
# )


@router.post("/isv/lookup", response_model=ISVLookupResponse)
async def isv_lookup(body: ISVLookupRequest):
    """
    ISV (Image Signature Verification) stub endpoint.

    Accepts account_number and sort_code, returns mock reference signatures.
    In production, this would call the real ISV API.
    """
    return ISVLookupResponse(
        sortCode=body.sort_code,
        accountNumber=body.account_number,
        accountName="CF07F791680453D417F8C6205",
        statusFlag="1",
        accountType="T9",
        accoperator="FAMERGER_W",
        accCreateDate="2020-01-15",
        accUpdateDate="2025-06-20",
        ruleCode="99",
        freeformRule="SST: SIGNATORIES: MS VALERIE HANDO, MR TONY MURPHY. Any one to sign.",
        signRuleCreateDate="2020-01-15",
        signRuleUpdateDate="2025-06-20",
        appName="cn_name.APP",
        signRuleOperator="FAMERGER_W",
        referenceNumber="REFSIG1",
        signatories=[
            ISVSignatory(
                sigId="SIG001",
                signerName="MS VALERIE HANDO",
                signatureGif=_STUB_SIGNATURE_1,
                signatureStatus="ACTIVE",
                signerRole="AUTHORISED_SIGNATORY",
                groupCode="A",
                desCode=0,
                verifiedFlag="1",
                statusFlag="1",
            ),
            # ISVSignatory(
            #     sigId="SIG002",
            #     signerName="MR TONY MURPHY",
            #     signatureGif=_STUB_SIGNATURE_2,
            #     signatureStatus="ACTIVE",
            #     signerRole="AUTHORISED_SIGNATORY",
            #     groupCode="A",
            #     desCode=0,
            #     verifiedFlag="1",
            #     statusFlag="1",
            # ),
        ],
    )
