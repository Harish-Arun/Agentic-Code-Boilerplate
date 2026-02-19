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

def _load_reference_blobs() -> list:
    """
    Load Base64-encoded signature blobs from data/reference/ref_*.txt files.
    Files are sorted by name, so ref_1.txt -> signatories[0], ref_2.txt -> signatories[1], etc.
    Returns an empty list if the directory or files are not found.
    """
    default_data_dir = Path(__file__).resolve().parent.parent.parent / "data"
    data_dir = Path(os.environ.get("DATA_DIR", str(default_data_dir)))
    ref_dir = data_dir / "reference"

    if not ref_dir.exists():
        return []

    blobs = []
    for txt_file in sorted(ref_dir.glob("ref_*.txt")):
        content = txt_file.read_text(encoding="utf-8").strip()
        if content:
            blobs.append(content)
    return blobs


@router.post("/isv/lookup", response_model=ISVLookupResponse)
async def isv_lookup(body: ISVLookupRequest):
    """
    ISV (Image Signature Verification) stub endpoint.

    Accepts account_number and sort_code, returns mock reference signatures.
    Signature blobs are loaded dynamically from data/reference/ref_*.txt files —
    add ref_1.txt, ref_2.txt, etc. to register new signatories without code changes.
    In production, this would call the real ISV API.
    """
    ref_blobs = _load_reference_blobs()

    # Stub signatory metadata — extend this list to match the number of ref_*.txt files
    _STUB_SIGNATORIES = [
        {"sigId": "SIG001", "signerName": "MS VALERIE HANDO", "signerRole": "AUTHORISED_SIGNATORY", "groupCode": "A"},
        {"sigId": "SIG002", "signerName": "MR TONY MURPHY",   "signerRole": "AUTHORISED_SIGNATORY", "groupCode": "A"},
    ]

    signatories = []
    for i, blob in enumerate(ref_blobs):
        meta = _STUB_SIGNATORIES[i] if i < len(_STUB_SIGNATORIES) else {
            "sigId": f"SIG{i + 1:03d}",
            "signerName": f"SIGNATORY {i + 1}",
            "signerRole": "AUTHORISED_SIGNATORY",
            "groupCode": "A",
        }
        signatories.append(
            ISVSignatory(
                sigId=meta["sigId"],
                signerName=meta["signerName"],
                signatureGif=blob,
                signatureStatus="ACTIVE",
                signerRole=meta["signerRole"],
                groupCode=meta["groupCode"],
                desCode=0,
                verifiedFlag="1",
                statusFlag="1",
            )
        )

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
        signatories=signatories,
    )
