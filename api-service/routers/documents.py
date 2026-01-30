"""
Documents Router - CRUD operations for documents.
"""
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from pydantic import BaseModel

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from models import Document, DocumentCreate, DocumentUpdate, DocumentStatus


router = APIRouter()


# ============================================
# Response Models
# ============================================
class DocumentListResponse(BaseModel):
    documents: List[Document]
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
    
    return DocumentListResponse(
        documents=[Document(**doc) for doc in docs],
        total=len(docs),  # In production, get actual count
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
    
    In production, this would save the file to object storage.
    For the boilerplate, we mock the file handling.
    """
    db = request.app.state.db
    
    # Mock file storage - in production, save to S3/GCS/Azure Blob
    mock_file_path = f"/data/uploads/{file.filename}"
    
    doc_data = {
        "source": source,
        "uploaded_by": uploaded_by,
        "status": "INGESTED",
        "raw_file_path": mock_file_path,
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
        updates["status"] = doc_update.status.value
    if doc_update.extracted_data:
        updates["extracted_data"] = doc_update.extracted_data
    if doc_update.signature_result:
        updates["signature_result"] = doc_update.signature_result
    
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
