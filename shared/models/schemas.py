"""
Shared Pydantic Models - Data schemas used across all services.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
import uuid


# ============================================
# Document Status Enum
# ============================================
class DocumentStatus(str, Enum):
    INGESTED = "INGESTED"
    PROCESSING = "PROCESSING"
    EXTRACTED = "EXTRACTED"
    VERIFIED = "VERIFIED"
    REVIEWED = "REVIEWED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


# ============================================
# Document Models
# ============================================
class DocumentCreate(BaseModel):
    """Request model for creating a new document."""
    source: str = "manual"
    uploaded_by: str = "system"
    raw_file_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentUpdate(BaseModel):
    """Request model for updating a document."""
    status: Optional[DocumentStatus] = None
    extracted_data: Optional[Dict[str, Any]] = None
    signature_result: Optional[Dict[str, Any]] = None


class Document(BaseModel):
    """Full document model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str
    uploaded_by: str
    status: DocumentStatus = DocumentStatus.INGESTED
    raw_file_path: Optional[str] = None
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    signature_result: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ============================================
# Extraction Models
# ============================================
class PaymentField(BaseModel):
    """A single extracted field with confidence."""
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = "ai"  # ai, ocr, manual


class ExtractedPayment(BaseModel):
    """Extracted payment fields from a document."""
    creditor_name: Optional[PaymentField] = None
    creditor_account: Optional[PaymentField] = None
    debtor_name: Optional[PaymentField] = None
    debtor_account: Optional[PaymentField] = None
    amount: Optional[PaymentField] = None
    currency: Optional[PaymentField] = None
    payment_type: Optional[PaymentField] = None
    payment_date: Optional[PaymentField] = None
    charges_account: Optional[PaymentField] = None
    raw_ocr_text: Optional[str] = None


# ============================================
# Signature Models
# ============================================
class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 1


class SignatureDetection(BaseModel):
    """Detected signature region."""
    bounding_box: BoundingBox
    confidence: float
    cropped_image_path: Optional[str] = None


class SignatureVerification(BaseModel):
    """Signature verification result."""
    match: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    reference_signature_id: Optional[str] = None


# ============================================
# Processing Models
# ============================================
class ProcessingRequest(BaseModel):
    """Request to process a document."""
    document_id: str
    run_extraction: bool = True
    run_signature_verification: bool = True


class ProcessingResult(BaseModel):
    """Result from document processing."""
    document_id: str
    status: DocumentStatus
    extracted_data: Optional[ExtractedPayment] = None
    signature_detections: List[SignatureDetection] = Field(default_factory=list)
    signature_verification: Optional[SignatureVerification] = None
    processing_time_ms: int = 0
    errors: List[str] = Field(default_factory=list)


# ============================================
# Agent State Models (LangGraph)
# ============================================
class AgentState(BaseModel):
    """Shared state passed between LangGraph nodes."""
    document_id: str
    document_path: str
    current_step: str = "start"
    
    # Human-in-the-loop tracking
    awaiting_approval: bool = False
    
    # Extraction results
    extracted_payment: Optional[ExtractedPayment] = None
    extraction_errors: List[str] = Field(default_factory=list)
    
    # Signature detection results
    signature_detections: List[SignatureDetection] = Field(default_factory=list)
    detection_errors: List[str] = Field(default_factory=list)
    
    # Verification results
    verification_result: Optional[SignatureVerification] = None
    verification_errors: List[str] = Field(default_factory=list)
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    
    # Metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


# ============================================
# API Response Models
# ============================================
class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    service: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response format."""
    error: str
    detail: Optional[str] = None
    code: str = "INTERNAL_ERROR"
