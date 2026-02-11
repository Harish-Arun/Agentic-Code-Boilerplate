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
    location: Optional[str] = None  # Where in document the field was found


class ExtractedPayment(BaseModel):
    """Extracted payment fields from a document."""
    creditor_name: Optional[PaymentField] = None
    creditor_account: Optional[PaymentField] = None
    creditor_bank: Optional[PaymentField] = None
    debtor_name: Optional[PaymentField] = None
    debtor_account: Optional[PaymentField] = None
    debtor_bank: Optional[PaymentField] = None
    amount: Optional[PaymentField] = None
    currency: Optional[PaymentField] = None
    payment_type: Optional[PaymentField] = None
    payment_date: Optional[PaymentField] = None
    charges_account: Optional[PaymentField] = None
    reference: Optional[PaymentField] = None
    raw_ocr_text: Optional[str] = None


# ============================================
# Signature Models
# ============================================
class BoundingBox(BaseModel):
    """Bounding box coordinates (as percentages 0.0-1.0)."""
    x1: float
    y1: float
    x2: float
    y2: float
    page: int = 1


class SignatureDetection(BaseModel):
    """Detected signature region."""
    signature_id: Optional[str] = None
    bounding_box: BoundingBox
    page: int = 0
    signature_type: str = "unknown"  # customer, bank_initiator, bank_authenticator, unknown
    signer_role: Optional[str] = None
    signer_name: Optional[str] = None
    confidence: float
    description: Optional[str] = None
    cropped_image_path: Optional[str] = None
    # Blob storage — base64-encoded binary image data (simulates ISV blob response)
    image_blob: Optional[str] = None        # base64-encoded cropped signature image
    blob_mime_type: str = "image/png"       # MIME type of the blob


class SimilarityFactors(BaseModel):
    """Detailed similarity analysis factors."""
    overall_shape: Dict[str, Any] = Field(default_factory=dict)
    stroke_patterns: Dict[str, Any] = Field(default_factory=dict)
    pressure_consistency: Dict[str, Any] = Field(default_factory=dict)
    unique_characteristics: Dict[str, Any] = Field(default_factory=dict)


class SignatureVerification(BaseModel):
    """Signature verification result."""
    match: bool
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    reference_signature_id: Optional[str] = None
    similarity_factors: Optional[SimilarityFactors] = None
    risk_indicators: List[str] = Field(default_factory=list)
    recommendation: str = "MANUAL_REVIEW"  # APPROVE, REJECT, MANUAL_REVIEW
    # Blob references for audit trail
    signature_blob: Optional[str] = None    # base64 questioned signature used
    reference_blob: Optional[str] = None    # base64 reference signature used
    blob_mime_type: str = "image/png"       # MIME type of the blobs
    # FIV 1.0 detailed scoring (for frontend metrics display)
    metrics: Optional[Dict[str, Any]] = None  # M1-M7 metric breakdown
    scoring_details: Optional[Dict[str, Any]] = None  # FIV scoring details


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
# History Tracking Models (Append-only state)
# ============================================
class StateHistoryEntry(BaseModel):
    """A single entry in the state history."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    step: str
    action: str  # e.g., "extraction_started", "extraction_completed", "retry_1"
    data: Dict[str, Any] = Field(default_factory=dict)
    agent: str = "system"  # Which agent/tool performed this action
    notes: Optional[str] = None


class ExtractionAttempt(BaseModel):
    """Record of a single extraction attempt."""
    attempt_number: int = 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = False
    extracted_payment: Optional[ExtractedPayment] = None
    errors: List[str] = Field(default_factory=list)
    model_used: str = "gemini-3-flash-preview"
    processing_time_ms: int = 0
    raw_response: Optional[str] = None  # For debugging
    
    # Thinking traces (Gemini thinking mode)
    thoughts: Optional[List[str]] = None  # Thought summaries from Gemini
    thoughts_token_count: Optional[int] = None  # Tokens used for thinking
    thinking_budget_used: Optional[int] = None  # Actual thinking budget consumed


class SignatureDetectionAttempt(BaseModel):
    """Record of a single signature detection attempt."""
    attempt_number: int = 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = False
    detections: List[SignatureDetection] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    model_used: str = "gemini-3-flash-preview"
    processing_time_ms: int = 0
    challenger_feedback: Optional[str] = None  # AI challenger notes
    
    # Thinking traces (Gemini thinking mode)
    thoughts: Optional[List[str]] = None  # Thought summaries from Gemini
    thoughts_token_count: Optional[int] = None  # Tokens used for thinking
    thinking_budget_used: Optional[int] = None  # Actual thinking budget consumed


class VerificationAttempt(BaseModel):
    """Record of a single verification attempt."""
    attempt_number: int = 1
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    success: bool = False
    signature_id: Optional[str] = None  # Which signature was verified
    reference_id: Optional[str] = None  # Reference signature used
    result: Optional[SignatureVerification] = None  # Single result (legacy)
    results: List[SignatureVerification] = Field(default_factory=list)  # Multiple results (new)
    errors: List[str] = Field(default_factory=list)
    model_used: str = "gemini-3-flash-preview"
    processing_time_ms: int = 0
    
    # Thinking traces (Gemini thinking mode)
    thoughts: Optional[List[str]] = None  # Thought summaries from Gemini
    thoughts_token_count: Optional[int] = None  # Tokens used for thinking
    thinking_budget_used: Optional[int] = None  # Actual thinking budget consumed


# ============================================
# Agent State Models (LangGraph)
# ============================================
class AgentState(BaseModel):
    """
    Shared state passed between LangGraph nodes.
    
    IMPORTANT: This state uses append-only patterns for history.
    Each step appends to lists rather than overwriting.
    """
    document_id: str
    document_path: str
    current_step: str = "start"
    
    # Human-in-the-loop tracking
    awaiting_approval: bool = False
    human_modifications: Dict[str, Any] = Field(default_factory=dict)
    
    # Extraction results (latest + history)
    extracted_payment: Optional[ExtractedPayment] = None
    extraction_attempts: List[ExtractionAttempt] = Field(default_factory=list)
    extraction_errors: List[str] = Field(default_factory=list)
    
    # Signature detection results (latest + history)
    signature_detections: List[SignatureDetection] = Field(default_factory=list)
    detection_attempts: List[SignatureDetectionAttempt] = Field(default_factory=list)
    detection_errors: List[str] = Field(default_factory=list)
    
    # Verification results (latest + history)  
    verification_result: Optional[SignatureVerification] = None
    verification_attempts: List[VerificationAttempt] = Field(default_factory=list)
    verification_errors: List[str] = Field(default_factory=list)
    
    # Full history log (append-only)
    history: List[StateHistoryEntry] = Field(default_factory=list)
    
    # Retry tracking
    retry_count: int = 0
    max_retries: int = 3
    
    # Metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def add_history(self, step: str, action: str, data: Dict[str, Any] = None, 
                    agent: str = "system", notes: str = None):
        """Helper method to add a history entry."""
        self.history.append(StateHistoryEntry(
            step=step,
            action=action,
            data=data or {},
            agent=agent,
            notes=notes
        ))
    
    def add_extraction_attempt(self, attempt: ExtractionAttempt):
        """Add an extraction attempt to history."""
        self.extraction_attempts.append(attempt)
        if attempt.success and attempt.extracted_payment:
            self.extracted_payment = attempt.extracted_payment
        if attempt.errors:
            self.extraction_errors.extend(attempt.errors)
    
    def add_detection_attempt(self, attempt: SignatureDetectionAttempt):
        """Add a signature detection attempt to history."""
        self.detection_attempts.append(attempt)
        if attempt.success and attempt.detections:
            self.signature_detections = attempt.detections
        if attempt.errors:
            self.detection_errors.extend(attempt.errors)
    
    def add_verification_attempt(self, attempt: VerificationAttempt):
        """Add a verification attempt to history."""
        self.verification_attempts.append(attempt)
        if attempt.success:
            # Handle both single result (legacy) and multiple results (new)
            if attempt.result:
                self.verification_result = attempt.result
            elif attempt.results and len(attempt.results) > 0:
                # For multiple results, store the first one or aggregate
                self.verification_result = attempt.results[0]
        if attempt.errors:
            self.verification_errors.extend(attempt.errors)
    
    def get_latest_detection(self) -> Optional[SignatureDetectionAttempt]:
        """Get the most recent signature detection attempt."""
        if not self.detection_attempts:
            return None
        return self.detection_attempts[-1]


# ============================================
# LLM Response Models
# ============================================
class LLMThinkingMetadata(BaseModel):
    """Metadata about LLM thinking process."""
    thoughts: Optional[List[str]] = None  # Thought summaries (when includeThoughts=true)
    thoughts_token_count: Optional[int] = None  # Tokens used for thinking
    thinking_budget_used: Optional[int] = None  # Actual thinking budget consumed
    total_token_count: Optional[int] = None  # Total tokens (prompt + output + thinking)
    prompt_token_count: Optional[int] = None  # Input tokens
    candidates_token_count: Optional[int] = None  # Output tokens


class LLMResponse(BaseModel):
    """Enhanced LLM response with thinking metadata."""
    text: str  # The actual generated text
    thinking: Optional[LLMThinkingMetadata] = None  # Thinking traces
    raw_response: Optional[Dict[str, Any]] = None  # Full API response for debugging


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
