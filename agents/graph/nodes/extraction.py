"""
Extraction Agent Node - Thin orchestrator.

Delegates ALL business logic to MCP tools:
  - extract_payment_fields: Gemini Vision + structured extraction
  - validate_extraction: Business rule validation

This node only manages state transitions, history tracking, and retry logic.
"""
import sys
import time
import json
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from models import (
    AgentState, ExtractedPayment, PaymentField,
    ExtractionAttempt
)
from config import AppConfig
from mcp_client import get_mcp_client, call_tool_on_session


async def extraction_node(state: AgentState, config: AppConfig) -> AgentState:
    """
    Extract payment fields by calling MCP extraction tools.

    Flow:
    1. Call MCP extract_payment_fields -> Gemini Vision extraction
    2. Call MCP validate_extraction -> Challenger validation
    3. Record attempt + update state
    """
    state.current_step = "extraction"
    state.add_history("extraction", "started", {
        "document_path": state.document_path
    }, agent="extraction_agent")

    start_time = time.time()
    attempt_number = len(state.extraction_attempts) + 1

    try:
        print("\n" + "="*80)
        print("🔌 EXTRACTION NODE - Connecting to MCP")
        print("="*80)
        print(f"Document path: {state.document_path}")
        print("="*80 + "\n")
        
        async with get_mcp_client() as mcp:
            # Step 1: Extract payment fields via MCP
            state.add_history("extraction", "calling_mcp_extract", {
                "document_path": state.document_path,
                "tool": "extract_payment_fields"
            }, agent="extraction_agent")

            print("✅ MCP connection established, calling extract_payment_fields...\n")
            
            extraction_result = await call_tool_on_session(mcp, "extract_payment_fields", {
                "document_path": state.document_path
            })

            processing_time = int((time.time() - start_time) * 1000)

            if not extraction_result.get("success"):
                error_msg = extraction_result.get("error", "Extraction failed")
                raise RuntimeError(error_msg)

            # Convert response to ExtractedPayment model
            raw_payment = extraction_result["extracted_payment"]
            extracted_payment = _convert_mcp_response(raw_payment)
            model_used = extraction_result.get("model_used", "")
            
            # DEBUG: Log extracted fields
            print("\n" + "="*80)
            print("💰 EXTRACTION NODE - Fields Extracted")
            print("="*80)
            import json
            print(json.dumps(raw_payment, indent=2, default=str))
            print(f"\nConverted to model with {_count_fields(extracted_payment)} non-null fields")
            print("="*80 + "\n")

            # Step 2: Validate extraction via MCP
            validation_result = await call_tool_on_session(mcp, "validate_extraction", {
                "extracted_fields": json.dumps(raw_payment)
            })

            if not validation_result.get("valid", True):
                state.add_history("extraction", "validation_warning", {
                    "issues": validation_result.get("issues", [])
                }, agent="challenger_agent", notes=validation_result.get("notes"))

            # Record successful attempt
            attempt = ExtractionAttempt(
                attempt_number=attempt_number,
                success=True,
                extracted_payment=extracted_payment,
                model_used=model_used,
                processing_time_ms=processing_time,
                raw_response=str(raw_payment)[:2000]
            )
            state.add_extraction_attempt(attempt)

            state.add_history("extraction", "completed", {
                "attempt": attempt_number,
                "success": True,
                "fields_extracted": _count_fields(extracted_payment),
                "processing_time_ms": processing_time
            }, agent="extraction_agent")

    except Exception as e:
        import traceback
        processing_time = int((time.time() - start_time) * 1000)
        error_msg = f"Extraction failed: {str(e)}"

        attempt = ExtractionAttempt(
            attempt_number=attempt_number,
            success=False,
            errors=[error_msg],
            model_used=config.llm.gemini.model,
            processing_time_ms=processing_time
        )
        state.add_extraction_attempt(attempt)
        state.add_history("extraction", "failed", {
            "error": error_msg,
            "attempt": attempt_number,
            "traceback": traceback.format_exc()[:1000]
        }, agent="extraction_agent")

        # Retry logic
        if state.retry_count < state.max_retries:
            state.retry_count += 1
            state.add_history("extraction", "retry_scheduled", {
                "retry_count": state.retry_count,
                "max_retries": state.max_retries
            }, agent="extraction_agent")

    return state


def _convert_mcp_response(raw: dict) -> ExtractedPayment:
    """Convert MCP tool response to ExtractedPayment model."""
    def make_field(data) -> Optional[PaymentField]:
        if not data or (isinstance(data, dict) and data.get("value") is None):
            return None
        if isinstance(data, dict):
            return PaymentField(
                value=data.get("value"),
                confidence=float(data.get("confidence", 0.0)),
                source="ai",
                location=data.get("location", "")
            )
        return None

    return ExtractedPayment(
        creditor_name=make_field(raw.get("creditor_name")),
        creditor_account=make_field(raw.get("creditor_account")),
        creditor_bank=make_field(raw.get("creditor_bank")),
        debtor_name=make_field(raw.get("debtor_name")),
        debtor_account=make_field(raw.get("debtor_account")),
        debtor_bank=make_field(raw.get("debtor_bank")),
        amount=make_field(raw.get("amount")),
        currency=make_field(raw.get("currency")),
        payment_type=make_field(raw.get("payment_type")),
        payment_date=make_field(raw.get("payment_date")),
        charges_account=make_field(raw.get("charges_account")),
        reference=make_field(raw.get("reference")),
        raw_ocr_text=raw.get("raw_ocr_text")
    )


def _count_fields(payment: ExtractedPayment) -> int:
    """Count non-null fields in extracted payment."""
    count = 0
    for field_name in ['creditor_name', 'creditor_account', 'creditor_bank',
                       'debtor_name', 'debtor_account', 'debtor_bank',
                       'amount', 'currency', 'payment_type', 'payment_date',
                       'charges_account', 'reference']:
        if getattr(payment, field_name, None) is not None:
            count += 1
    return count
