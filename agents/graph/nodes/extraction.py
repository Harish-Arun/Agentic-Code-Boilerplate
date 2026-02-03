"""
Extraction Agent Node - Extract payment fields from documents.

This node uses vision LLM (or mock) to extract structured payment data.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from models import AgentState, ExtractedPayment, PaymentField
from config import AppConfig
from adapters import get_llm_adapter


async def extraction_node(state: AgentState, config: AppConfig) -> AgentState:
    """
    Extract payment fields from the document.
    
    This agent:
    1. Reads the document (PDF/image)
    2. Uses vision LLM to extract structured fields
    3. Returns confidence scores for each field
    
    In mock mode, returns sample data.
    In production, calls Gemini Vision.
    """
    state.current_step = "extraction"
    
    try:
        llm = get_llm_adapter(config)
        
        # Get extraction prompt from config
        extraction_prompt = config.prompts.extraction
        
        # Use vision if document is an image/PDF
        # For mock, we just use regular generate
        if config.llm.provider == "mock":
            # Demonstrating MCP Tool Usage
            # If enabled, validte tool connection and use it
            try:
                from mcp_client import get_mcp_client
                
                print("🔌 Connecting to MCP Tools for OCR...")
                async with get_mcp_client() as mcp:
                    # Call the 'ocr_extract' tool exposed by mcp-tools service
                    ocr_result = await mcp.call_tool("ocr_extract", arguments={"image_path": state.document_path})
                    
                    # Log the result from the tool
                    print(f"✅ MCP Tool Result: {ocr_result}")
                    
                    # Parse tool result if needed, for mock we just use the existing logic below
                    # but typically you'd populate state.extracted_payment from ocr_result
            except Exception as e:
                import traceback
                print(f"⚠️ MCP Tool Call Failed: {e}")
                traceback.print_exc()
                print("   Falling back to internal mock logic.")

            # Return mock extracted data
            state.extracted_payment = ExtractedPayment(
                creditor_name=PaymentField(value="ACME Corporation Ltd", confidence=0.95, source="ai"),
                creditor_account=PaymentField(value="GB29NWBK60161331926819", confidence=0.92, source="ai"),
                debtor_name=PaymentField(value="John Smith", confidence=0.94, source="ai"),
                debtor_account=PaymentField(value="GB82WEST12345698765432", confidence=0.91, source="ai"),
                amount=PaymentField(value=15000.00, confidence=0.98, source="ai"),
                currency=PaymentField(value="GBP", confidence=0.99, source="ai"),
                payment_type=PaymentField(value="CHAPS", confidence=0.88, source="ai"),
                payment_date=PaymentField(value="2026-01-29", confidence=0.85, source="ai"),
                charges_account=PaymentField(value="GB82WEST12345698765432", confidence=0.80, source="ai"),
                raw_ocr_text="[Mock OCR text from document...]"
            )
        else:
            # Production: Use vision LLM
            result = await llm.generate_structured(
                prompt=f"{extraction_prompt}\n\nDocument path: {state.document_path}",
                schema={
                    "creditor_name": "string",
                    "creditor_account": "string",
                    "debtor_name": "string",
                    "debtor_account": "string",
                    "amount": "number",
                    "currency": "string",
                    "payment_type": "string",
                    "payment_date": "string"
                }
            )
            
            # Convert to ExtractedPayment model
            state.extracted_payment = ExtractedPayment(
                creditor_name=PaymentField(value=result.get("creditor_name", ""), confidence=0.9, source="ai"),
                creditor_account=PaymentField(value=result.get("creditor_account", ""), confidence=0.9, source="ai"),
                debtor_name=PaymentField(value=result.get("debtor_name", ""), confidence=0.9, source="ai"),
                debtor_account=PaymentField(value=result.get("debtor_account", ""), confidence=0.9, source="ai"),
                amount=PaymentField(value=result.get("amount", 0), confidence=0.9, source="ai"),
                currency=PaymentField(value=result.get("currency", ""), confidence=0.9, source="ai"),
                payment_type=PaymentField(value=result.get("payment_type", ""), confidence=0.9, source="ai"),
                payment_date=PaymentField(value=result.get("payment_date", ""), confidence=0.9, source="ai")
            )
    
    except Exception as e:
        state.extraction_errors.append(f"Extraction failed: {str(e)}")
    
    return state
