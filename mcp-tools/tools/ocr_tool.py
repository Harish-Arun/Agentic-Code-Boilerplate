"""
OCR Tool - Extract text from documents/images.

Supports mock mode for testing and Gemini Vision for production.
"""
from typing import Dict, Any
from fastmcp import FastMCP

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import AppConfig
from adapters import get_llm_adapter


def register_ocr_tools(mcp: FastMCP, config: AppConfig):
    """Register OCR-related tools with the MCP server."""
    
    @mcp.tool()
    async def ocr_extract(
        image_path: str,
        output_format: str = "text"
    ) -> Dict[str, Any]:
        """
        Extract text from an image or document page.
        
        Args:
            image_path: Path to the image file
            output_format: Output format - 'text', 'json', or 'structured'
        
        Returns:
            Extracted text content with metadata
        """
        ocr_config = config.mcp.tools.ocr
        
        if ocr_config.get("provider") == "mock":
            # Mock OCR response
            return {
                "success": True,
                "text": """PAYMENT INSTRUCTION
                
Date: 29/01/2026
Reference: PAY-2026-001234

Debtor Details:
Name: John Smith
Account: GB82 WEST 1234 5698 7654 32
Bank: Western Bank PLC

Creditor Details:
Name: ACME Corporation Ltd
Account: GB29 NWBK 6016 1331 9268 19
Bank: National Bank UK

Amount: GBP 15,000.00
Payment Type: CHAPS

Authorized Signature: [SIGNATURE PRESENT]
                """,
                "confidence": 0.95,
                "pages_processed": 1,
                "format": output_format
            }
        else:
            # Production: Use vision LLM
            llm = get_llm_adapter(config)
            
            result = await llm.generate_with_vision(
                prompt="Extract all text from this document. Preserve the structure and formatting.",
                images=[image_path]
            )
            
            return {
                "success": True,
                "text": result,
                "confidence": 0.9,
                "pages_processed": 1,
                "format": output_format
            }
    
    @mcp.tool()
    async def ocr_extract_structured(
        image_path: str,
        fields: list[str]
    ) -> Dict[str, Any]:
        """
        Extract specific fields from a document.
        
        Args:
            image_path: Path to the image file
            fields: List of field names to extract
        
        Returns:
            Dictionary with extracted field values
        """
        ocr_config = config.mcp.tools.ocr
        
        if ocr_config.get("provider") == "mock":
            # Mock structured extraction
            return {
                "success": True,
                "fields": {
                    "creditor_name": "ACME Corporation Ltd",
                    "creditor_account": "GB29NWBK60161331926819",
                    "debtor_name": "John Smith",
                    "debtor_account": "GB82WEST12345698765432",
                    "amount": "15000.00",
                    "currency": "GBP",
                    "payment_type": "CHAPS"
                },
                "confidence": 0.92
            }
        else:
            llm = get_llm_adapter(config)
            
            result = await llm.generate_structured(
                prompt=f"Extract these fields from the document: {', '.join(fields)}",
                schema={field: "string" for field in fields}
            )
            
            return {
                "success": True,
                "fields": result,
                "confidence": 0.9
            }
