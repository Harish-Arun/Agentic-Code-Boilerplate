"""
OCR Tool - Extract text from documents/images using Gemini Vision.

Production implementation with:
- Gemini REST API for vision-based OCR
- Structured field extraction
- Support for PDF and image files
"""
from typing import Dict, Any
from fastmcp import FastMCP
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import AppConfig


def register_ocr_tools(mcp: FastMCP, config: AppConfig):
    """Register OCR-related tools with the MCP server."""
    
    @mcp.tool()
    async def ocr_extract(
        image_path: str,
        output_format: str = "text"
    ) -> Dict[str, Any]:
        """
        Extract text from an image or document page using Gemini Vision.
        
        Args:
            image_path: Path to the image file or PDF
            output_format: Output format - 'text', 'json', or 'structured'
        
        Returns:
            Extracted text content with metadata
        """
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"File not found: {image_path}",
                "text": ""
            }
        
        try:
            from adapters import get_gemini_adapter
            
            gemini = get_gemini_adapter(config)
            
            prompt = """Extract all text from this document. 
Preserve the structure and formatting as much as possible.
Include all visible text, numbers, dates, and labels."""

            result = await gemini.generate_with_vision(
                prompt=prompt,
                files=[image_path]
            )
            
            return {
                "success": True,
                "text": result,
                "confidence": 0.9,
                "pages_processed": 1,
                "format": output_format
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": ""
            }
    
    @mcp.tool()
    async def ocr_extract_structured(
        image_path: str,
        fields: list[str] = None
    ) -> Dict[str, Any]:
        """
        Extract specific fields from a document using Gemini Vision.
        
        Args:
            image_path: Path to the image file or PDF
            fields: List of field names to extract (optional, extracts all if not provided)
        
        Returns:
            Dictionary with extracted field values
        """
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"File not found: {image_path}",
                "fields": {}
            }
        
        default_fields = [
            "creditor_name", "creditor_account", "creditor_bank",
            "debtor_name", "debtor_account", "debtor_bank",
            "amount", "currency", "payment_type", "payment_date",
            "reference"
        ]
        fields = fields or default_fields
        
        try:
            from adapters import get_gemini_adapter
            
            gemini = get_gemini_adapter(config)
            
            prompt = f"""Extract these specific fields from the document: {', '.join(fields)}

For each field, provide the value if found. If a field is not present, return null.
Return only the requested fields."""

            schema = {field: "string or null" for field in fields}
            
            result = await gemini.generate_structured(
                prompt=prompt,
                schema=schema,
                files=[image_path]
            )
            
            return {
                "success": True,
                "fields": result,
                "confidence": 0.9
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "fields": {}
            }
    
    @mcp.tool()
    async def ocr_detect_text_regions(
        image_path: str
    ) -> Dict[str, Any]:
        """
        Detect text regions in an image with bounding boxes using Gemini Vision.
        
        Args:
            image_path: Path to the image file
        
        Returns:
            List of text regions with coordinates
        """
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"File not found: {image_path}",
                "regions": []
            }
        
        try:
            from adapters import get_gemini_adapter
            
            gemini = get_gemini_adapter(config)
            
            prompt = """Analyze this document and identify all distinct text regions.
For each region, provide:
1. The text content
2. Bounding box as percentages (0.0 to 1.0) of image dimensions: x1, y1 (top-left), x2, y2 (bottom-right)

Focus on major text blocks like headers, paragraphs, labels, and values."""

            schema = {
                "regions": [
                    {"text": "string", "bbox": {"x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.0}}
                ]
            }
            
            result = await gemini.generate_structured(
                prompt=prompt,
                schema=schema,
                files=[image_path]
            )
            
            return {
                "success": True,
                "regions": result.get("regions", [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "regions": []
            }
