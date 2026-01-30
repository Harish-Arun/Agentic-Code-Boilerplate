"""
PDF Utilities Tool - PDF manipulation operations.

Provides tools for converting PDFs to images, cropping regions, etc.
"""
from typing import Dict, Any, List
from fastmcp import FastMCP

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import AppConfig


def register_pdf_tools(mcp: FastMCP, config: AppConfig):
    """Register PDF utility tools with the MCP server."""
    
    pdf_config = config.mcp.tools.pdf_utils
    
    @mcp.tool()
    async def pdf_to_images(
        pdf_path: str,
        pages: str = "all",
        dpi: int = 300
    ) -> Dict[str, Any]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to the PDF file
            pages: Which pages to convert - 'all', '1', '1-3', etc.
            dpi: Resolution for output images
        
        Returns:
            List of generated image paths
        """
        max_pages = pdf_config.get("max_pages", 50)
        actual_dpi = min(dpi, pdf_config.get("dpi", 300))
        
        # Mock implementation
        return {
            "success": True,
            "images": [
                f"/data/images/{Path(pdf_path).stem}_page_1.png",
                f"/data/images/{Path(pdf_path).stem}_page_2.png"
            ],
            "page_count": 2,
            "dpi": actual_dpi,
            "message": "[Mock] PDF converted to images"
        }
    
    @mcp.tool()
    async def crop_region(
        image_path: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Crop a region from an image.
        
        Args:
            image_path: Path to the source image
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            output_path: Optional output path (auto-generated if not provided)
        
        Returns:
            Path to the cropped image
        """
        if output_path is None:
            output_path = f"/data/crops/{Path(image_path).stem}_crop.png"
        
        # Mock implementation
        return {
            "success": True,
            "cropped_image_path": output_path,
            "original_region": {
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            },
            "message": "[Mock] Region cropped successfully"
        }
    
    @mcp.tool()
    async def get_pdf_metadata(
        pdf_path: str
    ) -> Dict[str, Any]:
        """
        Get metadata from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
        
        Returns:
            PDF metadata including page count, size, etc.
        """
        # Mock implementation
        return {
            "success": True,
            "metadata": {
                "page_count": 2,
                "title": "Payment Instruction",
                "author": "System",
                "creation_date": "2026-01-29T10:00:00Z",
                "file_size_bytes": 145678,
                "page_sizes": [
                    {"width": 612, "height": 792},  # Letter size
                    {"width": 612, "height": 792}
                ]
            }
        }
    
    @mcp.tool()
    async def merge_pdfs(
        pdf_paths: List[str],
        output_path: str
    ) -> Dict[str, Any]:
        """
        Merge multiple PDF files into one.
        
        Args:
            pdf_paths: List of PDF file paths to merge
            output_path: Path for the merged PDF
        
        Returns:
            Path to merged PDF and page count
        """
        # Mock implementation
        return {
            "success": True,
            "merged_pdf_path": output_path,
            "total_pages": len(pdf_paths) * 2,  # Mock: 2 pages per PDF
            "source_files": pdf_paths,
            "message": "[Mock] PDFs merged successfully"
        }
