"""
PDF Utilities Tool - PDF manipulation operations.

Production implementation with:
- PDF to image conversion using pdf2image/PIL
- Region cropping with proper coordinate handling
- PDF metadata extraction using PyMuPDF
- Support for percentage-based and pixel-based coordinates
"""
from typing import Dict, Any, List
from fastmcp import FastMCP
import os
from pathlib import Path

import sys
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
            dpi: Resolution for output images (default: 300)
        
        Returns:
            List of generated image paths
        """
        max_pages = pdf_config.get("max_pages", 50)
        actual_dpi = min(dpi, pdf_config.get("dpi", 300))
        
        data_dir = os.environ.get("DATA_DIR", "/data")
        images_dir = Path(data_dir) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}",
                "images": [],
                "page_count": 0
            }
        
        try:
            from pdf2image import convert_from_path
            
            # Parse page specification
            first_page = None
            last_page = None
            if pages != "all":
                if "-" in pages:
                    parts = pages.split("-")
                    first_page = int(parts[0])
                    last_page = int(parts[1])
                else:
                    first_page = int(pages)
                    last_page = int(pages)
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=actual_dpi,
                first_page=first_page,
                last_page=last_page
            )
            
            # Limit pages
            images = images[:max_pages]
            
            # Save images
            saved_paths = []
            pdf_stem = Path(pdf_path).stem
            for idx, image in enumerate(images):
                page_num = (first_page or 1) + idx
                output_path = images_dir / f"{pdf_stem}_page_{page_num}.png"
                image.save(str(output_path), "PNG")
                saved_paths.append(str(output_path))
            
            return {
                "success": True,
                "images": saved_paths,
                "page_count": len(saved_paths),
                "dpi": actual_dpi,
                "message": f"Converted {len(saved_paths)} pages from PDF"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "images": [],
                "page_count": 0
            }
    
    @mcp.tool()
    async def crop_region(
        image_path: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        output_path: str = None,
        coordinate_type: str = "pixel"
    ) -> Dict[str, Any]:
        """
        Crop a region from an image.
        
        Args:
            image_path: Path to the source image
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
            output_path: Optional output path (auto-generated if not provided)
            coordinate_type: 'pixel' or 'percentage' (0.0-1.0)
        
        Returns:
            Path to the cropped image
        """
        data_dir = os.environ.get("DATA_DIR", "/data")
        crops_dir = Path(data_dir) / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        if output_path is None:
            output_path = f"{crops_dir}/{Path(image_path).stem}_crop_{int(x1)}_{int(y1)}.png"
        
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Source image not found: {image_path}",
                "cropped_image_path": None
            }
        
        try:
            from PIL import Image
            
            with Image.open(image_path) as img:
                width, height = img.size
                
                # Convert percentage coordinates to pixels if needed
                if coordinate_type == "percentage":
                    x1_px = int(x1 * width)
                    y1_px = int(y1 * height)
                    x2_px = int(x2 * width)
                    y2_px = int(y2 * height)
                else:
                    x1_px = int(x1)
                    y1_px = int(y1)
                    x2_px = int(x2)
                    y2_px = int(y2)
                
                # Ensure valid crop region
                x1_px = max(0, min(x1_px, width))
                y1_px = max(0, min(y1_px, height))
                x2_px = max(0, min(x2_px, width))
                y2_px = max(0, min(y2_px, height))
                
                # Crop and save
                cropped = img.crop((x1_px, y1_px, x2_px, y2_px))
                
                Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                cropped.save(output_path, "PNG")
            
            return {
                "success": True,
                "cropped_image_path": output_path,
                "original_region": {
                    "x1": x1_px, "y1": y1_px, "x2": x2_px, "y2": y2_px
                },
                "crop_size": {
                    "width": x2_px - x1_px,
                    "height": y2_px - y1_px
                },
                "message": "Region cropped successfully"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "cropped_image_path": None
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
            PDF metadata including page count, dimensions, etc.
        """
        if not os.path.exists(pdf_path):
            return {
                "success": False,
                "error": f"PDF file not found: {pdf_path}"
            }
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(pdf_path)
            
            pages_info = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                pages_info.append({
                    "page_number": page_num + 1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation
                })
            
            metadata = {
                "success": True,
                "filename": Path(pdf_path).name,
                "page_count": len(doc),
                "pages": pages_info,
                "metadata": dict(doc.metadata) if doc.metadata else {},
                "file_size_bytes": os.path.getsize(pdf_path)
            }
            
            doc.close()
            return metadata
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    @mcp.tool()
    async def extract_page_as_image(
        pdf_path: str,
        page_number: int = 1,
        dpi: int = 300
    ) -> Dict[str, Any]:
        """
        Extract a single page from a PDF as an image.
        
        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract (1-indexed)
            dpi: Resolution for output image
        
        Returns:
            Path to the extracted image
        """
        result = await pdf_to_images(
            pdf_path=pdf_path,
            pages=str(page_number),
            dpi=dpi
        )
        
        if result.get("success") and result.get("images"):
            return {
                "success": True,
                "image_path": result["images"][0],
                "page_number": page_number,
                "dpi": dpi
            }
        return result
