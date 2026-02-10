"""Debug script to visualize signature crop coordinates."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

def visualize_crops(pdf_path, detections):
    """Draw bounding boxes on PDF to visualize detection coordinates."""
    
    doc = fitz.open(pdf_path)
    
    for idx, detection in enumerate(detections):
        page = doc.load_page(detection["page"] - 1)
        page_rect = page.rect
        
        print(f"\nSignature {idx + 1}:")
        print(f"  Page dimensions: {page_rect.width} x {page_rect.height} points")
        print(f"  Normalized coords: ({detection['x1']:.3f}, {detection['y1']:.3f}) → ({detection['x2']:.3f}, {detection['y2']:.3f})")
        
        # Convert normalized to actual
        x1 = detection["x1"] * page_rect.width
        y1 = detection["y1"] * page_rect.height
        x2 = detection["x2"] * page_rect.width
        y2 = detection["y2"] * page_rect.height
        
        print(f"  Actual coords: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f}) points")
        print(f"  Width x Height: {x2-x1:.1f} x {y2-y1:.1f} points")
        
        # Render full page with bounding box overlay
        pix = page.get_pixmap(dpi=150)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        # Draw bounding box
        draw = ImageDraw.Draw(img)
        
        # Scale coordinates to image resolution (150 DPI)
        scale = 150 / 72  # DPI / 72 (points per inch)
        box_x1 = x1 * scale
        box_y1 = y1 * scale
        box_x2 = x2 * scale
        box_y2 = y2 * scale
        
        draw.rectangle([box_x1, box_y1, box_x2, box_y2], outline="red", width=3)
        
        # Save annotated page
        output_dir = Path("./data/debug")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"annotated_sig{idx+1}.png"
        img.save(output_path)
        print(f"  Saved annotated image: {output_path}")
        
        # Also save the crop
        crop_rect = fitz.Rect(x1, y1, x2, y2)
        crop_pix = page.get_pixmap(clip=crop_rect, dpi=300)
        crop_output = output_dir / f"cropped_sig{idx+1}.png"
        crop_pix.save(crop_output)
        print(f"  Saved crop: {crop_output}")
    
    doc.close()


if __name__ == "__main__":
    # Example detections from your test run
    detections = [
        {
            "page": 1,
            "x1": 0.68, "y1": 0.83,
            "x2": 0.92, "y2": 0.89,
            "type": "customer",
            "description": "Handwritten signature in a designated signature field."
        },
        {
            "page": 1,
            "x1": 0.68, "y1": 0.89,
            "x2": 0.92, "y2": 0.95,
            "type": "bank_authenticator",
            "description": "Handwritten signature in a designated signature field, likely bank authenticator."
        }
    ]
    
    pdf_path = "./data/uploads/sample.pdf"
    
    print("="*80)
    print("PDF Signature Coordinate Debug Tool")
    print("="*80)
    
    visualize_crops(pdf_path, detections)
    
    print("\n" + "="*80)
    print("✅ Check ./data/debug/ for annotated images")
    print("="*80)
