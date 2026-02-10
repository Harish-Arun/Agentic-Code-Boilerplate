"""
Debug script to test signature detection on a PDF and see what Gemini returns.

Usage:
    python scripts/test_signature_detection.py [pdf_file_path]
"""
import asyncio
import sys
import os
import json
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed")

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from adapters.gemini_rest import GeminiRestAdapter


async def test_signature_detection(pdf_path: str):
    """Test signature detection directly with Gemini API."""
    
    print("=" * 80)
    print("🔍 Signature Detection Debug Test")
    print("=" * 80)
    print()
    
    # Validate PDF exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        return
    
    print(f"📄 Document: {pdf_file.name}")
    print(f"📂 Path: {pdf_file.absolute()}")
    print()
    
    # Check API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY not set in environment")
        return
    
    print("✅ Gemini API key configured")
    print()
    
    # Create adapter
    model = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
    print(f"🤖 Using model: {model}")
    print()
    
    gemini = GeminiRestAdapter(api_key=api_key, model=model)
    
    print("🔍 Calling Gemini to detect signatures...")
    print()
    
    try:
        result = await gemini.detect_signatures(str(pdf_file.absolute()))
        
        print("=" * 80)
        print("📊 RAW GEMINI RESPONSE:")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        print()
        
        # Parse results
        signatures = result.get("signatures", [])
        total = result.get("total_signatures_found", 0)
        has_empty = result.get("has_empty_signature_fields", False)
        notes = result.get("notes", "")
        
        print("=" * 80)
        print("📈 SUMMARY:")
        print("=" * 80)
        print(f"Total signatures found: {total}")
        print(f"Has empty fields: {has_empty}")
        print(f"Notes: {notes}")
        print()
        
        if signatures:
            print("🖊️ DETECTED SIGNATURES:")
            print("-" * 80)
            for i, sig in enumerate(signatures, 1):
                bbox = sig.get("bounding_box", {})
                sig_type = sig.get("signature_type", "unknown")
                confidence = sig.get("confidence", 0.0)
                description = sig.get("description", "")
                
                print(f"\nSignature {i}:")
                print(f"  Type: {sig_type}")
                print(f"  Confidence: {confidence*100:.1f}%")
                print(f"  Location: ({bbox.get('x1', 0):.2f}, {bbox.get('y1', 0):.2f}) → ({bbox.get('x2', 0):.2f}, {bbox.get('y2', 0):.2f})")
                print(f"  Page: {bbox.get('page', 1)}")
                print(f"  Description: {description}")
        else:
            print("⚠️ NO SIGNATURES DETECTED")
            print()
            print("Possible reasons:")
            print("  1. PDF contains no handwritten signatures")
            print("  2. Signatures are too faint or low quality")
            print("  3. Document is a digital form without physical signatures")
            print("  4. Gemini model needs different prompt or parameters")
        
        print()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main entry point."""
    
    # Get PDF path from command line or use default
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        # Try to find a PDF in uploads directory
        uploads_dir = Path(__file__).parent.parent / "data" / "uploads"
        pdf_files = list(uploads_dir.glob("*.pdf")) if uploads_dir.exists() else []
        
        if pdf_files:
            pdf_path = str(pdf_files[0])
            print(f"ℹ️  No PDF specified, using: {pdf_files[0].name}")
            print()
        else:
            print("❌ Error: No PDF file specified")
            print()
            print("Usage:")
            print("   python scripts/test_signature_detection.py <path_to_pdf>")
            return
    
    await test_signature_detection(pdf_path)


if __name__ == "__main__":
    asyncio.run(main())
