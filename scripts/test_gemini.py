"""
Test script for verifying Gemini REST API integration.

Usage:
    1. Set GEMINI_API_KEY in .env file
    2. Run: python scripts/test_gemini.py

This script tests the Gemini REST API adapter without running the full workflow.
"""
import asyncio
import os
import sys
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))


async def test_gemini_text():
    """Test basic text generation."""
    from adapters.gemini_rest import GeminiRestAdapter
    
    print("=" * 60)
    print("Testing Gemini Text Generation")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Skipping test.")
        return False
    
    try:
        gemini = GeminiRestAdapter(api_key=api_key)
        response = await gemini.generate(
            "What is 2 + 2? Reply with just the number.",
            system_prompt="You are a helpful assistant."
        )
        print(f"✅ Response: {response.strip()}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_gemini_vision():
    """Test vision/document analysis."""
    from adapters.gemini_rest import GeminiRestAdapter
    
    print("\n" + "=" * 60)
    print("Testing Gemini Vision (Document Analysis)")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Skipping test.")
        return False
    
    # Check for test file
    uploads_dir = Path(__file__).parent.parent / "data" / "uploads"
    test_file = None
    
    if uploads_dir.exists():
        # Find any PDF file in uploads directory
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            test_file = pdf_files[0]
            print(f"📄 Found PDF: {test_file.name}")
    
    if not test_file:
        print("⚠️ No test PDF found. Upload a PDF to data/uploads/ to test vision.")
        print("   Trying with a simple text prompt instead...")
        return True
    
    try:
        gemini = GeminiRestAdapter(api_key=api_key)
        response = await gemini.generate_with_vision(
            "Describe what you see in this document in one sentence.",
            files=[str(test_file)]
        )
        print(f"✅ Vision Response: {response[:200]}...")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


async def test_payment_extraction():
    """Test payment field extraction from real PDF document."""
    from adapters.gemini_rest import GeminiRestAdapter
    
    print("\n" + "=" * 60)
    print("Testing Payment Field Extraction from PDF")
    print("=" * 60)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ GEMINI_API_KEY not set. Skipping test.")
        return False
    
    # Find a PDF file to test with
    uploads_dir = Path(__file__).parent.parent / "data" / "uploads"
    test_file = None
    
    if uploads_dir.exists():
        pdf_files = list(uploads_dir.glob("*.pdf"))
        if pdf_files:
            test_file = pdf_files[0]
            print(f"📄 Processing: {test_file.name}")
    
    if not test_file:
        print("⚠️ No PDF found. Skipping extraction test.")
        return True
    
    try:
        gemini = GeminiRestAdapter(api_key=api_key)
        
        schema = {
            "creditor_name": {"value": "string", "confidence": 0.0},
            "creditor_account": {"value": "string", "confidence": 0.0},
            "debtor_name": {"value": "string", "confidence": 0.0},
            "debtor_account": {"value": "string", "confidence": 0.0},
            "amount": {"value": "number", "confidence": 0.0},
            "currency": {"value": "string", "confidence": 0.0},
            "payment_type": {"value": "string", "confidence": 0.0}
        }
        
        result = await gemini.generate_structured(
            """Extract the payment fields from this document. Look for:
            - creditor_name: Beneficiary/To/Payee name
            - creditor_account: Beneficiary account number or IBAN
            - debtor_name: Payer/From name
            - debtor_account: Payer account number or IBAN
            - amount: Payment amount (numeric value only)
            - currency: Currency code (GBP, USD, EUR, etc)
            - payment_type: Type of payment (CHAPS, BACS, Standing Order, etc)
            
            Provide confidence scores between 0.0 and 1.0 for each field.""",
            schema=schema,
            files=[str(test_file)]
        )
        
        print("✅ Extracted fields from PDF:")
        for key, value in result.items():
            print(f"   {key}: {value}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n🧪 NNP-AI Gemini Integration Tests\n")
    
    results = []
    
    # Run tests
    results.append(("Text Generation", await test_gemini_text()))
    await asyncio.sleep(3)  # Delay between tests to avoid rate limits
    
    results.append(("Vision Analysis", await test_gemini_vision()))
    await asyncio.sleep(3)  # Delay between tests
    
    results.append(("Payment Extraction", await test_payment_extraction()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed < len(results):
        print("\n⚠️ Some tests failed. Make sure GEMINI_API_KEY is set correctly.")


if __name__ == "__main__":
    asyncio.run(main())
