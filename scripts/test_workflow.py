"""
Test script for end-to-end M1-M7 signature verification workflow.

This script tests the full LangGraph workflow:
1. Document upload → data/uploads/
2. Extraction node → Extract payment fields from PDF
3. Signature detection node → Find and crop signatures
4. Verification node → M1-M7 metrics analysis with Gemini Vision
5. FIV 1.0 scoring engine → Deterministic confidence score

Usage:
    python scripts/test_workflow.py [pdf_file_path]
    
Example:
    python scripts/test_workflow.py data/uploads/sample.pdf
"""
import asyncio
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Install with: pip install python-dotenv")

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

from config import get_config
from models import AgentState
from graph.workflow import create_workflow, run_workflow


async def test_workflow(pdf_path: str):
    """
    Test the full signature verification workflow with a real PDF.
    
    Args:
        pdf_path: Path to PDF file to process
    """
    print("=" * 80)
    print("🧪 NNP-AI M1-M7 Signature Verification Workflow Test")
    print("=" * 80)
    print()
    
    # Validate PDF exists
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        print(f"   Please provide a valid PDF path")
        return False
    
    print(f"📄 Document: {pdf_file.name}")
    print(f"📂 Path: {pdf_file.absolute()}")
    print(f"📊 Size: {pdf_file.stat().st_size / 1024:.1f} KB")
    print()
    
    # Load configuration
    config = get_config()
    print(f"⚙️ Configuration:")
    print(f"   LLM Provider: {config.llm.provider}")
    if config.llm.provider == "gemini":
        print(f"   Model: {config.llm.gemini.model}")
    elif config.llm.provider == "openai":
        print(f"   Model: {config.llm.openai.model}")
    print(f"   Database: {config.database.type}")
    print(f"   Checkpointing: {'Enabled' if config.agents.checkpointing.enabled else 'Disabled'}")
    print(f"   Human-in-Loop: {'Enabled' if config.features.human_in_loop else 'Disabled'}")
    print()
    
    # Check API key
    if config.llm.provider == "gemini":
        if not os.environ.get("GEMINI_API_KEY"):
            print("❌ Error: GEMINI_API_KEY not set in environment")
            return False
        print(f"✅ Gemini API key configured")
        print()
    
    # Create workflow
    print("🔧 Initializing LangGraph workflow...")
    try:
        workflow, checkpointer = create_workflow(config)
        print("✅ Workflow created successfully")
        print()
    except Exception as e:
        print(f"❌ Error creating workflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Create initial state
    document_id = f"TEST-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    thread_id = f"thread-{document_id}"
    
    initial_state = AgentState(
        document_id=document_id,
        document_path=str(pdf_file.absolute()),
        thread_id=thread_id
    )
    
    print("=" * 80)
    print("🚀 Starting Workflow Execution")
    print("=" * 80)
    print()
    
    # Run workflow
    start_time = asyncio.get_event_loop().time()
    
    try:
        result = await run_workflow(
            workflow=workflow,
            initial_state=initial_state,
            thread_id=thread_id
        )
        
        elapsed = asyncio.get_event_loop().time() - start_time
        
        print()
        print("=" * 80)
        print("✅ Workflow Completed Successfully")
        print("=" * 80)
        print()
        print(f"⏱️  Total processing time: {elapsed:.2f}s")
        
        # Save full state to file for inspection
        import json
        output_file = Path("./data/debug/workflow_state.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert AgentState Pydantic model to dict for JSON serialization
        state_dict = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
        
        with open(output_file, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        print(f"💾 Full workflow state saved to: {output_file}")
        print()
        
        # Display results
        print_results(result)
        
        return True
        
    except Exception as e:
        print()
        print("=" * 80)
        print("❌ Workflow Failed")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_results(state: AgentState):
    """Print formatted workflow results."""
    
    print("📊 WORKFLOW RESULTS")
    print("-" * 80)
    print()
    
    # Extraction Results
    print("1️⃣ EXTRACTION RESULTS")
    print("-" * 40)
    if state.extracted_payment:
        payment = state.extracted_payment
        fields = [
            ("creditor_name", payment.creditor_name),
            ("creditor_account", payment.creditor_account),
            ("debtor_name", payment.debtor_name),
            ("debtor_account", payment.debtor_account),
            ("amount", payment.amount),
            ("currency", payment.currency),
            ("payment_type", payment.payment_type),
            ("payment_date", payment.payment_date),
        ]
        
        for field_name, field_data in fields:
            if field_data:
                value = field_data.value
                confidence = field_data.confidence
                print(f"   {field_name.replace('_', ' ').title()}: {value} ({confidence*100:.0f}%)")
    else:
        print("   ⚠️ No data extracted")
    print()
    
    # Signature Detection Results
    print("2️⃣ SIGNATURE DETECTION RESULTS")
    print("-" * 40)
    if state.signature_detections:
        print(f"   Found {len(state.signature_detections)} signature(s)")
        for i, sig in enumerate(state.signature_detections, 1):
            print(f"   Signature {i}:")
            print(f"      Type: {sig.signature_type}")
            print(f"      Location: ({sig.bounding_box.x1:.2f}, {sig.bounding_box.y1:.2f}) → ({sig.bounding_box.x2:.2f}, {sig.bounding_box.y2:.2f})")
            print(f"      Confidence: {sig.confidence*100:.0f}%")
            print(f"      Description: {sig.description}")
            if sig.cropped_image_path:
                print(f"      Saved to: {sig.cropped_image_path}")
    else:
        print("   ⚠️ No signatures detected")
        # Check detection attempts for errors
        if state.detection_attempts:
            latest = state.detection_attempts[-1]
            if not latest.success:
                print(f"   Last detection attempt failed:")
                for error in latest.errors:
                    print(f"      - {error}")
    print()
    
    # Signature Verification Results
    print("3️⃣ SIGNATURE VERIFICATION RESULTS (M1-M7)")
    print("-" * 40)
    if state.verification_result:
        result = state.verification_result
        
        print(f"   Match Status: {'✅ MATCH' if result.match else '❌ MISMATCH'}")
        print(f"   Confidence: {result.confidence*100:.1f}%")
        print(f"   Reference ID: {result.reference_signature_id}")
        print(f"   Reasoning: {result.reasoning}")
        
        if result.risk_indicators:
            print(f"\n   ⚠️  Risk Indicators:")
            for risk in result.risk_indicators:
                print(f"      - {risk}")
        
        print(f"\n   Recommendation: {result.recommendation}")
        
        if result.similarity_factors:
            print("\n   📊 Detailed Factors:")
            import json
            print(json.dumps(result.similarity_factors.model_dump(), indent=6))
            
        # Additional details would be in the full state file
        print()
        print(f"   📄 Full details saved to: data/debug/workflow_state.json")
    else:
        print("   ⚠️ No verification result")
    print()
    
    # Errors
    if state.extraction_errors or state.detection_errors or state.verification_errors:
        print("⚠️ ERRORS")
        print("-" * 40)
        for error in state.extraction_errors:
            print(f"   Extraction: {error}")
        for error in state.detection_errors:
            print(f"   Detection: {error}")
        for error in state.verification_errors:
            print(f"   Verification: {error}")
        print()
    
    # History
    if state.history:
        print("📜 WORKFLOW HISTORY")
        print("-" * 40)
        for entry in state.history[-10:]:  # Last 10 entries
            timestamp = entry.timestamp.strftime("%H:%M:%S") if entry.timestamp else "N/A"
            step = entry.step or "N/A"
            action = entry.action or "N/A"
            print(f"   [{timestamp}] {step} - {action}")
        print()


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
            print("   python scripts/test_workflow.py <path_to_pdf>")
            print()
            print("Example:")
            print("   python scripts/test_workflow.py data/uploads/sample.pdf")
            return
    
    # Run test
    success = await test_workflow(pdf_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
