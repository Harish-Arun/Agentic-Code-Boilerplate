"""
Verification Agent Node - Thin orchestrator.

Delegates ALL business logic to MCP tools:
  - get_reference_signature: Reference signature lookup
  - verify_signature: Gemini M1-M7 analysis + FIV 1.0 scoring

This node only manages state transitions, history tracking, and retry logic.
"""
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from models import (
    AgentState, SignatureVerification, SimilarityFactors,
    VerificationAttempt, ReferenceSignature
)
from config import AppConfig
from mcp_client import get_mcp_client, call_tool_on_session


async def verification_node(state: AgentState, config: AppConfig) -> AgentState:
    """
    Verify signatures by calling MCP verification tools.

    Flow:
    1. Get latest detections from state
    2. For each signature:
       a. Call MCP get_reference_signature -> lookup reference image
       b. Call MCP verify_signature -> Gemini M1-M7 + FIV 1.0 scoring
    3. Aggregate results + update state
    """
    state.current_step = "verification"
    state.add_history("verification", "started", {
        "document_path": state.document_path
    }, agent="verification_agent")

    start_time = time.time()
    attempt_number = len(state.verification_attempts) + 1

    # Get latest detections
    latest_detection = state.get_latest_detection()
    if not latest_detection or not latest_detection.detections:
        state.add_history("verification", "skipped", {
            "reason": "no_signatures_detected"
        }, agent="verification_agent")
        return state

    detections = latest_detection.detections
    state.add_history("verification", "signatures_to_verify", {
        "count": len(detections)
    }, agent="verification_agent")

    try:
        async with get_mcp_client() as mcp:
            verifications = []
            thinking_traces = []  # Collect thinking from all verifications

            for idx, detection in enumerate(detections):
                sig_start = time.time()

                if not detection.cropped_image_path and not detection.image_blob:
                    state.add_history("verification", "skipped_signature", {
                        "index": idx,
                        "reason": "no_cropped_image"
                    }, agent="verification_agent")
                    continue

                # Step 1: Get reference signature via MCP
                # Use customer name from extracted data or fallback to signer name
                customer_id = None
                if state.extracted_payment and state.extracted_payment.debtor_name:
                    # Extract string value from field object
                    debtor_field = state.extracted_payment.debtor_name
                    customer_id = debtor_field.get('value') if isinstance(debtor_field, dict) else str(debtor_field)
                
                customer_id = customer_id or detection.signer_name or "default_customer"
                
                ref_result = await call_tool_on_session(mcp, "get_reference_signature", {
                    "customer_id": customer_id
                })

                reference_path = None
                reference_blob = None
                reference_mime = None
                if ref_result.get("success"):
                    sig_data = ref_result.get("signature", {})
                    reference_path = sig_data.get("signature_image_path")
                    reference_blob = sig_data.get("image_blob")
                    reference_mime = sig_data.get("blob_mime_type", "image/png")
                    state.add_history("verification", "reference_found", {
                        "index": idx,
                        "customer_id": customer_id,
                        "has_blob": reference_blob is not None,
                        "reference_path": reference_path
                    }, agent="verification_agent")
                else:
                    state.add_history("verification", "no_reference", {
                        "index": idx,
                        "customer_id": customer_id,
                        "reason": ref_result.get("error", "not found")
                    }, agent="verification_agent")

                # Step 2: Verify signature via MCP (Gemini M1-M7 + FIV 1.0)
                # Prefer blob inputs (simulates ISV blob flow); fall back to paths
                verify_args = {}
                if detection.image_blob:
                    verify_args["signature_blob"] = detection.image_blob
                    verify_args["signature_mime_type"] = detection.blob_mime_type or "image/png"
                elif detection.cropped_image_path:
                    verify_args["signature_path"] = detection.cropped_image_path

                if reference_blob:
                    verify_args["reference_blob"] = reference_blob
                    verify_args["reference_mime_type"] = reference_mime or "image/png"
                elif reference_path:
                    verify_args["reference_path"] = reference_path

                state.add_history("verification", "calling_mcp_verify", {
                    "index": idx,
                    "tool": "verify_signature",
                    "has_reference": (reference_blob is not None or reference_path is not None),
                    "using_blobs": ("signature_blob" in verify_args)
                }, agent="verification_agent")

                verify_result = await call_tool_on_session(mcp, "verify_signature", verify_args)

                # DEBUG: Log raw MCP response
                print(f"\n🔍 DEBUG: MCP verify_signature FULL response for signature {idx}:")
                import json
                print("="*80)
                print(f"success: {verify_result.get('success')}")
                print(f"error: {verify_result.get('error', 'N/A')}")
                print("\nverification dict:")
                print(json.dumps(verify_result.get("verification", {}), indent=2, default=str))
                print("\nmetrics_score dict:")
                print(json.dumps(verify_result.get("metrics_score", {}), indent=2, default=str))
                print("="*80 + "\n")

                if verify_result.get("success"):
                    # Extract thinking metadata from this verification
                    if verify_result.get("thinking"):
                        thinking_traces.append(verify_result.get("thinking"))
                    
                    verification = _convert_verification(
                        verify_result, 
                        detection, 
                        idx, 
                        reference_blob, 
                        reference_mime,
                        customer_id
                    )
                    verifications.append(verification)
                    state.add_history("verification", "signature_verified", {
                        "index": idx,
                        "recommendation": verification.recommendation,
                        "confidence": verification.confidence,
                        "fiv_score": verify_result.get("fiv_score")
                    }, agent="verification_agent")
                else:
                    error_msg = verify_result.get("error", "unknown")
                    print(f"❌ Verification failed for signature {idx}: {error_msg}")
                    state.add_history("verification", "verification_failed", {
                        "index": idx,
                        "error": error_msg
                    }, agent="verification_agent")

            processing_time = int((time.time() - start_time) * 1000)

            # DEBUG: Log verification results
            print("\n" + "="*80)
            print("🔍 VERIFICATION NODE - Results")
            print("="*80)
            print(f"Total verifications: {len(verifications)}")
            for i, ver in enumerate(verifications):
                print(f"\nVerification {i+1}:")
                print(f"  - Match: {ver.match}")
                print(f"  - Confidence: {ver.confidence:.2%}")
                print(f"  - Recommendation: {ver.recommendation}")
                print(f"  - Reasoning: {ver.reasoning[:100]}..." if ver.reasoning else "")
                if ver.metrics:
                    print(f"  - Metrics: {list(ver.metrics.keys())}")
                if ver.scoring_details:
                    print(f"  - Scoring: vetoed={ver.scoring_details.get('vetoed')}, final={ver.scoring_details.get('final_score'):.2f}")
            print("="*80 + "\n")
            
            # Aggregate thinking metadata from all verifications
            thinking_metadata = {}
            if thinking_traces:
                # Combine all thinking traces
                all_thoughts = []
                total_thoughts_tokens = 0
                total_thinking_budget = 0
                
                for trace in thinking_traces:
                    if trace.get("thoughts"):
                        all_thoughts.extend(trace["thoughts"])
                    if trace.get("thoughts_token_count"):
                        total_thoughts_tokens += trace["thoughts_token_count"]
                    if trace.get("thinking_budget_used"):
                        total_thinking_budget += trace["thinking_budget_used"]
                
                thinking_metadata = {
                    "thoughts": all_thoughts if all_thoughts else None,
                    "thoughts_token_count": total_thoughts_tokens if total_thoughts_tokens else None,
                    "thinking_budget_used": total_thinking_budget if total_thinking_budget else None
                }
            
            # Record successful attempt
            attempt = VerificationAttempt(
                attempt_number=attempt_number,
                success=True,
                results=verifications,
                model_used=config.llm.gemini.model,
                processing_time_ms=processing_time,
                thoughts=thinking_metadata.get("thoughts"),
                thoughts_token_count=thinking_metadata.get("thoughts_token_count"),
                thinking_budget_used=thinking_metadata.get("thinking_budget_used")
            )
            state.add_verification_attempt(attempt)

            # Aggregate decision
            overall_decision = _aggregate_decision(verifications)
            state.add_history("verification", "completed", {
                "attempt": attempt_number,
                "signatures_verified": len(verifications),
                "overall_decision": overall_decision,
                "processing_time_ms": processing_time
            }, agent="verification_agent")

    except Exception as e:
        import traceback
        processing_time = int((time.time() - start_time) * 1000)
        error_msg = f"Verification failed: {str(e)}"

        attempt = VerificationAttempt(
            attempt_number=attempt_number,
            success=False,
            errors=[error_msg],
            model_used=config.llm.gemini.model,
            processing_time_ms=processing_time
        )
        state.add_verification_attempt(attempt)
        state.add_history("verification", "failed", {
            "error": error_msg,
            "attempt": attempt_number,
            "traceback": traceback.format_exc()[:1000]
        }, agent="verification_agent")

        if state.retry_count < state.max_retries:
            state.retry_count += 1

    return state


def _convert_verification(
    result: dict, 
    detection, 
    idx: int, 
    reference_blob: str = None, 
    reference_mime: str = None,
    customer_id: str = None
) -> SignatureVerification:
    """Convert MCP verify_signature response to SignatureVerification model."""
    # The MCP tool returns a 'verification' dict that already matches the model
    verification_data = result.get("verification", {})
    metrics_score = result.get("metrics_score", {})
    
    # Extract metrics and scoring details from FIV 1.0 result
    # Send FULL metrics objects with all details (score, status, notes, execution, etc.)
    metrics_dict = None
    scoring_details = None
    if metrics_score:
        # Keep the complete metrics objects for detailed frontend display
        metrics_dict = metrics_score.get("metrics", {})
        
        # Extract scoring details with full audit information
        veto_info = metrics_score.get("veto", {})
        scoring_details = {
            "vetoed": veto_info.get("vetoed", False),
            "veto_reason": veto_info.get("veto_reason", ""),
            "veto_metric": veto_info.get("veto_metric", ""),
            "base_score": metrics_score.get("base_score", 100.0),
            "penalties": metrics_score.get("penalties", []),
            "bonuses": metrics_score.get("bonuses", []),
            "penalties_applied": sum(p.get("amount", 0) for p in metrics_score.get("penalties", [])),
            "bonuses_applied": sum(b.get("amount", 0) for b in metrics_score.get("bonuses", [])),
            "final_score": metrics_score.get("final_confidence", 0.0),
            "fiv_version": metrics_score.get("fiv_version", "FIV-1.0"),
            "confidence_band": metrics_score.get("confidence_band", "UNKNOWN"),
            "decision": metrics_score.get("decision", "UNKNOWN"),
            "decision_reason": metrics_score.get("decision_reason", ""),
            "llm_model": metrics_score.get("llm_model", "unknown"),
            "processing_time_ms": metrics_score.get("processing_time_ms", 0),
            "audit_summary": metrics_score.get("audit_summary", {})
        }
    
    # Build reference signatures list
    reference_signatures = []
    if reference_blob:
        ref_sig = ReferenceSignature(
            reference_id=customer_id or f"ref_{idx}",
            blob=reference_blob,
            mime_type=reference_mime or "image/png",
            customer_id=customer_id,
            match_score=float(verification_data.get("confidence", 0.0))
        )
        reference_signatures.append(ref_sig)
    
    sig_verification = SignatureVerification(
        match=verification_data.get("match", False),
        confidence=float(verification_data.get("confidence", 0.0)),
        reasoning=verification_data.get("reasoning", ""),
        reference_signature_id=customer_id or f"ref_{idx}",
        similarity_factors=SimilarityFactors(**verification_data.get("similarity_factors", {})) if verification_data.get("similarity_factors") else None,
        risk_indicators=verification_data.get("risk_indicators", []),
        recommendation=verification_data.get("recommendation", "MANUAL_REVIEW"),
        signature_blob=detection.image_blob,
        reference_blob=reference_blob,  # Keep for backward compatibility
        blob_mime_type=detection.blob_mime_type or "image/png",
        reference_signatures=reference_signatures,  # New list structure
        metrics=metrics_dict,
        scoring_details=scoring_details
    )
    
    print(f"🔍 DEBUG: Created SignatureVerification with {len(reference_signatures)} reference signature(s)")
    return sig_verification


def _aggregate_decision(verifications: list) -> str:
    """Aggregate individual recommendations into overall document decision."""
    if not verifications:
        return "no_signatures"

    recommendations = [v.recommendation for v in verifications]

    if all(r == "APPROVE" for r in recommendations):
        return "approved"
    elif any(r == "REJECT" for r in recommendations):
        return "rejected"
    else:
        return "needs_review"
