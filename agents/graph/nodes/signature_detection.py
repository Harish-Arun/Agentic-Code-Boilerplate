"""
Signature Detection Agent Node - Thin orchestrator.

Delegates ALL business logic to MCP tools:
  - detect_signatures: Gemini Vision signature detection
  - crop_signature: PyMuPDF-based signature cropping at 300 DPI
  - validate_signature_crops: Quality & dimension validation

This node only manages state transitions, history tracking, and retry logic.
"""
import sys
import json
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from models import (
    AgentState, SignatureDetection, BoundingBox,
    SignatureDetectionAttempt
)
from config import AppConfig
from mcp_client import get_mcp_client, call_tool_on_session


async def signature_detection_node(state: AgentState, config: AppConfig) -> AgentState:
    """
    Detect and crop signatures by calling MCP detection tools.

    Flow:
    1. Call MCP detect_signatures -> Gemini Vision detection
    2. Call MCP crop_signature (loop) -> PyMuPDF cropping at 300 DPI
    3. Call MCP validate_signature_crops -> quality validation
    4. Record attempt + update state
    """
    state.current_step = "signature_detection"
    state.add_history("signature_detection", "started", {
        "document_path": state.document_path
    }, agent="signature_agent")

    start_time = time.time()
    attempt_number = len(state.detection_attempts) + 1

    try:
        async with get_mcp_client() as mcp:
            # Step 1: Detect signatures via MCP
            state.add_history("signature_detection", "calling_mcp_detect", {
                "document_path": state.document_path,
                "tool": "detect_signatures"
            }, agent="signature_agent")

            detection_result = await call_tool_on_session(mcp, "detect_signatures", {
                "document_path": state.document_path
            })

            if not detection_result.get("success"):
                error_msg = detection_result.get("error", "Detection failed")
                raise RuntimeError(error_msg)

            raw_detections = detection_result.get("detections", [])
            model_used = detection_result.get("model_used", "")
            detection_passes = detection_result.get("detection_passes", 1)

            state.add_history("signature_detection", "detections_received", {
                "count": len(raw_detections),
                "passes": detection_passes
            }, agent="signature_agent")

            # Step 2: Crop each detected signature via MCP
            # First, get actual page dimensions from PDF for proper normalization
            page_dimensions = {}
            try:
                import fitz
                doc = fitz.open(state.document_path)
                for page_num in range(doc.page_count):
                    page_obj = doc.load_page(page_num)
                    page_dimensions[page_num + 1] = {  # Store as 1-indexed
                        'width': page_obj.rect.width,
                        'height': page_obj.rect.height
                    }
                doc.close()
                print(f"📐 Loaded page dimensions from PDF: {len(page_dimensions)} pages")
                for p, d in page_dimensions.items():
                    print(f"   Page {p}: {d['width']:.2f} x {d['height']:.2f} points (approx {d['width']/72*25.4:.1f}x{d['height']/72*25.4:.1f} mm)")
            except Exception as e:
                print(f"⚠️ Warning: Could not read PDF dimensions: {e}")
                # Fallback to standard dimensions if PDF reading fails
                page_dimensions = {1: {'width': 612.0, 'height': 792.0}}
                print(f"   Using fallback dimensions (Letter): 612.0 x 792.0 points")
            
            cropped_detections = []
            for idx, det in enumerate(raw_detections):
                bbox = det.get("bounding_box", {})
                
                # Get page number - default to 1 if missing (NOT 0!)
                page_num = int(bbox.get("page", 1))
                if page_num == 0:
                    page_num = 1
                    print(f"⚠️ Page number was 0, correcting to 1")
                
                # Get actual page dimensions for this page
                page_dims = page_dimensions.get(page_num, {'width': 612.0, 'height': 792.0})
                page_width = page_dims['width']
                page_height = page_dims['height']
                
                # Auto-normalize coordinates logic
                x1, y1, x2, y2 = float(bbox.get("x1", 0)), float(bbox.get("y1", 0)), float(bbox.get("x2", 0)), float(bbox.get("y2", 0))
                
                # If coordinates are already normalized (0-1), use as is
                if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                    x1_norm, y1_norm, x2_norm, y2_norm = x1, y1, x2, y2
                else:
                    # HEURISTIC: Determine if coordinates are 1000-scale (Gemini default) or Pixels (PDF points)
                    # If any coordinate is > 1000, it MUST be pixels (since 1000-scale max is 1000)
                    # If page dimensions are small (e.g. < 1000 width/height), pixels are < 1000 too.
                    # But Gemini almost always outputs 0-1000 integers if not 0-1 floats.
                    
                    is_likely_1000_scale = (x1 <= 1000 and y1 <= 1000 and x2 <= 1000 and y2 <= 1000)
                    
                    if is_likely_1000_scale:
                         # Assume 1000-scale (standard for many Vision APIs including Gemini often)
                         x1_norm = x1 / 1000.0
                         y1_norm = y1 / 1000.0
                         x2_norm = x2 / 1000.0
                         y2_norm = y2 / 1000.0
                         print(f"⚠️ Detected 1000-scale coordinates. Normalizing by 1000.")
                    else:
                         # Must be pixels for a large document
                         x1_norm = x1 / page_width
                         y1_norm = y1 / page_height
                         x2_norm = x2 / page_width
                         y2_norm = y2 / page_height
                         print(f"⚠️ Detected large pixel coordinates. Normalizing by page size ({page_width}x{page_height})")

                # Clamp finally
                x1_norm = max(0.0, min(1.0, x1_norm))
                y1_norm = max(0.0, min(1.0, y1_norm))
                x2_norm = max(0.0, min(1.0, x2_norm))
                y2_norm = max(0.0, min(1.0, y2_norm))

                # Removed heuristic crop padding to ensure "pixel-perfect" match with detection.
                # If padding is desired, detection logic/prompts should handle it upstream.
                # box_w = max(0.0, x2_norm - x1_norm)
                # ... (padding removed)

                # Keep detection bbox aligned with the exact region used for crop,
                # so PDF highlight and cropped signature represent the same area.
                det["bounding_box"] = {
                    "x1": x1_norm,
                    "y1": y1_norm,
                    "x2": x2_norm,
                    "y2": y2_norm,
                    "page": page_num
                }
                
                # DEBUG: Print bounding box before cropping
                print(f"\n🔍 Cropping signature {idx}: normalized bbox=[{x1_norm:.3f}, {y1_norm:.3f}, {x2_norm:.3f}, {y2_norm:.3f}] page={page_num}")
                
                crop_result = await call_tool_on_session(mcp, "crop_signature", {
                    "document_path": state.document_path,
                    "bbox_x1": x1_norm,
                    "bbox_y1": y1_norm,
                    "bbox_x2": x2_norm,
                    "bbox_y2": y2_norm,
                    "page": page_num,
                    "document_id": state.document_id or "unknown",
                    "sig_index": idx
                })

                if crop_result.get("success"):
                    det["cropped_image_path"] = crop_result["cropped_image_path"]
                    det["image_blob"] = crop_result.get("image_blob")
                    det["blob_mime_type"] = crop_result.get("blob_mime_type", "image/png")
                    state.add_history("signature_detection", "signature_cropped", {
                        "index": idx,
                        "cropped_path": crop_result["cropped_image_path"],
                        "blob_size_bytes": crop_result.get("blob_size_bytes", 0)
                    }, agent="signature_agent")
                    print(f"✅ Crop successful: {crop_result['cropped_image_path']}")
                else:
                    error_msg = crop_result.get("error", "unknown")
                    print(f"❌ Crop failed: {error_msg}")
                    state.add_history("signature_detection", "crop_failed", {
                        "index": idx,
                        "error": error_msg
                    }, agent="signature_agent")

                cropped_detections.append(det)

            # Step 3: Validate crops via MCP
            validation_result = await call_tool_on_session(mcp, "validate_signature_crops", {
                "detections_json": json.dumps(cropped_detections)
            })

            if not validation_result.get("valid", True):
                state.add_history("signature_detection", "validation_warning", {
                    "issues": validation_result.get("issues", [])
                }, agent="challenger_agent")

            # Convert to models
            detections = _convert_detections(cropped_detections)
            processing_time = int((time.time() - start_time) * 1000)
            thinking_metadata = detection_result.get("thinking", {})

            # Record successful attempt
            attempt = SignatureDetectionAttempt(
                attempt_number=attempt_number,
                success=True,
                detections=detections,
                model_used=model_used,
                processing_time_ms=processing_time,
                raw_response=str(raw_detections)[:2000],
                thoughts=thinking_metadata.get("thoughts"),
                thoughts_token_count=thinking_metadata.get("thoughts_token_count"),
                thinking_budget_used=thinking_metadata.get("thinking_budget_used")
            )
            state.add_detection_attempt(attempt)

            # DEBUG: Log detection results
            print("\n" + "="*80)
            print("✍️  SIGNATURE DETECTION NODE - Signatures Found")
            print("="*80)
            print(f"Total signatures detected: {len(detections)}")
            for i, det in enumerate(detections):
                print(f"\nSignature {i+1}:")
                print(f"  - ID: {det.signature_id or 'N/A'}")
                print(f"  - Confidence: {det.confidence:.2f}")
                print(f"  - Page: {det.page}")
                print(f"  - Has blob: {det.image_blob is not None}")
                print(f"  - Has path: {det.cropped_image_path is not None}")
                print(f"  - Signer: {det.signer_name or 'unknown'}")
            print("="*80 + "\n")
            
            state.add_history("signature_detection", "completed", {
                "attempt": attempt_number,
                "signatures_found": len(detections),
                "detection_passes": detection_passes,
                "processing_time_ms": processing_time
            }, agent="signature_agent")

    except Exception as e:
        import traceback
        processing_time = int((time.time() - start_time) * 1000)
        error_msg = f"Signature detection failed: {str(e)}"

        attempt = SignatureDetectionAttempt(
            attempt_number=attempt_number,
            success=False,
            errors=[error_msg],
            model_used=config.llm.gemini.model,
            processing_time_ms=processing_time
        )
        state.add_detection_attempt(attempt)
        state.add_history("signature_detection", "failed", {
            "error": error_msg,
            "attempt": attempt_number,
            "traceback": traceback.format_exc()[:1000]
        }, agent="signature_agent")

        if state.retry_count < state.max_retries:
            state.retry_count += 1

    return state


def _convert_detections(raw_list: list) -> List[SignatureDetection]:
    """Convert MCP response dicts to SignatureDetection models."""
    detections = []
    for det in raw_list:
        bbox_data = det.get("bounding_box", {})
        bbox = BoundingBox(
            x1=float(bbox_data.get("x1", 0)),
            y1=float(bbox_data.get("y1", 0)),
            x2=float(bbox_data.get("x2", 0)),
            y2=float(bbox_data.get("y2", 0)),
            page=int(bbox_data.get("page", 1))  # Page is INSIDE bounding_box dict
        )
        detections.append(SignatureDetection(
            signature_id=det.get("signature_id", f"sig_{len(detections)}"),
            confidence=float(det.get("confidence", 0.0)),
            bounding_box=bbox,
            page=int(bbox_data.get("page", 1)),  # Page is INSIDE bounding_box dict
            cropped_image_path=det.get("cropped_image_path"),
            image_blob=det.get("image_blob"),
            blob_mime_type=det.get("blob_mime_type", "image/png"),
            signer_role=det.get("signer_role"),
            signer_name=det.get("signer_name")
        ))
    return detections
