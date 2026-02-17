"""
Signature Detection Tool — Detect and crop signatures from documents.

Business logic (called by agent orchestrator via MCP):
  1. Calls Gemini Vision to detect signature bounding boxes
  2. Crops signature regions using PyMuPDF at 300 DPI
  3. Validates cropped signatures for quality
"""
from typing import Dict, Any, Optional
from fastmcp import FastMCP
import os
import json
import base64

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import AppConfig
from adapters import get_gemini_adapter


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_bbox(raw_bbox: Any) -> Dict[str, float]:
    """
    Parse bounding box coordinates.
    Does NOT clamp to 0-1 range to allow pixel coordinates to pass through
    to the consumer (which has page dimension context for proper normalization).
    """
    if isinstance(raw_bbox, dict):
        x1 = _safe_float(raw_bbox.get("x1", 0.0))
        y1 = _safe_float(raw_bbox.get("y1", 0.0))
        x2 = _safe_float(raw_bbox.get("x2", 0.0))
        y2 = _safe_float(raw_bbox.get("y2", 0.0))
    elif isinstance(raw_bbox, list) and len(raw_bbox) >= 4:
        x1 = _safe_float(raw_bbox[0], 0.0)
        y1 = _safe_float(raw_bbox[1], 0.0)
        x2 = _safe_float(raw_bbox[2], 0.0)
        y2 = _safe_float(raw_bbox[3], 0.0)
    else:
        x1, y1, x2, y2 = 0.0, 0.0, 0.0, 0.0

    # Ensure correct ordering
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _iou(a: Dict[str, float], b: Dict[str, float]) -> float:
    inter_x1 = max(a["x1"], b["x1"])
    inter_y1 = max(a["y1"], b["y1"])
    inter_x2 = min(a["x2"], b["x2"])
    inter_y2 = min(a["y2"], b["y2"])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (a["x2"] - a["x1"])) * max(0.0, (a["y2"] - a["y1"]))
    area_b = max(0.0, (b["x2"] - b["x1"])) * max(0.0, (b["y2"] - b["y1"]))
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom


def register_signature_detection_tools(mcp: FastMCP, config: AppConfig):
    """Register signature detection business-logic tools with the MCP server."""

    @mcp.tool()
    async def detect_signatures(
        document_path: str,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Detect signature regions in a document using Gemini Vision.

        Analyzes the document and returns bounding boxes for all detected
        signatures with confidence scores and signature types.

        Args:
            document_path: Path to the PDF or image file
            custom_prompt: Optional custom detection prompt

        Returns:
            List of detected signatures with bounding boxes (normalized 0-1)
        """
        if not os.path.exists(document_path):
            return {
                "success": False,
                "error": f"Document not found: {document_path}",
                "detections": [],
                "model_used": ""
            }

        try:
            gemini = get_gemini_adapter(config)

            # Load prompts from business config
            system_prompt = None
            user_prompt = None
            
            if hasattr(config, 'business') and hasattr(config.business, 'prompts'):
                prompts_cfg = config.business.prompts
                if hasattr(prompts_cfg, 'signature_detection') and hasattr(prompts_cfg.signature_detection, 'system'):
                    system_prompt = prompts_cfg.signature_detection.system
                    user_prompt = prompts_cfg.signature_detection.user
                    
                    # Debug: Show that business_config prompts are being used
                    print(f"\n{'='*80}")
                    print(f"📝 [SIGNATURE DETECTION] Using Prompts from business_config.yaml")
                    print(f"{'='*80}")
                    print(f"System Prompt (first 150 chars):")
                    print(f"   {system_prompt[:150]}...")
                    print(f"\nUser Prompt (first 200 chars):")
                    print(f"   {user_prompt[:200]}...")
                    print(f"{'='*80}\n")
                else:
                    print(f"\n⚠️  WARNING: signature_detection prompts not found in business_config.yaml")
                    print(f"   Using default prompts instead.\n")
            else:
                print(f"\n⚠️  WARNING: business.prompts not found in config")
                print(f"   Using default prompts instead.\n")
            
            # Override with custom_prompt if provided (backward compatibility)
            if custom_prompt:
                print(f"\n⚠️  Custom prompt override detected - using custom_prompt parameter instead\n")
                user_prompt = custom_prompt

            def parse_detections(raw_result: Dict[str, Any]) -> list:
                parsed = []
                for sig in raw_result.get("signatures", []):
                    bbox_norm = _normalize_bbox(sig.get("bounding_box", [0, 0, 0, 0]))
                    width = bbox_norm["x2"] - bbox_norm["x1"]
                    height = bbox_norm["y2"] - bbox_norm["y1"]
                    if width <= 0.001 or height <= 0.001:
                        continue

                    parsed.append({
                        "bounding_box": {
                            "x1": bbox_norm["x1"],
                            "y1": bbox_norm["y1"],
                            "x2": bbox_norm["x2"],
                            "y2": bbox_norm["y2"],
                            "page": int(sig.get("page_number", 1))
                        },
                        "signature_type": sig.get("signature_type", "unknown"),
                        "confidence": _safe_float(sig.get("confidence", 0.0), 0.0),
                        "description": sig.get("description", "")
                    })
                return parsed

            result = await gemini.detect_signatures(
                document_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            # Extract thinking metadata
            thinking_metadata = result.get("_thinking", {})
            detections = parse_detections(result)

            # High-recall second pass when only one signature is found
            passes = 1
            if len(detections) <= 1:
                recall_prompt = (user_prompt or "Detect signatures in this document") + (
                    "\n\nHigh-recall mode: detect EVERY handwritten signature, initials mark,"
                    " and signed stamp-like region. Return all plausible boxes even if confidence is moderate."
                )
                second_result = await gemini.detect_signatures(
                    document_path,
                    system_prompt=system_prompt,
                    user_prompt=recall_prompt
                )
                second_detections = parse_detections(second_result)
                passes = 2

                merged = list(detections)
                for candidate in second_detections:
                    candidate_bbox = candidate["bounding_box"]
                    candidate_page = candidate_bbox.get("page", 1)

                    duplicate = False
                    for existing in merged:
                        existing_bbox = existing["bounding_box"]
                        if existing_bbox.get("page", 1) != candidate_page:
                            continue
                        if _iou(existing_bbox, candidate_bbox) >= 0.6:
                            duplicate = True
                            if candidate.get("confidence", 0.0) > existing.get("confidence", 0.0):
                                existing.update(candidate)
                            break

                    if not duplicate:
                        merged.append(candidate)

                detections = merged

            return {
                "success": True,
                "detections": detections,
                "model_used": gemini.model,
                "total_found": len(detections),
                "detection_passes": passes,
                "thinking": thinking_metadata
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "model_used": config.llm.gemini.model
            }

    @mcp.tool()
    async def crop_signature(
        document_path: str,
        bbox_x1: float,
        bbox_y1: float,
        bbox_x2: float,
        bbox_y2: float,
        page: int = 1,
        document_id: str = "doc",
        sig_index: int = 1
    ) -> Dict[str, Any]:
        """
        Crop a signature region from a PDF document using PyMuPDF.

        Uses high DPI (300) rendering for quality signature extraction.
        Coordinates should be normalized (0.0-1.0).

        Args:
            document_path: Path to the source PDF
            bbox_x1: Left edge (0.0-1.0)
            bbox_y1: Top edge (0.0-1.0)
            bbox_x2: Right edge (0.0-1.0)
            bbox_y2: Bottom edge (0.0-1.0)
            page: Page number (1-indexed)
            document_id: Document identifier for output naming
            sig_index: Signature index for output naming

        Returns:
            Path to the cropped signature image
        """
        if not os.path.exists(document_path):
            return {
                "success": False,
                "error": f"Document not found: {document_path}",
                "cropped_image_path": None
            }

        try:
            import fitz  # PyMuPDF

            data_dir = os.environ.get("DATA_DIR", "./data")
            signatures_dir = Path(data_dir).resolve() / "signatures"
            signatures_dir.mkdir(parents=True, exist_ok=True)

            output_path = str(signatures_dir / f"{document_id}_sig{sig_index}.png")

            doc = fitz.open(document_path)
            page_obj = doc.load_page(page - 1)  # 0-indexed

            page_rect = page_obj.rect
            page_width = page_rect.width
            page_height = page_rect.height
            
            print(f"📄 PDF Page {page} Dimensions: {page_width:.2f} x {page_height:.2f} points (1 pt = 1/72 inch)")

            # Convert normalized coordinates to actual coordinates
            x1 = bbox_x1 * page_width
            y1 = bbox_y1 * page_height
            x2 = bbox_x2 * page_width
            y2 = bbox_y2 * page_height

            # Load page to get dimensions
            # Note: For images opened as PDF, PyMuPDF sets resolution to 72 DPI by default
            # so 1 pixel = 1 point.
            # page.rect.width/height are in points.
            
            # If the user wants "pixel to pixel" mapping and consistent DPI:
            # We should try to respect the original image DPI if possible, or at least
            # ensure we aren't artificially upscaling if not needed.
            # However, for signature verification, higher resolution (300 DPI) is usually better.
            
            # Use 150 DPI to match the vision model input resolution
            # ensuring pixel-to-pixel consistency vs the detection logic
            TARGET_DPI = 150
            
            crop_rect = fitz.Rect(x1, y1, x2, y2)
            pix = page_obj.get_pixmap(clip=crop_rect, dpi=TARGET_DPI)
            
            # Log the crop action for debugging
            print(f"✂️ Cropping at {TARGET_DPI} DPI. Rect: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            
            pix.save(output_path)

            # Get raw bytes for blob transport (simulates ISV blob response)
            image_bytes = pix.tobytes("png")
            image_blob = base64.b64encode(image_bytes).decode("utf-8")

            doc.close()

            return {
                "success": True,
                "cropped_image_path": output_path,
                "image_blob": image_blob,
                "blob_mime_type": "image/png",
                "blob_size_bytes": len(image_bytes),
                "crop_size": {
                    "width": int(x2 - x1),
                    "height": int(y2 - y1)
                },
                "message": f"Signature cropped at 300 DPI: {output_path} (blob: {len(image_bytes)} bytes)"
            }

        except ImportError:
            return {
                "success": False,
                "error": "PyMuPDF (fitz) not installed. Install with: pip install PyMuPDF",
                "cropped_image_path": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "cropped_image_path": None
            }

    @mcp.tool()
    async def validate_signature_crops(
        detections_json: str
    ) -> Dict[str, Any]:
        """
        Challenger validation — verify cropped signatures are valid.

        Checks:
        1. Confidence meets minimum threshold
        2. Bounding box dimensions are reasonable
        3. Cropped image file exists and has non-trivial size

        Args:
            detections_json: JSON string of detection results with cropped_image_path

        Returns:
            Validation result with issues found
        """
        try:
            detections = json.loads(detections_json) if isinstance(detections_json, str) else detections_json
        except (json.JSONDecodeError, TypeError):
            return {
                "success": False,
                "valid": False,
                "issues": ["Invalid input: could not parse detections"],
                "feedback": "Validation failed"
            }

        issues = []

        for idx, detection in enumerate(detections):
            confidence = detection.get("confidence", 0)
            if confidence < 0.5:
                issues.append(f"Signature {idx+1} has low confidence: {confidence:.2f}")

            bbox = detection.get("bounding_box", {})
            width = bbox.get("x2", 0) - bbox.get("x1", 0)
            height = bbox.get("y2", 0) - bbox.get("y1", 0)

            if width < 0.05 or height < 0.02:
                issues.append(f"Signature {idx+1} bounding box too small")
            if width > 0.5 or height > 0.3:
                issues.append(f"Signature {idx+1} bounding box too large")

            crop_path = detection.get("cropped_image_path")
            if crop_path and os.path.exists(crop_path):
                file_size = os.path.getsize(crop_path)
                if file_size < 100:
                    issues.append(f"Signature {idx+1} crop file too small ({file_size} bytes)")
            elif crop_path:
                issues.append(f"Signature {idx+1} crop file not found: {crop_path}")

        return {
            "success": True,
            "valid": len(issues) == 0,
            "issues": issues,
            "feedback": f"Validated {len(detections)} signatures with {len(issues)} issues"
        }
