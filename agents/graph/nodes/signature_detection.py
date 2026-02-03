"""
Signature Detection Agent Node - Detect signature regions in documents.

This node identifies where signatures appear in the document.
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from models import AgentState, SignatureDetection, BoundingBox
from config import AppConfig
from adapters import get_llm_adapter


async def signature_detection_node(state: AgentState, config: AppConfig) -> AgentState:
    """
    Detect signature regions in the document.
    
    This agent:
    1. Analyzes each page of the document
    2. Identifies regions that contain signatures
    3. Returns bounding boxes with confidence scores
    
    In mock mode, returns sample bounding boxes.
    In production, uses Gemini Vision for detection.
    """
    state.current_step = "signature_detection"
    
    try:
        llm = get_llm_adapter(config)
        
        if config.llm.provider == "mock":
            data_dir = os.environ.get("DATA_DIR", "/data")
            # Return mock signature detections
            state.signature_detections = [
                SignatureDetection(
                    bounding_box=BoundingBox(
                        x1=450.0,
                        y1=680.0,
                        x2=600.0,
                        y2=720.0,
                        page=1
                    ),
                    confidence=0.92,
                    cropped_image_path=f"{Path(data_dir).as_posix()}/signatures/doc123_sig1.png"
                ),
                SignatureDetection(
                    bounding_box=BoundingBox(
                        x1=450.0,
                        y1=750.0,
                        x2=600.0,
                        y2=790.0,
                        page=1
                    ),
                    confidence=0.88,
                    cropped_image_path=f"{Path(data_dir).as_posix()}/signatures/doc123_sig2.png"
                )
            ]
        else:
            # Production: Use vision LLM
            detection_prompt = config.prompts.signature_detection
            
            result = await llm.generate_structured(
                prompt=f"{detection_prompt}\n\nDocument path: {state.document_path}",
                schema={
                    "signatures": [
                        {
                            "x1": "number",
                            "y1": "number",
                            "x2": "number",
                            "y2": "number",
                            "page": "number",
                            "confidence": "number"
                        }
                    ]
                }
            )
            
            # Convert to SignatureDetection models
            signatures = result.get("signatures", [])
            state.signature_detections = [
                SignatureDetection(
                    bounding_box=BoundingBox(
                        x1=sig.get("x1", 0),
                        y1=sig.get("y1", 0),
                        x2=sig.get("x2", 0),
                        y2=sig.get("y2", 0),
                        page=sig.get("page", 1)
                    ),
                    confidence=sig.get("confidence", 0.5)
                )
                for sig in signatures
            ]
    
    except Exception as e:
        state.detection_errors.append(f"Signature detection failed: {str(e)}")
    
    return state
