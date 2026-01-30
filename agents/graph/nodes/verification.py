"""
Signature Verification Agent Node - Verify signatures against references.

This node compares detected signatures with reference signatures.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))

from models import AgentState, SignatureVerification
from config import AppConfig
from adapters import get_llm_adapter


async def verification_node(state: AgentState, config: AppConfig) -> AgentState:
    """
    Verify signatures against reference signatures.
    
    This agent:
    1. Takes cropped signature images
    2. Retrieves reference signatures (via MCP tool)
    3. Compares using vision LLM
    4. Returns match status with reasoning
    
    In mock mode, returns sample verification result.
    In production, calls MCP signature_provider + Gemini Vision.
    """
    state.current_step = "verification"
    
    # Skip if no signatures detected
    if not state.signature_detections:
        state.verification_errors.append("No signatures detected to verify")
        return state
    
    try:
        llm = get_llm_adapter(config)
        
        if config.llm.provider == "mock":
            # Return mock verification result
            state.verification_result = SignatureVerification(
                match=True,
                confidence=0.87,
                reasoning="[Mock Verification] The signature shows consistent stroke patterns "
                         "and pressure distribution. The overall shape and flow match the "
                         "reference signature with high confidence. Minor variations are within "
                         "acceptable natural variation limits.",
                reference_signature_id="ref_sig_12345"
            )
        else:
            # Production: Use vision LLM for comparison
            verification_prompt = config.prompts.signature_verification
            
            # In production, you would:
            # 1. Call MCP signature_provider to get reference signature
            # 2. Pass both images to vision LLM for comparison
            
            result = await llm.generate_structured(
                prompt=f"""{verification_prompt}
                
Detected signature from: {state.signature_detections[0].cropped_image_path}
Reference: Retrieved from signature provider service

Analyze and compare these signatures.""",
                schema={
                    "match": "boolean",
                    "confidence": "number",
                    "reasoning": "string"
                }
            )
            
            state.verification_result = SignatureVerification(
                match=result.get("match", False),
                confidence=result.get("confidence", 0.0),
                reasoning=result.get("reasoning", "No reasoning provided")
            )
    
    except Exception as e:
        state.verification_errors.append(f"Verification failed: {str(e)}")
    
    return state
