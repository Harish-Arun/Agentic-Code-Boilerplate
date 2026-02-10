"""
Gemini REST API Adapter - Direct REST calls to Gemini API.

This adapter uses direct REST API calls instead of the google-generativeai library,
giving more control over requests and better compatibility in containerized environments.
"""
import os
import base64
import json
import asyncio
import httpx
from typing import Dict, List, Optional, Any, Union
from pathlib import Path


class GeminiRestAdapter:
    """
    Gemini Vision adapter using REST API directly.
    
    Supports:
    - Text generation
    - Vision/image analysis
    - Structured JSON output
    - PDF file processing
    """
    
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-1.5-flash",  # Valid stable model
        temperature: float = 0.0,  # 0 for deterministic, consistent output
        max_tokens: int = 16384  # Increased to prevent truncation in M1-M7 responses
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required. Set it in environment or pass to constructor.")
    
    def _get_mime_type(self, file_path: str) -> str:
        """Determine MIME type from file extension."""
        ext = Path(file_path).suffix.lower()
        mime_types = {
            ".pdf": "application/pdf",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".bmp": "image/bmp"
        }
        return mime_types.get(ext, "application/octet-stream")
    
    def _encode_file(self, file_path: str) -> tuple[str, str]:
        """Read and base64 encode a file. Returns (base64_data, mime_type)."""
        with open(file_path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return data, self._get_mime_type(file_path)
    
    def _encode_bytes(self, data: bytes, mime_type: str = "image/png") -> str:
        """Base64 encode bytes."""
        return base64.standard_b64encode(data).decode("utf-8")
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text response from a prompt."""
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
            contents.append({
                "role": "model",
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        
        contents.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }
        
        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 30.0  # Increased to 30s for rate limits
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=120.0) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                
                # Extract text from response
                return result["candidates"][0]["content"]["parts"][0]["text"]
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⏳ Rate limit hit. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def generate_with_vision(
        self,
        prompt: str,
        files: List[Union[str, bytes, tuple[bytes, str]]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate response with file/image inputs.
        
        Args:
            prompt: Text prompt
            files: List of file paths, bytes, or (bytes, mime_type) tuples
            system_prompt: Optional system instructions
        
        Returns:
            Generated text response
        """
        url = f"{self.BASE_URL}/models/{self.model}:generateContent"
        
        # Build parts with files
        parts = []
        
        # Add files first
        for file_input in files:
            if isinstance(file_input, str):
                # File path
                data, mime_type = self._encode_file(file_input)
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": data
                    }
                })
            elif isinstance(file_input, tuple):
                # (bytes, mime_type)
                data, mime_type = file_input
                parts.append({
                    "inlineData": {
                        "mimeType": mime_type,
                        "data": self._encode_bytes(data, mime_type)
                    }
                })
            else:
                # Raw bytes, assume PNG
                parts.append({
                    "inlineData": {
                        "mimeType": "image/png",
                        "data": self._encode_bytes(file_input)
                    }
                })
        
        # Add text prompt
        parts.append({"text": prompt})
        
        contents = []
        if system_prompt:
            contents.append({
                "role": "user",
                "parts": [{"text": system_prompt}]
            })
            contents.append({
                "role": "model", 
                "parts": [{"text": "Understood. I will follow these instructions."}]
            })
        
        contents.append({
            "role": "user",
            "parts": parts
        })
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }
        
        # Retry logic for rate limits
        max_retries = 3
        retry_delay = 30.0  # Increased to 30s for rate limits
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=180.0) as client:
                    response = await client.post(
                        url,
                        json=payload,
                        headers={
                            "Content-Type": "application/json",
                            "x-goog-api-key": self.api_key
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                
                return result["candidates"][0]["content"]["parts"][0]["text"]
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⏳ Rate limit hit. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    raise
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        files: List[Union[str, bytes]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.
        
        Args:
            prompt: Text prompt describing what to extract
            schema: JSON schema for expected output
            system_prompt: Optional system instructions
            files: Optional list of files to analyze
        
        Returns:
            Parsed JSON response matching schema
        """
        structured_prompt = f"""{prompt}

IMPORTANT: Return your response as valid JSON only. No markdown code blocks, no explanations.
The JSON should match this structure:
{json.dumps(schema, indent=2)}

Respond with ONLY the JSON object, nothing else."""

        if files:
            response = await self.generate_with_vision(structured_prompt, files, system_prompt)
        else:
            response = await self.generate(structured_prompt, system_prompt)
        
        # Clean response - remove markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith("```"):
            # Remove markdown code block
            lines = clean_response.split("\n")
            # Find first and last ``` and extract content between
            start_idx = 1 if lines[0].startswith("```") else 0
            end_idx = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            clean_response = "\n".join(lines[start_idx:end_idx])
        
        # Try multiple parsing strategies
        try:
            return json.loads(clean_response)
        except json.JSONDecodeError as e:
            # Strategy 1: Try to find JSON object in response
            import re
            json_match = re.search(r'\{[\s\S]*\}', clean_response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Strategy 2: Try to fix common JSON issues (trailing commas, incomplete objects)
            try:
                # Remove trailing commas before closing braces/brackets
                fixed_json = re.sub(r',(\s*[}\]])', r'\1', clean_response)
                
                # Count braces to detect incomplete JSON
                open_braces = fixed_json.count('{')
                close_braces = fixed_json.count('}')
                if open_braces > close_braces:
                    # Response was truncated - add missing closing braces
                    fixed_json = fixed_json.rstrip(',\n ')
                    for _ in range(open_braces - close_braces):
                        fixed_json += '\n}'
                
                return json.loads(fixed_json)
            except json.JSONDecodeError:
                pass
            
            # Strategy 3: Log the full problematic response for debugging
            print(f"⚠️ JSON Parse Error at position {e.pos}:")
            print(f"   Error: {str(e)}")
            print(f"   Context: ...{clean_response[max(0, e.pos-50):min(len(clean_response), e.pos+50)]}...")
            print(f"   Full response length: {len(response)} chars")
            print(f"\n📄 Full Response (for debugging):")
            print("="*80)
            print(clean_response)
            print("="*80)
            
            # Return empty structure matching schema to avoid complete failure
            if "signatures" in schema:
                return {"signatures": [], "total_signatures_found": 0, "has_empty_signature_fields": False, "notes": f"JSON parsing failed: {str(e)}"}
            
            raise ValueError(f"Failed to parse JSON from response: {response[:500]}")
    
    async def extract_payment_fields(
        self,
        file_path: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract payment instruction fields from a document.
        
        Args:
            file_path: Path to PDF or image file
            system_prompt: System instruction for the extraction task
            user_prompt: User prompt describing the extraction requirements
        
        Returns:
            Extracted payment fields with confidence scores
        """
        # Use provided prompts or fallback to basic defaults
        _system_prompt = system_prompt or "You are a payment document extraction specialist. Be precise and accurate."
        _user_prompt = user_prompt or """Extract payment fields from this document with confidence scores."""

        schema = {
            "creditor_name": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "creditor_account": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "creditor_bank": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "debtor_name": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "debtor_account": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "debtor_bank": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "amount": {"value": "number or null", "confidence": 0.0, "location": "string"},
            "currency": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "payment_type": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "payment_date": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "charges_account": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "reference": {"value": "string or null", "confidence": 0.0, "location": "string"},
            "raw_text": "string - full OCR text of the document"
        }
        
        return await self.generate_structured(
            _user_prompt,
            schema,
            system_prompt=_system_prompt,
            files=[file_path]
        )
    
    async def detect_signatures(
        self,
        file_path: str,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        page_width: int = 612,
        page_height: int = 792
    ) -> Dict[str, Any]:
        """
        Detect signature regions in a document.
        
        Args:
            file_path: Path to PDF or image file
            system_prompt: System instruction for signature detection
            user_prompt: User prompt describing detection requirements
            page_width: Document width in points (default: US Letter)
            page_height: Document height in points (default: US Letter)
        
        Returns:
            List of detected signatures with bounding boxes
        """
        # Use provided prompts or fallback to basic defaults
        _system_prompt = system_prompt or "You are a document signature identifier."
        _user_prompt = user_prompt or """Detect signature regions in this document."""

        schema = {
            "signatures": [
                {
                    "page_number": 1,
                    "bounding_box": [0.0, 0.0, 0.0, 0.0],  # [x_min, y_min, x_max, y_max] normalized 0-1
                    "signature_type": "string",
                    "confidence": 0.0,
                    "signer_name": "string or null",
                    "description": "string"
                }
            ],
            "total_signatures_found": 0
        }
        
        return await self.generate_structured(
            _user_prompt,
            schema,
            system_prompt=_system_prompt,
            files=[file_path]
        )
    
    async def verify_signatures(
        self,
        signature_image: Union[str, bytes, tuple],
        reference_image: Union[str, bytes, tuple],
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare a signature with a reference signature using the M1–M7 metrics framework.
        
        Each metric is individually computed by the LLM and returned with raw values,
        a score (0–100), and a status. The scoring engine then applies veto checks,
        penalties, and bonuses to produce a final confidence and decision.
        
        Metrics (FIV 1.0 scope):
          M1: Global Form — bounding box aspect ratio comparison
          M2: Line Quality — tremor, hesitation, pen control
          M3: Slant Angle — dominant stroke angle delta
          M4: Baseline Stability — writing line drift and slope variance
          M5: Terminal Strokes — distinctive hooks, loops, flourishes
          M6: Spacing & Density — ink density ratio comparison
          M7: Pressure Inference — grayscale intensity as pressure proxy
        
        Args:
            signature_image: Path, raw bytes, or (bytes, mime_type) tuple of extracted signature
            reference_image: Path, raw bytes, or (bytes, mime_type) tuple of reference signature
            system_prompt: System instruction for verification
            user_prompt: User prompt describing M1-M7 framework
        
        Returns:
            Dict containing all M1–M7 metric results individually
        """
        # Use provided prompts or fallback to basic defaults
        _system_prompt = system_prompt or "You are a forensic signature analysis expert."
        _user_prompt = user_prompt or """Compare these two signatures using forensic metrics."""

        schema = {
            "M1_global_form": {
                "aspect_ratio_reference": 0.0,
                "aspect_ratio_questioned": 0.0,
                "aspect_ratio_delta": 0.0,
                "score": 0,
                "status": "PASS | WARNING | VETO",
                "notes": "string"
            },
            "M2_line_quality": {
                "tremor_detected": False,
                "hesitation_marks": 0,
                "quality_score_reference": 0,
                "quality_score_questioned": 0,
                "score": 0,
                "status": "PASS | WARNING | FAIL",
                "notes": "string"
            },
            "M3_slant_angle": {
                "slant_angle_reference": 0.0,
                "slant_angle_questioned": 0.0,
                "slant_delta_degrees": 0.0,
                "score": 0,
                "status": "PASS | WARNING | VETO",
                "notes": "string"
            },
            "M4_baseline_stability": {
                "drift_reference": 0.0,
                "drift_questioned": 0.0,
                "drift_delta": 0.0,
                "slope_variance_reference": 0.0,
                "slope_variance_questioned": 0.0,
                "score": 0,
                "status": "PASS | WARNING | FAIL",
                "notes": "string"
            },
            "M5_terminal_strokes": {
                "match_status": "MATCH | PARTIAL_MATCH | COMPLETE_MISMATCH",
                "markers_reference": ["list of quirks"],
                "markers_questioned": ["list of quirks"],
                "markers_matched": ["list of matching quirks"],
                "marker_confidence": 0.0,
                "score": 0,
                "status": "PASS | WARNING | VETO",
                "notes": "string"
            },
            "M6_spacing_density": {
                "density_reference": 0.0,
                "density_questioned": 0.0,
                "density_delta": 0.0,
                "score": 0,
                "status": "PASS | WARNING | FAIL",
                "notes": "string"
            },
            "M7_pressure_inference": {
                "pressure_mean_reference": 0.0,
                "pressure_mean_questioned": 0.0,
                "pressure_delta": 0.0,
                "variance_pct_reference": 0.0,
                "variance_pct_questioned": 0.0,
                "score": 0,
                "status": "PASS | WARNING | FAIL",
                "notes": "string"
            }
        }
        
        # Prepare files list — supports paths, bytes, or (bytes, mime_type) tuples
        files = [reference_image, signature_image]
        
        return await self.generate_structured(
            _user_prompt,
            schema,
            system_prompt=_system_prompt,
            files=files
        )


# Factory function
def get_gemini_adapter(config=None) -> GeminiRestAdapter:
    """
    Get a configured Gemini REST adapter.
    
    Args:
        config: Optional AppConfig. If not provided, uses environment variables.
    
    Returns:
        Configured GeminiRestAdapter instance
    """
    if config and hasattr(config, 'llm') and config.llm.provider == "gemini":
        gemini_cfg = config.llm.gemini
        return GeminiRestAdapter(
            api_key=gemini_cfg.api_key or os.environ.get("GEMINI_API_KEY"),
            model=gemini_cfg.model,
            temperature=gemini_cfg.temperature,
            max_tokens=gemini_cfg.max_tokens
        )
    
    # Default from environment
    return GeminiRestAdapter()
