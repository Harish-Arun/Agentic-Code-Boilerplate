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

from models.schemas import LLMResponse, LLMThinkingMetadata


def get_api_key() -> str:
    """
    Get API key for Gemini via enterprise token service.
    
    Uses environment variables:
        - GENAI_SERVICE_ACCOUNT
        - GENAI_SERVICE_ACCOUNT_PASSWORD
    
    Returns:
        API key/token string from enterprise token service
    """
    service_account = os.environ.get("GENAI_SERVICE_ACCOUNT")
    service_password = os.environ.get("GENAI_SERVICE_ACCOUNT_PASSWORD")
    
    if not service_account or not service_password:
        raise ValueError(
            "Enterprise authentication required. Set both:\n"
            "  - GENAI_SERVICE_ACCOUNT\n"
            "  - GENAI_SERVICE_ACCOUNT_PASSWORD"
        )
    token = os.environ.get("GEMINI_API_KEY")
    # TODO: Replace this with actual enterprise token service call
    # Example:
    # token_response = requests.post(
    #     "https://your-enterprise-token-service/api/token",
    #     json={"account": service_account, "password": service_password}
    # )
    # return token_response.json()["access_token"]
    
    print(f"🔐 Enterprise auth: {service_account}")
    print(f"🔑 Fetching token from service...")
    
    # TEMPORARY: Simulate token service returning API key for testing
    # TODO: Replace with actual token service implementation before production
    return token


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
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        top_k: int = None,
        top_p: float = None,
        candidate_count: int = None,
        thinking_budget: int = None,
        thinking_level: str = None,
        include_thoughts: bool = None
    ):
        """
        Initialize Gemini adapter.
        
        Args:
            api_key: Optional API key (uses get_api_key() if not provided)
            model: Model name (default from env or gemini-3-flash-preview)
            temperature: Sampling temperature (default 0.0)
            max_tokens: Max output tokens (default None = unlimited)
            top_k: Top-K sampling (default 1)
            top_p: Top-P sampling (default 0.1)
            candidate_count: Number of response candidates (default 1)
            thinking_budget: Thinking tokens budget (default -1 = dynamic)
                            -1 = dynamic thinking, 0 = off, >0 = specific token count
                            For Gemini 2.5 series: 0-24576
            thinking_level: Thinking level for Gemini 3 (minimal, low, medium, high)
            include_thoughts: Include thought summaries in response (default False)
        """
        self.api_key = api_key or get_api_key()
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
        self.temperature = temperature if temperature is not None else float(os.environ.get("GEMINI_TEMPERATURE", 0.0))
        self.max_tokens = max_tokens if max_tokens is not None else (int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS")) if os.environ.get("GEMINI_MAX_OUTPUT_TOKENS") else None)
        self.top_k = top_k if top_k is not None else int(os.environ.get("GEMINI_TOP_K", 1))
        self.top_p = top_p if top_p is not None else float(os.environ.get("GEMINI_TOP_P", 0.1))
        self.candidate_count = candidate_count if candidate_count is not None else int(os.environ.get("GEMINI_CANDIDATE_COUNT", 1))
        
        # Thinking configuration
        self.thinking_budget = thinking_budget if thinking_budget is not None else (int(os.environ.get("GEMINI_THINKING_BUDGET", -1)))
        self.thinking_level = thinking_level or os.environ.get("GEMINI_THINKING_LEVEL")
        self.include_thoughts = include_thoughts if include_thoughts is not None else os.environ.get("GEMINI_INCLUDE_THOUGHTS", "false").lower() == "true"
        
        # Store last thinking metadata for high-level methods
        self._last_thinking = None
        
        if not self.api_key:
            raise ValueError("Failed to obtain API key. Check authentication configuration.")
        
        # Log model configuration
        print(f"\n{'='*80}")
        print(f"🤖 GEMINI MODEL LOADED")
        print(f"{'='*80}")
        print(f"Model: {self.model}")
        model_source = "config" if model else ("env:GEMINI_MODEL" if os.environ.get("GEMINI_MODEL") else "default")
        print(f"Source: {model_source}")
        print(f"\nHyperparameters:")
        print(f"   Temperature: {self.temperature}")
        print(f"   Max Tokens: {self.max_tokens or 'unlimited'}")
        print(f"   Top-K: {self.top_k}")
        print(f"   Top-P: {self.top_p}")
        print(f"   Candidate Count: {self.candidate_count}")
        print(f"\nThinking Config:")
        print(f"   Budget: {self.thinking_budget} (-1=dynamic, 0=off, >0=tokens)")
        print(f"   Level: {self.thinking_level or 'not set'}")
        print(f"   Include Thoughts: {self.include_thoughts}")
        print(f"{'='*80}\n")
    
    def _extract_thinking_metadata(self, result: Dict[str, Any]) -> Optional[LLMThinkingMetadata]:
        """Extract thinking metadata from Gemini API response."""
        try:
            usage_metadata = result.get("usageMetadata", {})
            candidates = result.get("candidates", [])
            
            if not candidates:
                return None
            
            # Extract thought summaries if available
            thoughts = []
            candidate = candidates[0]
            content = candidate.get("content", {})
            parts = content.get("parts", [])
            
            for part in parts:
                if "thought" in part:
                    thoughts.append(part["thought"])
            
            # Extract token usage
            thoughts_token_count = usage_metadata.get("thoughtsTokenCount")
            total_token_count = usage_metadata.get("totalTokenCount")
            prompt_token_count = usage_metadata.get("promptTokenCount")
            candidates_token_count = usage_metadata.get("candidatesTokenCount")
            
            # Only create metadata object if we have thinking data
            if thoughts or thoughts_token_count:
                return LLMThinkingMetadata(
                    thoughts=thoughts if thoughts else None,
                    thoughts_token_count=thoughts_token_count,
                    thinking_budget_used=thoughts_token_count,  # Same as token count
                    total_token_count=total_token_count,
                    prompt_token_count=prompt_token_count,
                    candidates_token_count=candidates_token_count
                )
            
            return None
        except Exception as e:
            print(f"Warning: Could not extract thinking metadata: {e}")
            return None
    
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
        
        generation_config = {
            "temperature": self.temperature,
            "topK": self.top_k,
            "topP": self.top_p,
            "candidateCount": self.candidate_count
        }
        
        # Add maxOutputTokens only if specified
        if self.max_tokens:
            generation_config["maxOutputTokens"] = self.max_tokens
        
        # Add thinking configuration
        thinking_config = {}
        if self.thinking_budget is not None:
            thinking_config["thinkingBudget"] = self.thinking_budget
        if self.thinking_level:
            thinking_config["thinkingLevel"] = self.thinking_level
        if self.include_thoughts:
            thinking_config["includeThoughts"] = True
        
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config
        
        payload = {
            "contents": contents,
            "generationConfig": generation_config
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
                
                # Extract text and thinking metadata
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                thinking = self._extract_thinking_metadata(result)
                
                return LLMResponse(
                    text=text,
                    thinking=thinking,
                    raw_response=result if thinking and thinking.thoughts else None
                )
                
                # Extract thinking metadata
                thinking = self._extract_thinking_metadata(result)
                
                return LLMResponse(
                    text=text,
                    thinking=thinking,
                    raw_response=result if thinking and thinking.thoughts else None
                )
            
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
        
        generation_config = {
            "temperature": self.temperature,
            "topK": self.top_k,
            "topP": self.top_p,
            "candidateCount": self.candidate_count
        }
        
        # Add maxOutputTokens only if specified
        if self.max_tokens:
            generation_config["maxOutputTokens"] = self.max_tokens
        
        # Add thinking configuration
        thinking_config = {}
        if self.thinking_budget is not None:
            thinking_config["thinkingBudget"] = self.thinking_budget
        if self.thinking_level:
            thinking_config["thinkingLevel"] = self.thinking_level
        if self.include_thoughts:
            thinking_config["includeThoughts"] = True
        
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config
        
        payload = {
            "contents": contents,
            "generationConfig": generation_config
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
                
                # Extract text from response
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                
                # Extract thinking metadata
                thinking = self._extract_thinking_metadata(result)
                
                return LLMResponse(
                    text=text,
                    thinking=thinking,
                    raw_response=result if thinking and thinking.thoughts else None
                )
            
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

        # Call the appropriate generate function
        if files:
            llm_response = await self.generate_with_vision(structured_prompt, files, system_prompt)
        else:
            llm_response = await self.generate(structured_prompt, system_prompt)
        
        # Store thinking metadata for high-level methods
        self._last_thinking = llm_response.thinking
        
        # Extract just the text for JSON parsing
        response = llm_response.text
        
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
        
        # Debug: Log which prompts are being used
        if system_prompt:
            print(f"\n🤖 Gemini API [EXTRACTION] - Using CUSTOM system prompt from business_config.yaml:")
            print(f"   Length: {len(system_prompt)} chars")
            print(f"   First line: {system_prompt.split(chr(10))[0][:80]}...")
        else:
            print(f"\n⚠️  Gemini API [EXTRACTION] - Using DEFAULT system prompt")
        
        if user_prompt:
            print(f"   Using CUSTOM user prompt from business_config.yaml:")
            print(f"   Length: {len(user_prompt)} chars\n")
        else:
            print(f"   Using DEFAULT user prompt\n")

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
        
        result = await self.generate_structured(
            _user_prompt,
            schema,
            system_prompt=_system_prompt,
            files=[file_path]
        )
        
        # Include thinking metadata in response
        if self._last_thinking:
            result["_thinking"] = {
                "thoughts": self._last_thinking.thoughts,
                "thoughts_token_count": self._last_thinking.thoughts_token_count,
                "thinking_budget_used": self._last_thinking.thinking_budget_used,
                "total_token_count": self._last_thinking.total_token_count
            }
        
        return result
    
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
        
        # Debug: Log which prompts are being used
        if system_prompt:
            print(f"\n🤖 Gemini API [DETECTION] - Using CUSTOM system prompt from business_config.yaml:")
            print(f"   Length: {len(system_prompt)} chars")
            print(f"   First line: {system_prompt.split(chr(10))[0][:80]}...")
        else:
            print(f"\n⚠️  Gemini API [DETECTION] - Using DEFAULT system prompt")
        
        if user_prompt:
            print(f"   Using CUSTOM user prompt from business_config.yaml:")
            print(f"   Length: {len(user_prompt)} chars\n")
        else:
            print(f"   Using DEFAULT user prompt\n")

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
        
        result = await self.generate_structured(
            _user_prompt,
            schema,
            system_prompt=_system_prompt,
            files=[file_path]
        )
        
        # Include thinking metadata in response
        if self._last_thinking:
            result["_thinking"] = {
                "thoughts": self._last_thinking.thoughts,
                "thoughts_token_count": self._last_thinking.thoughts_token_count,
                "thinking_budget_used": self._last_thinking.thinking_budget_used,
                "total_token_count": self._last_thinking.total_token_count
            }
        
        return result
    
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
        
        # Debug: Log which prompts are being used
        if system_prompt:
            print(f"\n🤖 Gemini API - Using CUSTOM system prompt from business_config.yaml:")
            print(f"   Length: {len(system_prompt)} chars")
            print(f"   First line: {system_prompt.split(chr(10))[0][:80]}...")
        else:
            print(f"\n⚠️  Gemini API - Using DEFAULT system prompt (business_config not loaded)")
        
        if user_prompt:
            print(f"   Using CUSTOM user prompt from business_config.yaml:")
            print(f"   Length: {len(user_prompt)} chars")
            print(f"   Includes M1-M7?: {'M1' in user_prompt and 'M7' in user_prompt}")
            print()
        else:
            print(f"   Using DEFAULT user prompt (business_config not loaded)\n")

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
        
        result = await self.generate_structured(
            _user_prompt,
            schema,
            system_prompt=_system_prompt,
            files=files
        )
        
        # Include thinking metadata in response
        if self._last_thinking:
            result["_thinking"] = {
                "thoughts": self._last_thinking.thoughts,
                "thoughts_token_count": self._last_thinking.thoughts_token_count,
                "thinking_budget_used": self._last_thinking.thinking_budget_used,
                "total_token_count": self._last_thinking.total_token_count
            }
        
        return result


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
        print(f"\n📥 Loading Gemini adapter from app_config.yaml")
        print(f"   Provider: {config.llm.provider}")
        print(f"   Model: {gemini_cfg.model}\n")
        
        return GeminiRestAdapter(
            api_key=None,  # Always use get_api_key() for enterprise auth
            model=gemini_cfg.model,
            temperature=getattr(gemini_cfg, 'temperature', None),
            max_tokens=getattr(gemini_cfg, 'max_tokens', None),
            top_k=getattr(gemini_cfg, 'top_k', None),
            top_p=getattr(gemini_cfg, 'top_p', None),
            candidate_count=getattr(gemini_cfg, 'candidate_count', None),
            thinking_budget=getattr(gemini_cfg, 'thinking_budget', None),
            thinking_level=getattr(gemini_cfg, 'thinking_level', None),
            include_thoughts=getattr(gemini_cfg, 'include_thoughts', None)
        )
    
    # Default from environment (uses get_api_key() internally)
    print(f"\n📥 Loading Gemini adapter from environment variables")
    return GeminiRestAdapter()
