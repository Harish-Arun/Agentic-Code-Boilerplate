"""
Pluggable LLM Adapters - Abstract base class with multiple implementations.
Swap LLM providers by changing config without code changes.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import asyncio
import base64


class LLMAdapter(ABC):
    """Abstract base class for LLM operations."""
    
    @abstractmethod
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate text response from a prompt."""
        pass
    
    @abstractmethod
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Union[str, bytes]],  # file paths or base64
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response with image inputs (for vision models)."""
        pass
    
    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output matching the provided schema."""
        pass


class MockLLMAdapter(LLMAdapter):
    """Mock implementation for testing and development."""
    
    def __init__(self, delay_ms: int = 100):
        self.delay_ms = delay_ms
    
    async def _simulate_delay(self):
        await asyncio.sleep(self.delay_ms / 1000)
    
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        await self._simulate_delay()
        return f"[MOCK LLM Response] Processed prompt with {len(prompt)} characters."
    
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        await self._simulate_delay()
        return f"[MOCK Vision Response] Analyzed {len(images)} image(s). Prompt: {prompt[:50]}..."
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        await self._simulate_delay()
        
        # Return mock structured data based on common extraction fields
        return {
            "creditor": {"name": "ACME Corp", "account": "1234567890"},
            "debtor": {"name": "John Doe", "account": "0987654321"},
            "amount": {"value": 10000.00, "currency": "USD"},
            "payment_type": "wire_transfer",
            "date": "2026-01-29",
            "confidence": 0.95,
            "raw_text_preview": "[Mock extracted text...]"
        }


class GeminiAdapter(LLMAdapter):
    """Google Gemini implementation - for production vision tasks."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", 
                 temperature: float = 0.1, max_tokens: int = 4096):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = await asyncio.to_thread(
            model.generate_content,
            full_prompt,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        return response.text
    
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        import google.generativeai as genai
        from PIL import Image
        import io
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        
        # Prepare images
        image_parts = []
        for img in images:
            if isinstance(img, str):
                # File path
                image_parts.append(Image.open(img))
            else:
                # Bytes
                image_parts.append(Image.open(io.BytesIO(img)))
        
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = await asyncio.to_thread(
            model.generate_content,
            [full_prompt] + image_parts,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
        )
        return response.text
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        import json
        
        structured_prompt = f"""
{system_prompt or ''}

{prompt}

Return your response as valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Respond ONLY with valid JSON, no additional text.
"""
        response = await self.generate(structured_prompt)
        
        # Parse JSON from response
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Failed to parse JSON from response: {response[:200]}")


class OpenAIAdapter(LLMAdapter):
    """OpenAI implementation - for GPT-4 based workflows."""
    
    def __init__(self, api_key: str, model: str = "gpt-4", temperature: float = 0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
    
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        # Prepare image content
        content = [{"type": "text", "text": prompt}]
        for img in images:
            if isinstance(img, str):
                with open(img, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode()
            else:
                img_data = base64.b64encode(img).decode()
            
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_data}"}
            })
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})
        
        response = await client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        from openai import AsyncOpenAI
        import json
        
        client = AsyncOpenAI(api_key=self.api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)


class AzureOpenAIAdapter(LLMAdapter):
    """Azure OpenAI implementation - for enterprise deployments."""
    
    def __init__(self, endpoint: str, api_key: str, deployment: str, 
                 api_version: str = "2024-02-15-preview"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment = deployment
        self.api_version = api_version
    
    async def generate(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        from openai import AsyncAzureOpenAI
        
        client = AsyncAzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = await client.chat.completions.create(
            model=self.deployment,
            messages=messages
        )
        return response.choices[0].message.content
    
    async def generate_with_vision(
        self,
        prompt: str,
        images: List[Union[str, bytes]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        # Similar to OpenAI implementation
        return await self.generate(f"{prompt}\n[Vision not implemented for Azure]")
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        import json
        response = await self.generate(
            f"{prompt}\n\nRespond with valid JSON matching: {json.dumps(schema)}",
            system_prompt
        )
        return json.loads(response)


def get_llm_adapter(config) -> LLMAdapter:
    """
    Factory function to get the appropriate LLM adapter based on config.
    
    Usage:
        from shared.config import get_config
        from shared.adapters.llm import get_llm_adapter
        
        config = get_config()
        llm = get_llm_adapter(config)
        response = await llm.generate("Hello!")
    """
    llm_config = config.llm
    
    if llm_config.provider == "mock":
        return MockLLMAdapter(llm_config.mock.delay_ms)
    elif llm_config.provider == "gemini":
        g = llm_config.gemini
        return GeminiAdapter(g.api_key, g.model, g.temperature, g.max_tokens)
    elif llm_config.provider == "openai":
        o = llm_config.openai
        return OpenAIAdapter(o.api_key, o.model, o.temperature)
    elif llm_config.provider == "azure":
        a = llm_config.azure
        return AzureOpenAIAdapter(a.endpoint, a.api_key, a.deployment, a.api_version)
    else:
        raise ValueError(f"Unknown LLM provider: {llm_config.provider}")
