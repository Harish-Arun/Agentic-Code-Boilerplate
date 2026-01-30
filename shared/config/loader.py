"""
Configuration Loader - Pydantic-validated config with environment variable support.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from functools import lru_cache

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


# ============================================
# Database Config Models
# ============================================
class SQLiteConfig(BaseModel):
    path: str = "./data/nnp_ai.db"


class PostgresConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    database: str = "nnp_ai"
    user: str = "admin"
    password: str = ""


class MongoConfig(BaseModel):
    uri: str = "mongodb://localhost:27017/nnp_ai"


class DatabaseConfig(BaseModel):
    type: str = "sqlite"  # sqlite, postgres, mongo
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    mongo: MongoConfig = Field(default_factory=MongoConfig)


# ============================================
# LLM Config Models
# ============================================
class MockLLMConfig(BaseModel):
    delay_ms: int = 100


class GeminiConfig(BaseModel):
    api_key: str = ""
    model: str = "gemini-1.5-pro"
    temperature: float = 0.1
    max_tokens: int = 4096


class OpenAIConfig(BaseModel):
    api_key: str = ""
    model: str = "gpt-4"
    temperature: float = 0.1


class AzureConfig(BaseModel):
    endpoint: str = ""
    api_key: str = ""
    deployment: str = ""
    api_version: str = "2024-02-15-preview"


class LLMConfig(BaseModel):
    provider: str = "mock"  # mock, gemini, openai, azure
    mock: MockLLMConfig = Field(default_factory=MockLLMConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    azure: AzureConfig = Field(default_factory=AzureConfig)


# ============================================
# Agent Config Models
# ============================================
class RetryConfig(BaseModel):
    max_attempts: int = 3
    backoff_seconds: int = 1


class CheckpointConfig(BaseModel):
    enabled: bool = True
    backend: str = "memory"  # memory, redis, postgres


class AgentConfig(BaseModel):
    enabled: List[str] = Field(default_factory=lambda: ["extraction", "signature_detection", "verification"])
    retry: RetryConfig = Field(default_factory=RetryConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)


# ============================================
# MCP Config Models
# ============================================
class MCPServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8002
    transport: str = "sse"


class MCPToolsConfig(BaseModel):
    enabled: List[str] = Field(default_factory=lambda: ["ocr", "pdf_utils", "signature_provider"])
    ocr: Dict[str, Any] = Field(default_factory=lambda: {"provider": "mock"})
    pdf_utils: Dict[str, Any] = Field(default_factory=lambda: {"max_pages": 50, "dpi": 300})
    signature_provider: Dict[str, Any] = Field(default_factory=lambda: {"cache_enabled": True, "cache_ttl_seconds": 3600})


class MCPConfig(BaseModel):
    server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    tools: MCPToolsConfig = Field(default_factory=MCPToolsConfig)


# ============================================
# Feature Flags
# ============================================
class FeatureConfig(BaseModel):
    human_in_loop: bool = True
    signature_verification: bool = True
    auto_extraction: bool = True
    audit_logging: bool = True
    retry_on_failure: bool = True


# ============================================
# Logging Config
# ============================================
class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    include_timestamp: bool = True


# ============================================
# Prompts Config
# ============================================
class PromptsConfig(BaseModel):
    extraction: str = "Extract payment fields from this document..."
    signature_detection: str = "Identify signature regions..."
    signature_verification: str = "Compare signatures..."


# ============================================
# Main App Config
# ============================================
class AppConfig(BaseModel):
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    document_states: List[str] = Field(
        default_factory=lambda: ["INGESTED", "PROCESSING", "EXTRACTED", "VERIFIED", "REVIEWED", "CONFIRMED", "REJECTED"]
    )


def _substitute_env_vars(data: Any) -> Any:
    """Recursively substitute environment variables in config values."""
    if isinstance(data, str):
        # Handle ${VAR:-default} pattern
        import re
        pattern = r'\$\{([^}:]+)(?::-([^}]*))?\}'
        
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(var_name, default)
        
        return re.sub(pattern, replacer, data)
    elif isinstance(data, dict):
        return {k: _substitute_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_substitute_env_vars(item) for item in data]
    return data


@lru_cache()
def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    Load configuration from YAML file with environment variable substitution.
    
    Args:
        config_path: Path to config file. Defaults to ./config/app_config.yaml
    
    Returns:
        Validated AppConfig instance
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "./config/app_config.yaml")
    
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, "r") as f:
            raw_config = yaml.safe_load(f)
        
        # Substitute environment variables
        config_data = _substitute_env_vars(raw_config)
        return AppConfig(**config_data)
    
    # Return defaults if config file doesn't exist
    return AppConfig()


def get_config() -> AppConfig:
    """Get the cached configuration instance."""
    return load_config()
