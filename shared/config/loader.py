"""
Configuration Loader - Pydantic-validated config with environment variable support.

Loads TWO config files:
  1. config/app_config.yaml      — Infrastructure / tech (Engineering owned)
  2. config/business_config.yaml — Business rules / thresholds (Business sign-off)
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


class DynamoDBConfig(BaseModel):
    region: str = "eu-west-2"
    table_prefix: str = "nnp_"
    endpoint_url: str = ""


class DatabaseConfig(BaseModel):
    type: str = "sqlite"  # sqlite, postgres, mongo, dynamodb
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    postgres: PostgresConfig = Field(default_factory=PostgresConfig)
    mongo: MongoConfig = Field(default_factory=MongoConfig)
    dynamodb: DynamoDBConfig = Field(default_factory=DynamoDBConfig)


# ============================================
# LLM Config Models
# ============================================
class GeminiConfig(BaseModel):
    model: str = "gemini-3-flash-preview"
    temperature: float = 0.0
    max_tokens: Optional[int] = None  # None = unlimited
    top_k: int = 1
    top_p: float = 0.1
    candidate_count: int = 1
    # Thinking configuration
    thinking_budget: Optional[int] = -1  # -1=dynamic, 0=off, >0=specific tokens
    thinking_level: Optional[str] = None  # minimal, low, medium, high (Gemini 3)
    include_thoughts: bool = False  # Include thought summaries in response


class OpenAIConfig(BaseModel):
    model: str = "gpt-4"
    temperature: float = 0.1


class AzureConfig(BaseModel):
    endpoint: str = ""
    deployment: str = ""
    api_version: str = "2024-02-15-preview"


class LLMConfig(BaseModel):
    provider: str = "gemini"  # gemini, openai, azure
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
    enabled: List[str] = Field(default_factory=lambda: ["extraction", "signature_detection", "signature_verification", "pdf_utils", "signature_provider"])
    pdf_utils: Dict[str, Any] = Field(default_factory=lambda: {"max_pages": 50, "dpi": 300})
    signature_provider: Dict[str, Any] = Field(default_factory=lambda: {"cache_enabled": True, "cache_ttl_seconds": 3600})


class MCPConfig(BaseModel):
    server: MCPServerConfig = Field(default_factory=MCPServerConfig)
    tools: MCPToolsConfig = Field(default_factory=MCPToolsConfig)


# ============================================
# Service Ports Config
# ============================================
class ServiceEndpoint(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000


class ServicesConfig(BaseModel):
    api_gateway: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(port=8000))
    agents: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(port=8001))
    mcp_tools: ServiceEndpoint = Field(default_factory=lambda: ServiceEndpoint(port=8002))


# ============================================
# Storage Config
# ============================================
class StorageConfig(BaseModel):
    data_dir: str = "./data"
    uploads_dir: str = "./data/uploads"
    signatures_dir: str = "./data/signatures"
    reference_dir: str = "./data/reference"
    debug_dir: str = "./data/debug"


# ============================================
# Ingestion Sources Config
# ============================================
class ManualUploadConfig(BaseModel):
    enabled: bool = True


class NetworkDriveConfig(BaseModel):
    enabled: bool = False
    protocol: str = "smb"
    path: str = ""
    poll_interval_seconds: int = 60


class S3BucketConfig(BaseModel):
    enabled: bool = False
    bucket: str = ""
    prefix: str = "incoming/"
    region: str = "eu-west-2"
    poll_interval_seconds: int = 30


class IngestionSourcesConfig(BaseModel):
    manual_upload: ManualUploadConfig = Field(default_factory=ManualUploadConfig)
    network_drive: NetworkDriveConfig = Field(default_factory=NetworkDriveConfig)
    s3_bucket: S3BucketConfig = Field(default_factory=S3BucketConfig)


class IngestionConfig(BaseModel):
    sources: IngestionSourcesConfig = Field(default_factory=IngestionSourcesConfig)


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
# Prompts Config (Business-Owned)
# ============================================
class PromptTemplate(BaseModel):
    """Individual prompt template with system and user parts."""
    system: str = ""
    user: str = ""


class PromptsConfig(BaseModel):
    """AI prompts for extraction, detection, and verification."""
    extraction: PromptTemplate = Field(default_factory=lambda: PromptTemplate(
        system="You are a payment document processing assistant.",
        user="Extract payment fields from this document..."
    ))
    signature_detection: PromptTemplate = Field(default_factory=lambda: PromptTemplate(
        system="You are a forensic document examiner.",
        user="Identify signature regions..."
    ))
    signature_verification: PromptTemplate = Field(default_factory=lambda: PromptTemplate(
        system="You are a Forensic Document Examiner (FDE).",
        user="Compare signatures..."
    ))


# ============================================
# Metric Thresholds Config (Business-Owned)
# ============================================
class MetricThresholdsConfig(BaseModel):
    m1_tolerance: float = 0.10
    m1_veto: float = 0.50
    m1_weight: int = 15
    m2_min_quality: int = 60
    m2_weight: int = 10
    m3_tolerance: float = 5.0
    m3_veto: float = 45.0
    m3_weight: int = 10
    m4_tolerance: float = 0.10
    m4_weight: int = 10
    m5_weight: int = 20
    m6_tolerance: float = 0.15
    m6_weight: int = 10
    m7_tolerance: float = 30.0
    m7_weight: int = 15


# ============================================
# Decision Thresholds Config (Business-Owned)
# ============================================
class DecisionThresholdsConfig(BaseModel):
    approve_threshold: int = 80
    flag_min_threshold: int = 60


# ============================================
# Extraction Rules Config (Business-Owned)
# ============================================
class ExtractionRulesConfig(BaseModel):
    required_fields: List[str] = Field(default_factory=lambda: [
        "creditor_name", "creditor_account", "debtor_name",
        "debtor_account", "amount", "currency"
    ])
    minimum_confidence: float = 0.70
    account_format: str = "IBAN"
    iban_min_length: int = 15
    iban_max_length: int = 34
    max_amount_no_review: float = 50000.0
    min_amount: float = 0.01


# ============================================
# Authentication Config (Business-Owned)
# ============================================
class SignatureVerificationRules(BaseModel):
    enabled: bool = True
    require_reference: bool = True
    max_signatures_per_doc: int = 5


class AuthenticationConfig(BaseModel):
    signature_verification: SignatureVerificationRules = Field(default_factory=SignatureVerificationRules)


# ============================================
# HITL Config (Business-Owned)
# ============================================
class HITLConfig(BaseModel):
    high_confidence_threshold: int = 95   # ≥ 95 → quick approve guidance
    critical_threshold: int = 30          # ≤ 30 → thorough investigation
    max_review_time_hours: int = 24


# ============================================
# Business Config (loaded from business_config.yaml)
# ============================================
class BusinessConfig(BaseModel):
    """All business rules, requiring stakeholder sign-off."""
    metric_thresholds: MetricThresholdsConfig = Field(default_factory=MetricThresholdsConfig)
    decision_thresholds: DecisionThresholdsConfig = Field(default_factory=DecisionThresholdsConfig)
    document_states: List[str] = Field(default_factory=lambda: [
        "INGESTED", "PROCESSING", "EXTRACTED", "AUTHENTICATED",
        "REVIEW_PENDING", "APPROVED", "REJECTED", "DISPATCHED"
    ])
    extraction_rules: ExtractionRulesConfig = Field(default_factory=ExtractionRulesConfig)
    authentication: AuthenticationConfig = Field(default_factory=AuthenticationConfig)
    hitl: HITLConfig = Field(default_factory=HITLConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)


# ============================================
# Main App Config (merges both files)
# ============================================
class AppConfig(BaseModel):
    # --- Infrastructure (app_config.yaml) ---
    services: ServicesConfig = Field(default_factory=ServicesConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    agents: AgentConfig = Field(default_factory=AgentConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    # --- Business Rules (business_config.yaml) ---
    business: BusinessConfig = Field(default_factory=BusinessConfig)
    # --- Legacy accessors (backward-compat) ---
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    document_states: List[str] = Field(
        default_factory=lambda: ["INGESTED", "PROCESSING", "EXTRACTED", "AUTHENTICATED",
                                 "REVIEW_PENDING", "APPROVED", "REJECTED", "DISPATCHED"]
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
    Load configuration from YAML files with environment variable substitution.
    
    Loads two files:
      1. app_config.yaml      — Infrastructure / tech settings
      2. business_config.yaml — Business rules / thresholds
    
    Args:
        config_path: Path to app config file. Defaults to ./config/app_config.yaml
    
    Returns:
        Validated AppConfig instance (merged)
    """
    if config_path is None:
        config_path = os.environ.get("CONFIG_PATH", "./config/app_config.yaml")
    
    config_file = Path(config_path)
    config_dir = config_file.parent
    business_config_path = os.environ.get(
        "BUSINESS_CONFIG_PATH",
        str(config_dir / "business_config.yaml")
    )
    
    config_data = {}
    
    # Load app config (infrastructure)
    if config_file.exists():
        print(f"\n{'='*80}")
        print(f"📋 LOADING APP CONFIG: {config_file}")
        print(f"{'='*80}")
        with open(config_file, "r") as f:
            raw_app = yaml.safe_load(f) or {}
        config_data.update(_substitute_env_vars(raw_app))
        
        # Log LLM configuration
        if "llm" in config_data:
            llm_cfg = config_data["llm"]
            provider = llm_cfg.get("provider", "unknown")
            print(f"\n✅ LLM Configuration:")
            print(f"   Provider: {provider}")
            
            if provider == "gemini" and "gemini" in llm_cfg:
                gemini = llm_cfg["gemini"]
                print(f"   Model: {gemini.get('model', 'not set')}")
                print(f"   Temperature: {gemini.get('temperature', 'not set')}")
                print(f"   Top-K: {gemini.get('top_k', 'not set')}")
                print(f"   Top-P: {gemini.get('top_p', 'not set')}")
                print(f"   Thinking Budget: {gemini.get('thinking_budget', 'not set')}")
        print(f"{'='*80}\n")
    
    # Load business config (rules & thresholds)
    business_file = Path(business_config_path)
    if business_file.exists():
        print(f"\n{'='*80}")
        print(f"📋 LOADING BUSINESS CONFIG: {business_file}")
        print(f"{'='*80}")
        with open(business_file, "r") as f:
            raw_biz = yaml.safe_load(f) or {}
        biz_data = _substitute_env_vars(raw_biz)
        
        # Log key business config values
        if "metric_thresholds" in biz_data:
            thresholds = biz_data["metric_thresholds"]
            print(f"\n✅ Business Config Loaded Successfully:")
            print(f"   M1 tolerance: {thresholds.get('m1_tolerance')} (veto: {thresholds.get('m1_veto')})")
            print(f"   M3 tolerance: {thresholds.get('m3_tolerance')}° (veto: {thresholds.get('m3_veto')}°)")
            print(f"   M1 weight: {thresholds.get('m1_weight')}")
            print(f"   M3 weight: {thresholds.get('m3_weight')}")
        
        if "decision_thresholds" in biz_data:
            decisions = biz_data["decision_thresholds"]
            print(f"\n   Approve threshold: {decisions.get('approve_threshold')}")
            print(f"   Flag threshold: {decisions.get('flag_min_threshold')}")
        
        if "prompts" in biz_data:
            prompts = biz_data["prompts"]
            print(f"\n✅ Prompts Loaded:")
            
            # Extraction prompts
            if "extraction" in prompts:
                ext_prompt = prompts["extraction"]
                sys_len = len(ext_prompt.get("system", "")) if isinstance(ext_prompt, dict) else 0
                user_len = len(ext_prompt.get("user", "")) if isinstance(ext_prompt, dict) else 0
                print(f"   extraction: system={sys_len} chars, user={user_len} chars")
            
            # Signature detection prompts
            if "signature_detection" in prompts:
                det_prompt = prompts["signature_detection"]
                sys_len = len(det_prompt.get("system", "")) if isinstance(det_prompt, dict) else 0
                user_len = len(det_prompt.get("user", "")) if isinstance(det_prompt, dict) else 0
                print(f"   signature_detection: system={sys_len} chars, user={user_len} chars")
            
            # Signature verification prompts
            if "signature_verification" in prompts:
                sig_prompt = prompts["signature_verification"]
                sys_len = len(sig_prompt.get("system", "")) if isinstance(sig_prompt, dict) else 0
                user_len = len(sig_prompt.get("user", "")) if isinstance(sig_prompt, dict) else 0
                print(f"   signature_verification: system={sys_len} chars, user={user_len} chars")
                if user_len > 0:
                    user_text = sig_prompt.get("user", "")
                    has_m1_m7 = "M1" in user_text and "M7" in user_text
                    print(f"   → M1-M7 framework: {'✓ YES' if has_m1_m7 else '✗ NO'}")
        
        print(f"{'='*80}\n")
        
        config_data["business"] = biz_data
        # Backward-compat: also set top-level prompts & document_states
        if "prompts" in biz_data:
            config_data["prompts"] = biz_data["prompts"]
        if "document_states" in biz_data:
            config_data["document_states"] = biz_data["document_states"]
    else:
        print(f"\n⚠️  WARNING: Business config file not found: {business_file}")
        print(f"   Using default thresholds instead.\n")
    
    if config_data:
        return AppConfig(**config_data)
    
    # Return defaults if no config files exist
    return AppConfig()


def get_config() -> AppConfig:
    """Get the cached configuration instance."""
    return load_config()
