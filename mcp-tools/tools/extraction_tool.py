"""
Extraction Tool — Payment field extraction from documents.

Business logic (called by agent orchestrator via MCP):
  1. Calls Gemini Vision to extract structured payment fields
  2. Validates extracted fields against business rules
  3. Returns extraction result with confidence scores
"""
from typing import Dict, Any, Optional
from fastmcp import FastMCP
import os
import json

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


def _validate_normalized_field_bboxes(extracted_payment: Dict[str, Any]) -> Dict[str, Any]:
    validated = dict(extracted_payment)

    for field_name, field_data in validated.items():
        if not isinstance(field_data, dict):
            continue

        bbox = field_data.get("bounding_box")
        if not isinstance(bbox, dict):
            continue

        page = int(bbox.get("page", 1) or 1)
        if page <= 0:
            page = 1

        x1 = _safe_float(bbox.get("x1", 0.0), 0.0)
        y1 = _safe_float(bbox.get("y1", 0.0), 0.0)
        x2 = _safe_float(bbox.get("x2", 0.0), 0.0)
        y2 = _safe_float(bbox.get("y2", 0.0), 0.0)

        is_normalized = (0.0 <= x1 <= 1.0 and 0.0 <= y1 <= 1.0 and 0.0 <= x2 <= 1.0 and 0.0 <= y2 <= 1.0)
        has_valid_order = (x1 < x2 and y1 < y2)

        if not is_normalized or not has_valid_order:
            print(
                f"⚠️ [EXTRACTION] Invalid/non-normalized bbox for {field_name} on page {page}: "
                f"({x1}, {y1}, {x2}, {y2}) — dropping bbox"
            )
            field_data["bounding_box"] = None
            continue

        field_data["bounding_box"] = {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "page": page
        }

    return validated


def register_extraction_tools(mcp: FastMCP, config: AppConfig):
    """Register extraction business-logic tools with the MCP server."""

    @mcp.tool()
    async def extract_payment_fields(
        document_path: str,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract payment fields from a document using Gemini Vision.

        Sends the document to Gemini Vision API, extracts structured payment
        fields (creditor, debtor, amount, etc.) and returns confidence scores.

        Args:
            document_path: Path to the PDF or image file
            custom_prompt: Optional custom extraction prompt (uses business config default)

        Returns:
            Extracted payment fields with confidence scores and appendix metadata
        """
        if not os.path.exists(document_path):
            return {
                "success": False,
                "error": f"Document not found: {document_path}",
                "extracted_payment": None,
                "model_used": ""
            }

        try:
            gemini = get_gemini_adapter(config)

            # Load prompts from business config
            system_prompt = None
            user_prompt = None
            
            if hasattr(config, 'business') and hasattr(config.business, 'prompts'):
                prompts_cfg = config.business.prompts
                if hasattr(prompts_cfg, 'extraction') and hasattr(prompts_cfg.extraction, 'system'):
                    system_prompt = prompts_cfg.extraction.system
                    user_prompt = prompts_cfg.extraction.user
                    
                    # Debug: Show that business_config prompts are being used
                    print(f"\n{'='*80}")
                    print(f"📝 [EXTRACTION] Using Prompts from business_config.yaml")
                    print(f"{'='*80}")
                    print(f"System Prompt (first 150 chars):")
                    print(f"   {system_prompt[:150]}...")
                    print(f"\nUser Prompt (first 200 chars):")
                    print(f"   {user_prompt[:200]}...")
                    print(f"{'='*80}\n")
                else:
                    print(f"\n⚠️  WARNING: extraction prompts not found in business_config.yaml")
                    print(f"   Using default prompts instead.\n")
            else:
                print(f"\n⚠️  WARNING: business.prompts not found in config")
                print(f"   Using default prompts instead.\n")
            
            # Override with custom_prompt if provided (backward compatibility)
            if custom_prompt:
                print(f"\n⚠️  Custom prompt override detected - using custom_prompt parameter instead\n")
                user_prompt = custom_prompt

            result = await gemini.extract_payment_fields(
                document_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            thinking_metadata = result.get("_thinking", {})

            # Handle new schema with additional_fields (catch-all like **kwargs)
            known_result_keys = {
                "creditor_name", "creditor_account", "creditor_sort_code", "creditor_bank",
                "debtor_name", "debtor_account", "debtor_sort_code", "debtor_bank",
                "amount", "currency", "payment_type", "payment_date",
                "additional_fields", "_thinking"
            }

            # Extract additional fields (the catch-all dict)
            additional_fields = result.get("additional_fields")
            if not isinstance(additional_fields, dict):
                additional_fields = {}

            # Legacy fallback: if old 'appendix' structure exists, merge it
            if "appendix" in result and isinstance(result["appendix"], dict):
                appendix = result["appendix"]
                if "all_fields_dump" in appendix:
                    additional_fields.update(appendix["all_fields_dump"])
                if "key_values" in appendix:
                    additional_fields.update(appendix["key_values"])

            # Capture any other unexpected top-level fields
            for key, value in result.items():
                if key in known_result_keys:
                    continue
                if key in {"raw_text", "raw_ocr_text", "appendix"}:
                    continue
                if value is None:
                    continue
                if isinstance(value, dict) and {"value", "confidence"}.issubset(set(value.keys())):
                    # This is a field object, add to additional_fields if not already there
                    if key not in additional_fields:
                        additional_fields[key] = value
                    continue

            extracted_payment = {
                "creditor_name": result.get("creditor_name"),
                "creditor_account": result.get("creditor_account"),
                "creditor_sort_code": result.get("creditor_sort_code"),
                "creditor_bank": result.get("creditor_bank"),
                "debtor_name": result.get("debtor_name"),
                "debtor_account": result.get("debtor_account"),
                "debtor_sort_code": result.get("debtor_sort_code"),
                "debtor_bank": result.get("debtor_bank"),
                "amount": result.get("amount"),
                "currency": result.get("currency"),
                "payment_type": result.get("payment_type"),
                "payment_date": result.get("payment_date"),
                "additional_fields": additional_fields
            }

            extracted_payment = _validate_normalized_field_bboxes(extracted_payment)

            return {
                "success": True,
                "extracted_payment": extracted_payment,
                "model_used": gemini.model,
                "raw_response_length": len(str(result)),
                "thinking": thinking_metadata
            }

        except Exception as e:
            import traceback
            error_msg = str(e) or f"{type(e).__name__} (no message)"
            print(f"\n❌ [EXTRACTION TOOL] Exception: {type(e).__name__}: {error_msg}")
            print(traceback.format_exc())
            return {
                "success": False,
                "error": f"{type(e).__name__}: {error_msg}",
                "extracted_payment": None,
                "model_used": config.llm.gemini.model
            }

    @mcp.tool()
    async def validate_extraction(
        extracted_fields: str
    ) -> Dict[str, Any]:
        """
        Challenger validation — check extraction quality against business rules.

        Validates:
        - Required fields are present
        - Confidence scores meet minimum threshold
        - Account formats are valid (IBAN check)
        - Amount is positive and non-zero

        Args:
            extracted_fields: JSON string of extracted payment fields

        Returns:
            Validation result with list of issues found
        """
        try:
            fields = json.loads(extracted_fields) if isinstance(extracted_fields, str) else extracted_fields
        except (json.JSONDecodeError, TypeError):
            return {
                "valid": False,
                "issues": ["Invalid input: could not parse extracted_fields"],
                "notes": "Validation failed due to invalid input"
            }

        issues = []

        # Get business rules from config
        try:
            biz_rules = config.business.extraction_rules
            required = biz_rules.required_fields
            min_confidence = biz_rules.minimum_confidence
            iban_min = biz_rules.iban_min_length
            iban_max = biz_rules.iban_max_length
        except Exception:
            required = ['creditor_name', 'creditor_sort_code', 'debtor_name', 'debtor_sort_code', 'amount', 'currency']
            min_confidence = 0.70
            iban_min = 15
            iban_max = 34

        # Check required fields
        for field_name in required:
            field_data = fields.get(field_name)
            if field_data is None or (isinstance(field_data, dict) and field_data.get("value") is None):
                issues.append(f"Missing required field: {field_name}")
            elif isinstance(field_data, dict) and field_data.get("confidence", 0) < min_confidence:
                issues.append(f"Low confidence for {field_name}: {field_data.get('confidence', 0):.2f}")

        # Validate account formats (basic IBAN check)
        for account_field in ['creditor_account', 'debtor_account']:
            field_data = fields.get(account_field)
            if field_data and isinstance(field_data, dict) and field_data.get("value"):
                value = str(field_data["value"]).replace(" ", "").upper()
                if len(value) < iban_min or len(value) > iban_max:
                    issues.append(f"Invalid account format for {account_field}: length {len(value)}")

        # Validate amount
        amount_data = fields.get("amount")
        if amount_data and isinstance(amount_data, dict) and amount_data.get("value") is not None:
            try:
                amount_val = float(str(amount_data["value"]).replace(",", ""))
                if amount_val <= 0:
                    issues.append("Amount must be positive")
            except (ValueError, TypeError):
                issues.append("Invalid amount format")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "notes": f"Validation completed with {len(issues)} issues" if issues else "All fields valid"
        }
