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
            Extracted payment fields with confidence scores and raw OCR text
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
            
            # Override with custom_prompt if provided (backward compatibility)
            if custom_prompt:
                user_prompt = custom_prompt

            result = await gemini.extract_payment_fields(
                document_path,
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )

            raw_text = result.get("raw_text", "")
            thinking_metadata = result.get("_thinking", {})

            extracted_payment = {
                "creditor_name": result.get("creditor_name"),
                "creditor_account": result.get("creditor_account"),
                "creditor_bank": result.get("creditor_bank"),
                "debtor_name": result.get("debtor_name"),
                "debtor_account": result.get("debtor_account"),
                "debtor_bank": result.get("debtor_bank"),
                "amount": result.get("amount"),
                "currency": result.get("currency"),
                "payment_type": result.get("payment_type"),
                "payment_date": result.get("payment_date"),
                "charges_account": result.get("charges_account"),
                "reference": result.get("reference"),
                "raw_ocr_text": raw_text
            }

            return {
                "success": True,
                "extracted_payment": extracted_payment,
                "model_used": gemini.model,
                "raw_response_length": len(str(result)),
                "thinking": thinking_metadata
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
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
            required = ['creditor_name', 'debtor_name', 'amount', 'currency']
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
