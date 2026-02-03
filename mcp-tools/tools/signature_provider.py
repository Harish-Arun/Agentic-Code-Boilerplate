"""
Signature Provider Tool - Reference signature management.

Provides tools to retrieve and store reference signatures for verification.
"""
from typing import Dict, Any, Optional
from fastmcp import FastMCP
from datetime import datetime
import os
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import AppConfig


# Mock signature database
DATA_DIR = os.environ.get("DATA_DIR", "/data")

_mock_signatures = {
    "CUST001": {
        "customer_id": "CUST001",
        "customer_name": "John Smith",
        "signature_image_path": f"{Path(DATA_DIR).as_posix()}/references/cust001_signature.png",
        "created_at": "2025-01-15T10:00:00Z",
        "verified": True
    },
    "CUST002": {
        "customer_id": "CUST002",
        "customer_name": "Jane Doe",
        "signature_image_path": f"{Path(DATA_DIR).as_posix()}/references/cust002_signature.png",
        "created_at": "2025-02-20T14:30:00Z",
        "verified": True
    }
}


def register_signature_tools(mcp: FastMCP, config: AppConfig):
    """Register signature-related tools with the MCP server."""
    
    sig_config = config.mcp.tools.signature_provider
    
    @mcp.tool()
    async def get_reference_signature(
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Get reference signature for a customer.
        
        Args:
            customer_id: Customer identifier
        
        Returns:
            Reference signature details including image path
        """
        # Check mock database
        if customer_id in _mock_signatures:
            sig_data = _mock_signatures[customer_id]
            return {
                "success": True,
                "signature": sig_data,
                "message": "Reference signature found"
            }
        
        # Return not found (mock fallback with generic signature)
        return {
            "success": True,
            "signature": {
                "customer_id": customer_id,
                "customer_name": f"Customer {customer_id}",
                "signature_image_path": f"{Path(DATA_DIR).as_posix()}/references/default_signature.png",
                "created_at": datetime.utcnow().isoformat(),
                "verified": False
            },
            "message": "[Mock] No reference signature found, using default"
        }
    
    @mcp.tool()
    async def store_signature(
        customer_id: str,
        customer_name: str,
        signature_image_path: str,
        verified: bool = False
    ) -> Dict[str, Any]:
        """
        Store a new reference signature.
        
        Args:
            customer_id: Customer identifier
            customer_name: Customer's name
            signature_image_path: Path to the signature image
            verified: Whether the signature has been verified
        
        Returns:
            Confirmation of storage
        """
        # Mock store
        _mock_signatures[customer_id] = {
            "customer_id": customer_id,
            "customer_name": customer_name,
            "signature_image_path": signature_image_path,
            "created_at": datetime.utcnow().isoformat(),
            "verified": verified
        }
        
        return {
            "success": True,
            "signature_id": customer_id,
            "message": "[Mock] Signature stored successfully"
        }
    
    @mcp.tool()
    async def list_signatures(
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all stored reference signatures.
        
        Args:
            limit: Maximum number of signatures to return
            offset: Pagination offset
        
        Returns:
            List of signature records
        """
        signatures = list(_mock_signatures.values())
        paginated = signatures[offset:offset + limit]
        
        return {
            "success": True,
            "signatures": paginated,
            "total": len(signatures),
            "limit": limit,
            "offset": offset
        }
    
    @mcp.tool()
    async def delete_signature(
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Delete a reference signature.
        
        Args:
            customer_id: Customer identifier
        
        Returns:
            Confirmation of deletion
        """
        if customer_id in _mock_signatures:
            del _mock_signatures[customer_id]
            return {
                "success": True,
                "message": f"Signature for {customer_id} deleted"
            }
        
        return {
            "success": False,
            "message": f"Signature for {customer_id} not found"
        }
    
    @mcp.tool()
    async def compare_signatures(
        extracted_signature_path: str,
        reference_signature_path: str
    ) -> Dict[str, Any]:
        """
        Compare two signature images.
        
        This is a simplified comparison - in production, use ML models.
        
        Args:
            extracted_signature_path: Path to extracted signature
            reference_signature_path: Path to reference signature
        
        Returns:
            Comparison result with similarity score
        """
        # Mock comparison
        return {
            "success": True,
            "match": True,
            "similarity_score": 0.87,
            "confidence": 0.85,
            "analysis": {
                "stroke_consistency": 0.89,
                "shape_similarity": 0.85,
                "pressure_pattern": 0.86
            },
            "message": "[Mock] Signatures compared using visual analysis"
        }
