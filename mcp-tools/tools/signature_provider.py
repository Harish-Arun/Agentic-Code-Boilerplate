"""
Signature Provider Tool - Reference signature management.

Provides tools to retrieve and manage reference signatures for verification.
Reference signatures are stored in the local filesystem (data/reference/),
simulating the ISV (Identity & Signature Verification) service that returns
signature blobs in production.

Folder structure:
  data/reference/
    ├── CUST001.png       # Customer ID maps directly to filename
    ├── CUST002.jpg
    └── john_smith.png    # Also supports name-based lookup
"""
from typing import Dict, Any, Optional, List
from fastmcp import FastMCP
from datetime import datetime
import os
import base64
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "shared"))

from config import AppConfig


# Supported image extensions for reference signatures
SIGNATURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def _scan_reference_dir(reference_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Scan the reference directory and build an index of available signatures.
    Maps customer_id (stem of filename) -> signature metadata.
    """
    ref_path = Path(reference_dir)
    index = {}
    
    if not ref_path.exists():
        return index
    
    for f in ref_path.iterdir():
        if f.is_file() and f.suffix.lower() in SIGNATURE_EXTENSIONS:
            customer_id = f.stem.upper()
            # Read file bytes and base64 encode to simulate ISV blob response
            file_bytes = f.read_bytes()
            mime_ext = f.suffix.lstrip(".").lower()
            mime_type = f"image/{mime_ext}" if mime_ext != "jpg" else "image/jpeg"
            
            index[customer_id] = {
                "customer_id": customer_id,
                "customer_name": f.stem.replace("_", " ").title(),
                "signature_image_path": str(f.resolve()),
                "image_blob": base64.b64encode(file_bytes).decode("utf-8"),
                "blob_mime_type": mime_type,
                "blob_size_bytes": len(file_bytes),
                "file_size_bytes": len(file_bytes),
                "format": f.suffix.lstrip(".").upper(),
                "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                "verified": True,
            }
    
    return index


def register_signature_tools(mcp: FastMCP, config: AppConfig):
    """Register signature-related tools with the MCP server."""
    
    reference_dir = config.storage.reference_dir
    
    @mcp.tool()
    async def get_reference_signature(
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Get reference signature for a customer from the local reference folder.
        Simulates calling the ISV service which returns a signature blob.
        
        Args:
            customer_id: Customer identifier (matched against filenames)
        
        Returns:
            Reference signature details including image path
        """
        # Rebuild index on each call (folder contents may change)
        index = _scan_reference_dir(reference_dir)
        
        lookup_id = customer_id.upper()
        
        # Direct match by customer ID
        if lookup_id in index:
            return {
                "success": True,
                "signature": index[lookup_id],
                "message": "Reference signature found"
            }
        
        # Fuzzy match: search by partial name in filenames
        for cid, sig_data in index.items():
            if lookup_id in cid or cid in lookup_id:
                return {
                    "success": True,
                    "signature": sig_data,
                    "message": f"Reference signature found (partial match: {cid})"
                }
        
        # If only one reference exists, use it as default
        if len(index) == 1:
            only_sig = list(index.values())[0]
            return {
                "success": True,
                "signature": only_sig,
                "message": "Using only available reference signature as default"
            }
        
        return {
            "success": False,
            "signature": None,
            "message": f"No reference signature found for customer: {customer_id}. "
                       f"Place a signature image in {reference_dir}/"
        }
    
    @mcp.tool()
    async def store_signature(
        customer_id: str,
        customer_name: str,
        signature_image_path: str,
        verified: bool = False
    ) -> Dict[str, Any]:
        """
        Store a new reference signature by copying it to the reference folder.
        
        Args:
            customer_id: Customer identifier
            customer_name: Customer's name
            signature_image_path: Path to the source signature image
            verified: Whether the signature has been verified
        
        Returns:
            Confirmation of storage
        """
        import shutil
        
        ref_path = Path(reference_dir)
        ref_path.mkdir(parents=True, exist_ok=True)
        
        source = Path(signature_image_path)
        if not source.exists():
            return {
                "success": False,
                "message": f"Source signature not found: {signature_image_path}"
            }
        
        # Copy to reference directory with customer_id as filename
        dest = ref_path / f"{customer_id.upper()}{source.suffix}"
        shutil.copy2(str(source), str(dest))
        
        return {
            "success": True,
            "signature_id": customer_id.upper(),
            "stored_path": str(dest),
            "message": f"Signature stored for {customer_name} at {dest}"
        }
    
    @mcp.tool()
    async def list_signatures(
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        List all stored reference signatures in the reference folder.
        
        Args:
            limit: Maximum number of signatures to return
            offset: Pagination offset
        
        Returns:
            List of signature records
        """
        index = _scan_reference_dir(reference_dir)
        signatures = list(index.values())
        paginated = signatures[offset:offset + limit]
        
        return {
            "success": True,
            "signatures": paginated,
            "total": len(signatures),
            "reference_dir": reference_dir,
            "limit": limit,
            "offset": offset
        }
    
    @mcp.tool()
    async def delete_signature(
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Delete a reference signature from the reference folder.
        
        Args:
            customer_id: Customer identifier
        
        Returns:
            Confirmation of deletion
        """
        index = _scan_reference_dir(reference_dir)
        lookup_id = customer_id.upper()
        
        if lookup_id in index:
            sig_path = Path(index[lookup_id]["signature_image_path"])
            if sig_path.exists():
                sig_path.unlink()
                return {
                    "success": True,
                    "message": f"Signature for {customer_id} deleted from {sig_path}"
                }
        
        return {
            "success": False,
            "message": f"Signature for {customer_id} not found in {reference_dir}"
        }
