"""
NNP-AI MCP Tools Service - FastMCP tool server.

Hosts pluggable tools for the agentic pipeline.
Uses FastMCP's built-in HTTP transport - no separate FastAPI needed.
"""
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file for local development
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    print(f"📋 Loading environment from: {env_file}")
    load_dotenv(env_file)
    
    # Check enterprise authentication
    if os.getenv('GENAI_SERVICE_ACCOUNT') and os.getenv('GENAI_SERVICE_ACCOUNT_PASSWORD'):
        print(f"✅ Enterprise auth configured: {os.getenv('GENAI_SERVICE_ACCOUNT')}")
    else:
        print(f"⚠️  No enterprise authentication configured")
else:
    print(f"⚠️  No .env file found at {env_file}")

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import get_config

from tools.ocr_tool import register_ocr_tools
from tools.pdf_utils import register_pdf_tools
from tools.signature_provider import register_signature_tools
from tools.extraction_tool import register_extraction_tools
from tools.signature_detection_tool import register_signature_detection_tools
from tools.signature_verification_tool import register_signature_verification_tools


# Initialize config
config = get_config()

# Create FastMCP server
mcp = FastMCP(name="nnp-ai-tools")


# ============================================
# Custom Routes (Health Check, Info)
# ============================================
@mcp.custom_route("/health", methods=["GET"])
async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({
        "status": "healthy",
        "service": "mcp-tools",
        "enabled_tools": config.mcp.tools.enabled
    })


@mcp.custom_route("/", methods=["GET"])
async def root(request: Request) -> JSONResponse:
    """Root endpoint with service info."""
    return JSONResponse({
        "service": "NNP-AI MCP Tools",
        "version": "1.0.0",
        "transport": config.mcp.server.transport,
        "enabled_tools": config.mcp.tools.enabled,
        "tools_config": {
            "pdf_utils": config.mcp.tools.pdf_utils,
            "signature_provider": config.mcp.tools.signature_provider
        }
    })


@mcp.custom_route("/tools", methods=["GET"])
async def list_tools(request: Request) -> JSONResponse:
    """List all registered tools."""
    return JSONResponse({
        "tools": [
            # Business-level tools (called by agent orchestrator)
            {"name": "extract_payment_fields", "description": "Extract payment fields from document via Gemini Vision"},
            {"name": "validate_extraction", "description": "Validate extracted fields against business rules"},
            {"name": "detect_signatures", "description": "Detect signature regions in document via Gemini Vision"},
            {"name": "crop_signature", "description": "Crop a signature region from PDF using PyMuPDF"},
            {"name": "validate_signature_crops", "description": "Validate cropped signature quality"},
            {"name": "verify_signature", "description": "Verify signature with M1-M7 metrics + FIV 1.0 scoring"},
            # Reference signature management
            {"name": "get_reference_signature", "description": "Get reference signature for a customer"},
            {"name": "store_signature", "description": "Store a new signature reference"},
            {"name": "list_signatures", "description": "List all reference signatures"},
            {"name": "delete_signature", "description": "Delete a reference signature"},
            # Low-level utilities
            {"name": "ocr_extract", "description": "Extract text from images/documents"},
            {"name": "pdf_to_images", "description": "Convert PDF pages to images"},
            {"name": "crop_region", "description": "Crop a region from an image"},
            {"name": "get_pdf_metadata", "description": "Get PDF file metadata"},
        ]
    })


# ============================================
# Register Tools Based on Config
# ============================================
enabled_tools = config.mcp.tools.enabled

if "ocr" in enabled_tools:
    register_ocr_tools(mcp, config)

if "pdf_utils" in enabled_tools:
    register_pdf_tools(mcp, config)

if "signature_provider" in enabled_tools:
    register_signature_tools(mcp, config)

if "extraction" in enabled_tools:
    register_extraction_tools(mcp, config)

if "signature_detection" in enabled_tools:
    register_signature_detection_tools(mcp, config)

if "signature_verification" in enabled_tools:
    register_signature_verification_tools(mcp, config)


# ============================================
# Run Server
# ============================================
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="NNP-AI MCP Tools Server")
    parser.add_argument(
        "--transport", 
        choices=["http", "sse", "stdio"], 
        default="sse",
        help="Transport protocol (default: sse)"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8002,
        help="Port to bind to (default: 8002)"
    )
    args = parser.parse_args()
    
    # Run with FastMCP's built-in server
    mcp.run(
        transport=args.transport,
        host=args.host,
        port=args.port
    )
