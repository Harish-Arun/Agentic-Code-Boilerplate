"""
NNP-AI MCP Tools Service - FastMCP tool server.

Hosts pluggable tools for the agentic pipeline.
Uses FastMCP's built-in HTTP transport - no separate FastAPI needed.
"""
import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from config import get_config

from tools.ocr_tool import register_ocr_tools
from tools.pdf_utils import register_pdf_tools
from tools.signature_provider import register_signature_tools


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
            "ocr": config.mcp.tools.ocr,
            "pdf_utils": config.mcp.tools.pdf_utils,
            "signature_provider": config.mcp.tools.signature_provider
        }
    })


@mcp.custom_route("/tools", methods=["GET"])
async def list_tools(request: Request) -> JSONResponse:
    """List all registered tools."""
    return JSONResponse({
        "tools": [
            {"name": "ocr_extract", "description": "Extract text from images/documents"},
            {"name": "pdf_to_images", "description": "Convert PDF pages to images"},
            {"name": "crop_region", "description": "Crop a region from an image"},
            {"name": "get_reference_signature", "description": "Get reference signature for a customer"},
            {"name": "store_signature", "description": "Store a new signature reference"}
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


# ============================================
# Run Server
# ============================================
if __name__ == "__main__":
    import argparse
    
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
