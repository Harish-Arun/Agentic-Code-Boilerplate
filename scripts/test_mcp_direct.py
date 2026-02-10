"""Test MCP connection directly."""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agents"))

from mcp_client import get_mcp_client


async def test_mcp():
    """Test MCP client connection."""
    print("🔧 Testing MCP connection to http://localhost:8002/sse")
    
    try:
        async with get_mcp_client() as mcp:
            print("✅ MCP connection established")
            
            # List available tools
            tools = await mcp.list_tools()
            print(f"\n📋 Available tools ({len(tools.tools)}):")
            for tool in tools.tools:
                print(f"   - {tool.name}: {tool.description}")
            
            # Test a simple tool call
            print("\n🧪 Testing pdf_to_images tool...")
            result = await mcp.call_tool("pdf_to_images", arguments={
                "pdf_path": str(Path(__file__).parent.parent / "data/uploads/sample.pdf"),
                "pages": "1"
            })
            print(f"✅ Tool call result: {result}")
            
    except Exception as e:
        print(f"❌ MCP connection failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_mcp())
