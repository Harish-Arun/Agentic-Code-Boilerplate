import json
import os
from contextlib import asynccontextmanager
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

# We are connecting to the SSE endpoint of our mcp-tools service
# Uses environment variable for Docker networking, falls back to localhost for local dev
MCP_SERVER_URL = os.environ.get("MCP_SERVICE_URL", "http://localhost:8002") + "/sse"

@asynccontextmanager
async def get_mcp_client():
    """
    Context manager to get a connected MCP client session.
    """
    try:
        print(f"🔗 Connecting to MCP server at: {MCP_SERVER_URL}")
        # Connect via SSE (Server-Sent Events)
        async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
            print("✅ SSE streams established")
            async with ClientSession(read_stream, write_stream) as session:
                print("✅ ClientSession created, initializing...")
                # Initialize with the server
                await session.initialize()
                print("✅ MCP session initialized successfully")
                
                yield session
    except Exception as e:
        print(f"\n❌ MCP CONNECTION FAILED")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise


async def call_tool_on_session(session: ClientSession, tool_name: str, arguments: dict) -> dict:
    """
    Call an MCP tool on an existing session and parse the JSON result.

    Args:
        session: Active MCP ClientSession
        tool_name: Name of the tool to call
        arguments: Tool arguments

    Returns:
        Parsed dict from tool response
    """
    try:
        print(f"🛠️ Calling MCP tool: {tool_name}")
        result = await session.call_tool(tool_name, arguments=arguments)
        if result.content:
            text = result.content[0].text
            try:
                parsed = json.loads(text)
                print(f"✅ Tool {tool_name} returned successfully (success={parsed.get('success')})")
                return parsed
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"⚠️ Tool {tool_name} returned non-JSON: {text[:200]}")
                return {"raw_text": text}
        print(f"⚠️ Tool {tool_name} returned empty content")
        return {}
    except Exception as e:
        print(f"\n❌ TOOL CALL FAILED: {tool_name}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        raise


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """
    Call an MCP tool and return the parsed result.

    Creates a new session for each call. For multiple calls,
    use get_mcp_client() context manager and call_tool_on_session() instead.
    """
    async with get_mcp_client() as session:
        return await call_tool_on_session(session, tool_name, arguments)
