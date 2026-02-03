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
    # Connect via SSE (Server-Sent Events)
    async with sse_client(MCP_SERVER_URL) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize with the server
            await session.initialize()
            
            yield session
