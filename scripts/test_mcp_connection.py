import httpx
import asyncio
import json

MCP_HOST = "http://localhost:8002"

async def test_mcp_connection():
    print(f"🔌 Testing connection to MCP Tools Service at {MCP_HOST}...\n")
    
    async with httpx.AsyncClient() as client:
        # 1. Check Root Info
        print("1. Checking Service Info (GET /)...")
        try:
            resp = await client.get(f"{MCP_HOST}/")
            if resp.status_code == 200:
                print("   ✅ Service is UP")
                print(f"   Response: {json.dumps(resp.json(), indent=2)}")
            else:
                print(f"   ❌ Failed: {resp.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return

        print("\n------------------------------------------------\n")

        # 2. Check Health
        print("2. Checking Health (GET /health)...")
        try:
            resp = await client.get(f"{MCP_HOST}/health")
            if resp.status_code == 200:
                print("   ✅ Healthy")
                print(f"   Response: {json.dumps(resp.json(), indent=2)}")
            else:
                print(f"   ❌ Failed: {resp.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("\n------------------------------------------------\n")

        # 3. List Tools
        print("3. Listing Registered Tools (GET /tools)...")
        try:
            resp = await client.get(f"{MCP_HOST}/tools")
            if resp.status_code == 200:
                tools = resp.json().get("tools", [])
                print(f"   ✅ Found {len(tools)} tools:")
                for tool in tools:
                    print(f"   - 🛠️  {tool['name']}: {tool['description']}")
            else:
                print(f"   ❌ Failed: {resp.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")

        print("\n------------------------------------------------\n")
        
        # 4. Note on Tool Execution
        print("ℹ️  To execute tools, a full MCP client (like ClientSession) is required.")
        print("   The endpoint for tool execution is via SSE at: /sse")
        print("   But verifying the /tools list confirms the server is correctly configured.")

if __name__ == "__main__":
    asyncio.run(test_mcp_connection())
