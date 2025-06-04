#!/usr/bin/env python3
"""
Quick health check script to test the PSX Financial MCP Server
"""

import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_health_check():
    """Connect to the MCP server and call the health check tool"""
    try:
        print("🔍 Connecting to PSX Financial MCP Server...")
        
        # Connect to the server
        server_params = StdioServerParameters(
            command="python",
            args=["server.py"],
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                # Initialize the session
                await session.initialize()
                print("✅ Connected to MCP server")
                
                # List available tools
                tools = await session.list_tools()
                print(f"📋 Available tools: {[tool.name for tool in tools.tools]}")
                
                # Call the health check tool
                print("\n🩺 Calling health check...")
                result = await session.call_tool("psx_health_check", arguments={})
                
                # Parse and display the health check result
                if result.content:
                    health_data = json.loads(result.content[0].text)
                    print("\n" + "="*50)
                    print("📊 PSX FINANCIAL SERVER HEALTH CHECK RESULTS")
                    print("="*50)
                    print(f"Status: {health_data.get('status', 'Unknown')}")
                    print(f"Server Name: {health_data.get('server_name', 'Unknown')}")
                    print(f"Version: {health_data.get('version', 'Unknown')}")
                    print(f"Resource Manager Healthy: {health_data.get('resource_manager_healthy', False)}")
                    print(f"Index Documents: {health_data.get('index_documents', 0)}")
                    print(f"Companies Available: {health_data.get('companies_available', 0)}")
                    
                    models = health_data.get('models_available', {})
                    print(f"\nModel Status:")
                    print(f"  - Embeddings: {'✅' if models.get('embeddings') else '❌'}")
                    print(f"  - LLM: {'✅' if models.get('llm') else '❌'}")
                    print(f"  - Index: {'✅' if models.get('index') else '❌'}")
                    
                    print(f"\nCapabilities: {', '.join(health_data.get('capabilities', []))}")
                    
                    if health_data.get('status') == 'healthy':
                        print("\n🎉 All systems operational!")
                    else:
                        print(f"\n⚠️ System status: {health_data.get('status')}")
                    
                    print("="*50)
                else:
                    print("❌ No health check data received")
                    
    except Exception as e:
        print(f"❌ Failed to connect to server or call health check: {e}")
        print("💡 Make sure the server is running with: chainlit run server.py")

if __name__ == "__main__":
    asyncio.run(test_health_check()) 