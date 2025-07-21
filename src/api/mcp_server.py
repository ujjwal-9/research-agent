"""MCP (Model Context Protocol) server implementation."""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from src.tools.function_calls import FunctionCallManager
from src.config import settings


@dataclass
class MCPTool:
    """MCP tool definition."""
    name: str
    description: str
    inputSchema: Dict[str, Any]


@dataclass
class MCPResponse:
    """MCP response structure."""
    content: List[Dict[str, Any]]
    isError: bool = False


class MCPServer:
    """MCP server for exposing research system tools."""
    
    def __init__(self):
        self.function_manager = FunctionCallManager()
        self.server_info = {
            "name": settings.mcp_server_name,
            "version": settings.mcp_server_version
        }
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        return self.server_info
    
    def list_tools(self) -> List[MCPTool]:
        """List available tools."""
        function_defs = self.function_manager.get_all_function_definitions()
        
        tools = []
        for func_def in function_defs:
            tool = MCPTool(
                name=func_def["name"],
                description=func_def["description"],
                inputSchema=func_def["parameters"]
            )
            tools.append(tool)
        
        return tools
    
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """Execute a tool call."""
        try:
            result = await self.function_manager.execute_function(name, arguments)
            
            if result.get("success", False):
                content = [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
                return MCPResponse(content=content, isError=False)
            else:
                content = [
                    {
                        "type": "text", 
                        "text": f"Tool execution failed: {result.get('error', 'Unknown error')}"
                    }
                ]
                return MCPResponse(content=content, isError=True)
                
        except Exception as e:
            content = [
                {
                    "type": "text",
                    "text": f"Tool execution error: {str(e)}"
                }
            ]
            return MCPResponse(content=content, isError=True)
    
    def to_mcp_format(self) -> Dict[str, Any]:
        """Convert server to MCP format."""
        tools = self.list_tools()
        
        return {
            "jsonrpc": "2.0",
            "result": {
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": self.server_info,
                "tools": [asdict(tool) for tool in tools]
            }
        }


# Example MCP configuration for .kiro/settings/mcp.json
MCP_CONFIG_EXAMPLE = {
    "mcpServers": {
        "research-system": {
            "command": "python",
            "args": ["-m", "src.api.mcp_server"],
            "env": {},
            "disabled": False,
            "autoApprove": [
                "search_documents",
                "list_documents"
            ]
        }
    }
}


if __name__ == "__main__":
    # Simple MCP server runner for testing
    import asyncio
    
    async def main():
        server = MCPServer()
        
        print("MCP Server Info:")
        print(json.dumps(server.get_server_info(), indent=2))
        
        print("\nAvailable Tools:")
        tools = server.list_tools()
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")
        
        print("\nTesting document search...")
        response = await server.call_tool("search_documents", {"query": "test"})
        print(f"Response: {response.content[0]['text'][:200]}...")
    
    asyncio.run(main())