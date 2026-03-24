import os
import token
from dotenv import load_dotenv
load_dotenv()

from langchain_mcp_adapters.client import MultiServerMCPClient

COMPOSIO_API_KEY = os.getenv("COMPOSIO_API_KEY" ,"")
COMPOSIO_MCP_URL = os.getenv("COMPOSIO_MCP_URL", "")

def get_mcp_client():
    """Get the MCP client fot Composio MCP Server."""

    client = MultiServerMCPClient(
    {
         "composio": {
            # make sure you start your weather server on port 8000
            "url": COMPOSIO_MCP_URL,
            "transport": "http",
            "headers": {
                "x-api-key": COMPOSIO_API_KEY,
            },
        }
    }
    )
    return client

async def get_calendar_tools(client: MultiServerMCPClient):
    """Get calendar tools from the Composio MCP Server."""
    tools = await client.get_tools()
    return [t for t in tools if "calendar" in t.name.lower()]


async def get_gmail_tools(client: MultiServerMCPClient):
    """Get Gmail tools from the Composio MCP Server."""
    tools = await client.get_tools()
    return [t for t in tools if "gmail" in t.name.lower()]

async def get_all_tools(client: MultiServerMCPClient):
    """Get all tools from the Composio MCP Server."""
    tools = await client.get_tools()
    return tools
