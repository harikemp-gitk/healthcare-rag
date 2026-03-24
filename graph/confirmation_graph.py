"""
This module wires the confirmation agent node into Langgraph. It defines a function to create the confirmation node using the Gmail tool, and then builds the graph by adding the confirmation agent and its associated tools. The graph is structured to allow the confirmation agent to invoke the Gmail tool when needed, based on the messages state.
"""
from langgraph.graph import StateGraph,MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from agents.confirmation_agent import create_confirmation_node
from tools.mcp_tools import get_mcp_client, get_gmail_tools

async def build_confirmation_graph():
    client = get_mcp_client()
    gmail_tools  = await get_gmail_tools(client)
    

    if not gmail_tools:
        raise RuntimeError("No Gmail tools found in MCP server. Please add tools with the 'gmail' tag.")

    confirmation_node, tools = create_confirmation_node(gmail_tools)

    builder = StateGraph(MessagesState)

    builder.add_node("confirmation_agent", confirmation_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "confirmation_agent")
    builder.add_conditional_edges("confirmation_agent", tools_condition)
    builder.add_edge("tools", "confirmation_agent")

    confirmation_graph = builder.compile()
    return confirmation_graph, client