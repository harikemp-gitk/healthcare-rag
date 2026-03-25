# User Message
#      ↓
# Supervisor (routes based on intent)
#      ├→ FAQ question? → FAQ Agent → (uses RAG tools to search knowledge base)
#      ├→ Book appointment? → Booking Agent → (uses Calendar tools) → Confirmation Agent (sends email) 
#      └→ Goodbye? → FINISH


# route_supervisor


from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import ToolMessage
from config.memory import checkpointer, store
from agents.state import AgentState
from agents.supervisor import supervisor_node
from agents.faq_agent import faq_node, tools as faq_tools
from agents.booking_agent import create_booking_node
from agents.confirmation_agent import create_confirmation_node
from tools.mcp_tools import get_mcp_client, get_calendar_tools, get_gmail_tools


def route_supervisor(state:AgentState):
    """Route to the appropriate agent based on the suprvisor's decision."""
    next_agent = state.get("next_agent","FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent

# Booking Agent Response
#          ↓
# Has tool_calls? → Yes → booking_tools (check/create calendar)
#          ↓
#         No
#          ↓
# Previous msg is ToolMessage? → Yes → confirmation_agent (send email)
#          ↓
#         No
#          ↓
#      → END (still collecting user info)

def route_after_booking(state:AgentState):
    """After booking is complete, route to confirmation agent to send email.
    - Has tool calls -> "booking_tools" (executes calendar actions to finalize booking)
    -Just processed a tool result -> "confirmation_agent" (booking done, send confirmation email)
    - Otherwise- > END
    """
    last_msg=state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "booking_tools"
    
    if len(state["messages"]) >= 2 and isinstance(state["messages"][-2], ToolMessage):
        return "confirmation_agent"
    
    return END


async def build_workflow():
    """Build the entire workflow graph with supervisor, FAQ agent, booking agent, and confirmation agent."""
    client = get_mcp_client()
    calendar_tools = await get_calendar_tools(client)
    gmail_tools = await get_gmail_tools(client)

    if not calendar_tools:
        raise RuntimeError("No calendar tools found. Is Composio MCP server running?")

    if not gmail_tools:
        raise RuntimeError("No Gmail tools found. Is Composio MCP server running?")
    
    booking_node, booking_tools = create_booking_node(calendar_tools)
    confirmation_node, confirmation_tools = create_confirmation_node(gmail_tools)

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("faq_agent", faq_node)
    builder.add_node("booking_agent", booking_node)
    builder.add_node("confirmation_agent", confirmation_node)

    builder.add_node("booking_tools", ToolNode(booking_tools))
    builder.add_node("confirmation_tools", ToolNode(confirmation_tools))
    builder.add_node("faq_tools", ToolNode(faq_tools))

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route_supervisor,{
        "faq_agent": "faq_agent",
        "booking_agent": "booking_agent",
        END: END,
    })

    #  start-> supervisor -> faq_agent -> faq_tools -> faq_agent (loop back to faq_agent after using tools)

    builder.add_conditional_edges("faq_agent", tools_condition, {
        "faq_tools": "faq_tools",
        END: END
    })
    builder.add_edge("faq_tools", "faq_agent")

    # start -> supervisor -> booking_agent -> booking_tools -> booking_agent (loop back to booking_agent after using tools)

    builder.add_conditional_edges("booking_agent", route_after_booking, {
        "booking_tools": "booking_tools",
        "confirmation_agent": "confirmation_agent",
        END: END
    })
    builder.add_edge("booking_tools", "booking_agent")

    # confirmation_agent -> confirmation_tools -> END (one way flow to send confirmation email after booking)

    builder.add_conditional_edges("confirmation_agent", tools_condition, {
        "confirmation_tools": "confirmation_tools",
        END: END
    })
    
    builder.add_edge("confirmation_tools", "confirmation_agent")

    graph= builder.compile(checkpointer=checkpointer, store=store)

    return graph, client


def build_faq_only_workflow():
    """Build a simplified workflow graph with just the supervisor and FAQ agent for testing."""
    from langgraph.graph import MessagesState
    builder = StateGraph(MessagesState)
    builder.add_node("faq_agent", faq_node)
    builder.add_node("tools", ToolNode(faq_tools))
    builder.add_edge(START, "faq_agent")
    builder.add_conditional_edges("faq_agent", tools_condition)
    builder.add_edge("tools", "faq_agent")
    graph= builder.compile(checkpointer=checkpointer)
    return graph