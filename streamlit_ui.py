"""
Streamlit frontend for HealthFirst Clinical Facility Multi-Agent System
Run: streamlit run streamlit_app.py
"""
import asyncio
import uuid
import streamlit as st
from langchain_core.messages import AIMessage
from graph.workflow import build_workflow, build_faq_only_workflow


def _extract_text(messages) -> str:
    """Extract text from the last AI message, handling Bedrock's list content format."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
            if parts:
                return "\n".join(parts)
    return "I'm sorry, I couldn't generate a response. Please try again."

# --- Page Config ---
st.set_page_config(
    page_title="HealthFirst Clinical Facility",
    page_icon="🏥",
    layout="centered",
)

# --- Custom CSS for Background ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #9fc0d8;
        color: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Session State Init ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "user_id" not in st.session_state:
    st.session_state.user_id = "customer"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "mode" not in st.session_state:
    st.session_state.mode = "faq_only"
if "graph" not in st.session_state:
    st.session_state.graph = build_faq_only_workflow()
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = None


def connect_full_system():
    """Try to connect to Composio MCP and build the full multi-agent graph."""
    try:
        # Use a manual loop to avoid closing it (which asyncio.run does), 
        # keeping the MCP client connection alive.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        graph, client = loop.run_until_complete(build_workflow())
        st.session_state.graph = graph
        st.session_state.mcp_client = client
        st.session_state.mode = "full"
        return True
    except Exception as e:
        st.error(f"Could not connect to MCP server: {e}")
        return False

# --- Sidebar ---
with st.sidebar:
    st.title("HealthFirst Clinical Facility")
    st.caption("Multi-Agent Appointment System")

    st.divider()

    # Mode selector
    st.subheader("Mode")
    if st.session_state.mode == "faq_only":
        st.info("Frequently Asked Questions (FAQ) Mode")
        if st.button("Connect Full System"):
            with st.spinner("Connecting to Composio MCP..."):
                if connect_full_system():
                    st.success("Connected! Full system active.")
                    st.rerun()
    else:
        st.success("Full Multi-Agent System")

    st.divider()

    # User ID
    st.subheader("User")
    new_user = st.text_input("User ID", value=st.session_state.user_id)
    if new_user != st.session_state.user_id:
        st.session_state.user_id = new_user

    st.divider()

    # Session info
    st.subheader("Session")
    st.text(f"Thread: {st.session_state.thread_id}")
    st.text(f"User: {st.session_state.user_id}")

    if st.button("New Chat"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.subheader("Try asking:")
    if st.session_state.mode == "full":
        st.markdown("""
        - What are your clinic hours?
        - I'd like to book an appointment
        - Which doctors are available?
        - What's the cancellation policy?
        - Book me with Dr. Chen tomorrow at 10 AM
        """)
    else:
        st.markdown("""
        - What are your clinic hours?
        - Which doctors work here?
        - What's the cancellation policy?
        - Do you accept insurance?
        - Where is the clinic located?
        """)

    st.divider()
    st.caption("Powered by LangGraph + AWS Bedrock")

# @st.cache_resource
# def get_graph_and_client():
#     """
#     Initialize the graph and client.
#     Cached so it runs only once per session/reload cycle.
#     """
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
#     try:
#         # Try full workflow
#         graph, client = loop.run_until_complete(build_workflow())
#         return graph, client, "Full Multi-Agent System"
#     except Exception as e:
#         print(f"MCP Connection failed or other error: {e}")
#         # Fallback to FAQ only
#         graph = build_faq_only_workflow()
#         return graph, None, "FAQ-Only Mode"
    
# --- Main Chat Area ---
st.title("🩺 HealthFirst Clinical Facility")
if st.session_state.mode == "full":
    st.caption("Ask questions, book appointments, or request confirmations. The supervisor routes automatically.")
else:
    st.caption("Ask me anything about our clinic, doctors, policies, and services.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
            "user_id": st.session_state.user_id,
        }
    }

    graph = st.session_state.graph

    with st.chat_message("assistant"):
        with st.spinner("Answering your question, please wait..."):
            try:
                if st.session_state.mode == "full":
                    # Create a new loop for this interaction
                    result = asyncio.run(graph.ainvoke(
                        {"messages": [("user", prompt)]},
                        config,
                    ))
                else:
                    result = graph.invoke(
                        {"messages": [("user", prompt)]},
                        config,
                    )
                response = _extract_text(result["messages"])
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")


