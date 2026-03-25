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
        background-color: #e8f4f8;
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
        graph, client = asyncio.run(build_workflow())
        st.session_state.graph = graph
        st.session_state.mcp_client = client
        st.session_state.mode = "full"
        return True
    except Exception as e:
        st.error(f"Could not connect to MCP server: {e}")
        return False

# --- Sidebar ---
with st.sidebar:
    st.title("Healthcare Clinical Facility")
    st.caption("Multi-Agent Appointment System")

    st.divider()

    # Mode selector
    st.subheader("Mode")
    if st.session_state.mode == "faq_only":
        st.info("FAQ Only (no MCP)")
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
st.title("Healthcare Clinical Facility")
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
        with st.spinner("Thinking..."):
            if st.session_state.mode == "full":
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


# def main():
#     st.set_page_config(page_title="HealthFirst Clinic", page_icon="🏥")
    
#     st.title("🏥 HealthFirst Medical Clinic")
#     st.markdown("Welcome to our intelligent scheduling assistant.")

#     # Sidebar configuration
#     st.sidebar.header("System Status")
    
#     # Initialize graph (cached)
#     graph, client, mode = get_graph_and_client()
#     st.sidebar.success(f"Running in: {mode}")

#     # Session State for conversation
#     if "thread_id" not in st.session_state:
#         st.session_state.thread_id = str(uuid.uuid4())[:8]
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
    
#     st.sidebar.markdown(f"**Thread ID:** `{st.session_state.thread_id}`")
#     if st.sidebar.button("Reset Conversation"):
#         st.session_state.thread_id = str(uuid.uuid4())[:8]
#         st.session_state.messages = []
#         st.rerun()

#     # Display Chat History
#     for msg in st.session_state.messages:
#         role = msg["role"]
#         content = msg["content"]
#         with st.chat_message(role):
#             st.markdown(content)

#     # User Input
#     if user_input := st.chat_input("How can I help you today?"):
#         # Display user message
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         # Prepare config
#         config = {
#             "configurable": {
#                 "thread_id": st.session_state.thread_id,
#                 "user_id": "streamlit_user"
#             }
#         }

#         # Generate response
#         with st.chat_message("assistant"):
#             message_placeholder = st.empty()
#             message_placeholder.markdown("Thinking...")
            
#             try:
#                 # We need a new loop for the invoke since we are inside Streamlit's flow
#                 # asyncio.run handles creating/closing a loop for this specific call
#                 response = asyncio.run(graph.ainvoke(
#                     {"messages": [("user", user_input)]}, 
#                     config
#                 ))
                
#                 bot_text = _extract_text(response["messages"])
#                 message_placeholder.markdown(bot_text)
                
#                 # Update history
#                 st.session_state.messages.append({"role": "assistant", "content": bot_text})
                
#             except Exception as e:
#                 message_placeholder.error(f"Error: {str(e)}")


# if __name__ == "__main__":
#     main()