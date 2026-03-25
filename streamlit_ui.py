"""
Streamlit frontend for HealthFirst Medical Clinic Multi-Agent System
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


@st.cache_resource
def get_graph_and_client():
    """
    Initialize the graph and client.
    Cached so it runs only once per session/reload cycle.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Try full workflow
        graph, client = loop.run_until_complete(build_workflow())
        return graph, client, "Full Multi-Agent System"
    except Exception as e:
        print(f"MCP Connection failed or other error: {e}")
        # Fallback to FAQ only
        graph = build_faq_only_workflow()
        return graph, None, "FAQ-Only Mode"


def main():
    st.set_page_config(page_title="HealthFirst Clinic", page_icon="🏥")
    
    st.title("🏥 HealthFirst Medical Clinic")
    st.markdown("Welcome to our intelligent scheduling assistant.")

    # Sidebar configuration
    st.sidebar.header("System Status")
    
    # Initialize graph (cached)
    graph, client, mode = get_graph_and_client()
    st.sidebar.success(f"Running in: {mode}")

    # Session State for conversation
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())[:8]
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.sidebar.markdown(f"**Thread ID:** `{st.session_state.thread_id}`")
    if st.sidebar.button("Reset Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

    # User Input
    if user_input := st.chat_input("How can I help you today?"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Prepare config
        config = {
            "configurable": {
                "thread_id": st.session_state.thread_id,
                "user_id": "streamlit_user"
            }
        }

        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # We need a new loop for the invoke since we are inside Streamlit's flow
                # asyncio.run handles creating/closing a loop for this specific call
                response = asyncio.run(graph.ainvoke(
                    {"messages": [("user", user_input)]}, 
                    config
                ))
                
                bot_text = _extract_text(response["messages"])
                message_placeholder.markdown(bot_text)
                
                # Update history
                st.session_state.messages.append({"role": "assistant", "content": bot_text})
                
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()