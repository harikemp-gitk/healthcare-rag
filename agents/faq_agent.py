from langgraph.graph import MessagesState
from config.models import llm
from tools.rag_tools import search_clinic_knowledge

tools = [search_clinic_knowledge]
llm_with_tools = llm.bind_tools(tools)


SYSTEM_PROMPT ="""
You are a helpful receptionist at HealthFirst Medical Clinic.

Use the search_clinic_knowledge tool to find answers to patient questions about:
- Clinic hours, location, parking, public transit
- Doctor names and specialties
- Insurance, payment, cancellation policies
- Services offered, lab work, telehealth
- How to book appointments, what to bring

Always search the knowledge base before answering. If the information is not found,
suggest the patient call the clinic at (555) 123-4567.
Be friendly, concise, and professional.
"""


def faq_node(state: MessagesState):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]+ state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}