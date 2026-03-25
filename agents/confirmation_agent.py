"""Confirmation Agent : This is responsible for confirming the appointment details with the user
before finalizing the booking."""
from agents.state import AgentState 
from config.models import llm


SYSTEM_PROMPT = """You are a confirmation assistant at HealthFirst Medical Clinic.
 
Your job is to send appointment confirmation emails to patients after booking.
Use the Gmail tool to send the email.
 
The confirmation email should include:
- Patient name
- Doctor name
- Appointment date and time
- Clinic address: 456 Wellness Blvd, Springfield, IL 62701
- Cancellation policy: Cancel 24 hours in advance to avoid $50 fee
- Clinic phone: (555) 123-4567
 
Keep the email professional and friendly. Use a clear subject line like:
"Appointment Confirmation - HealthFirst Medical Clinic"
"""

def create_confirmation_node(gmail_tools):
    """
    Create a confirmation node function that uses the provided Gmail tools.
    """
    llm_with_tools = llm.bind_tools(gmail_tools)
    def confirmation_node(state: AgentState):
        message = [{"role": "system", "content": SYSTEM_PROMPT},]+state["messages"] 
        response = llm_with_tools.invoke(message)
        return {"messages": [response]}
    
    return confirmation_node, gmail_tools
