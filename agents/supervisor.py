"""
Supervisor agent : Oversees the entire system, monitoring the interactions between agents and tools, ensuring that the workflow is progressing smoothly, and intervening when necessary to resolve issues or optimize performance. The Supervisor agent is responsible for maintaining the overall coherence of the system, coordinating the activities of different agents, and providing guidance or feedback to ensure that the system's goals are achieved effectively and efficiently.
"""

from config.models import llm
from agents.state import AgentState
from pydantic import Field,BaseModel


SYSTEM_PROMPT = """
You are a supervisor routing requests at HealthFirst Medical Clinic.
 
Analyze the user's message and route to the appropriate agent:
 
- **faq_agent**: Questions about clinic hours, location, doctors, policies, services, insurance, what to bring, parking, telehealth, lab work.
- **booking_agent**: Requests to book, schedule, or make an appointment. Also if the user is in the middle of providing booking details (name, email, doctor, date, time, reason).
- **FINISH**: The user is saying goodbye, thanks, or the conversation is complete.
 
Rules:
- If booking is in progress (booking_complete is False and user seems to be providing details), route to booking_agent.
- If unsure, route to faq_agent.
- Only route to FINISH if the user clearly wants to end the conversation.

"""

class RouteDecision(BaseModel):
    next_agent: str = Field(
        description="he next agent to handle the request."
        "Must be one of: 'faq_agent', 'booking_agent', 'FINISH'"
    ) 

    reasoning: str = Field(
        description="The reasoning behind the routing decision."
    )

router_llm = llm.with_structured_output(RouteDecision)

def supervisor_node(state: AgentState):
    """
    Route the user request to the appropriate agent based on the conversation context and the user's needs. The supervisor agent analyzes the current state of the conversation, including the user's input, previous interactions, and any relevant information stored in the system. Based on this analysis, the supervisor agent decides which agent is best suited to handle the next step in the conversation, ensuring that the user's needs are addressed effectively and efficiently.
    """

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"]
    if state.get("booking_complete") == False and state.get("booking_details"):
        messages.append({"role": "system", 
                         "content": "Note: A booking is currently in progress. Route to booking_agent to continue collecting the details and finalize the appointment. Unless user explicitly days they want to switch topics."})
        
    decision = router_llm.invoke(messages)
    return {"next_agent": decision.next_agent}     