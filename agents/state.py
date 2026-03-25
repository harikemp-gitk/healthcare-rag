"""
Shared agent state across multiple agents. This is useful for storing information that is relevant to multiple agents, such as user preferences, conversation history, or shared context.
"""

from typing import Optional
from typing_extensions import TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class BookingDetails(TypedDict,total=False):
    patient_name: str
    doctor_name: str
    appointment_date: str
    appointment_time: str
    reason : str
    patient_email: str

class AgentState(TypedDict):
    booking_details: Optional[BookingDetails]
    messages: Annotated[list[BaseMessage], add_messages]  # This will automatically add messages to the state when returned by an agent
    next_agent: str
    booking_complete: bool
    confirmation_sent: bool

