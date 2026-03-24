"""
This module defines the FAQGraph class, which constructs a knowledge graph from a collection of FAQ documents.
 The graph is built using LangChain's Graph class and is designed to facilitate efficient retrieval of information based on user queries. 
 The FAQGraph class includes methods for loading documents, creating nodes and edges, and querying the graph for relevant information.

"""

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from agents.faq_agent import faq_node, tools
from config.memory import checkpointer, store

builder = StateGraph(MessagesState)

builder.add_node("faq_agent", faq_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "faq_agent")
builder.add_conditional_edges("faq_agent", tools_condition)
builder.add_edge("tools", "faq_agent")

faq_graph = builder.compile(checkpointer=checkpointer, store=store)
