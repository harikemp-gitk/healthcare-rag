"""
Memory configuration: This file is used to configure the memory settings for the application. It includes settings for memory limits, cache sizes, and other related parameters. Adjust these settings based on the requirements of your application and the available system resources.
"""

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore


#Short term memory configuration
checkpointer = InMemorySaver()

#Long term memory configuration
store = InMemoryStore()