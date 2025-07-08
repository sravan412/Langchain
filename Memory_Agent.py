import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import OllamaLLM

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables (if any)
load_dotenv()

# Define the state structure
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Initialize the LLM
llm = OllamaLLM(model="llama2")

# Processing node for the LangGraph
def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    # Call the LLM — returns a string
    response_text = llm.invoke(state["messages"])

    # Wrap it in an AIMessage
    ai_message = AIMessage(content=response_text)
    state["messages"].append(ai_message)

    print(f"\nAI: {response_text}")
    print("CURRENT STATE: ", state["messages"])

    return state

# Create the LangGraph
graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# Chat loop
conversation_history = []

user_input = input("Enter: ")
while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")

# Save the conversation to a log file
with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")
