from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class AgentState(TypedDict):
    messages: List[HumanMessage]

# ✅ Use a local model via Ollama
llm = OllamaLLM(model="llama2")  # or "mistral", "gemma", etc.

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])  # Returns a plain string
    print(f"\nAI: {response}")                # No .content needed
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("Enter: ")
while user_input.lower() != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")