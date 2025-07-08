from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Define the agent's state
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define tools
@tool
def add(a: int, b: int):
    """This is an addition function that adds 2 numbers together"""
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtraction function"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplication function"""
    return a * b

tools = [add, subtract, multiply]

# Define the LLM (Ollama)
model = OllamaLLM(model="llama2")

# Node: LLM Call
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant, please answer my query to the best of your ability.")
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

# Conditional logic to continue or end
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
        return "continue"
    else:
        return "end"

# Construct the graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()

# Print output helper
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# User input wrapped as HumanMessage
inputs = {
    "messages": [
        HumanMessage(content="Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")
    ]
}

# Run the app
print_stream(app.stream(inputs, stream_mode="values"))
