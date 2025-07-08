from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import OllamaLLM
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# Global document storage
document_content = ""

# State structure for LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# === TOOLS ===

@tool
def update(content: str) -> str:
    """Update the current document with new content."""
    global document_content
    document_content = content
    return f"✅ Document updated:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the document to a file with the given filename."""
    global document_content
    if not filename.endswith('.txt'):
        filename += ".txt"
    try:
        with open(filename, "w") as file:
            file.write(document_content)
        return f"💾 Saved to '{filename}'"
    except Exception as e:
        return f"❌ Error saving file: {str(e)}"

tools = [update, save]

# === LLM ===
model = OllamaLLM(model="llama2")

# === MAIN AGENT NODE ===
def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
You are Drafter, an AI assistant for editing documents.

Instructions:
- Use the 'update' tool if the user wants to change or add content.
- Use the 'save' tool if the user wants to store the document.
- Always respond using tools only.
- Current document content:
{document_content}
""")

    if not state["messages"]:
        user_input = "I'm ready to help you edit a document. What would you like to do?"
    else:
        user_input = input("\n👤 What would you like to do with the document? ")

    user_msg = HumanMessage(content=user_input)
    full_messages = [system_prompt] + list(state["messages"]) + [user_msg]

    # Invoke Ollama (returns a string, not AIMessage)
    response_text = model.invoke(full_messages)
    print(f"\n🤖 AI responded: {response_text}")

    # Simulate tool-calling behavior
    response_text = response_text.strip()
    if response_text.lower().startswith("action: update"):
        content = response_text.split(":", 2)[-1].strip()
        tool_call = AIMessage(
            content="Calling update tool...",
            tool_calls=[{"name": "update", "args": {"content": content}}]
        )
    elif response_text.lower().startswith("action: save"):
        filename = response_text.split(":", 2)[-1].strip()
        tool_call = AIMessage(
            content="Calling save tool...",
            tool_calls=[{"name": "save", "args": {"filename": filename}}]
        )
    else:
        tool_call = AIMessage(content=response_text)

    return {"messages": list(state["messages"]) + [user_msg, tool_call]}

# === ENDING CONDITION ===
def should_continue(state: AgentState) -> str:
    for msg in reversed(state["messages"]):
        if isinstance(msg, ToolMessage):
            if "saved" in msg.content.lower():
                return "end"
    return "continue"

# === DISPLAY TOOL RESPONSES ===
def print_messages(messages):
    for msg in messages[-2:]:
        if isinstance(msg, ToolMessage):
            print(f"\n🛠️ TOOL RESPONSE: {msg.content}")

# === LANGGRAPH FLOW ===
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_edge("agent", "tools")
graph.add_conditional_edges("tools", should_continue, {"continue": "agent", "end": END})
app = graph.compile()

# === RUNNING LOOP ===
def run_document_agent():
    print("\n📄 DRAFTER STARTED")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n✅ DRAFTER FINISHED")

if __name__ == "__main__":
    run_document_agent()
