from dotenv import load_dotenv
import os
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from operator import add as add_messages
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

# === Load env if needed ===
load_dotenv()

# === 1. LLM ===
llm = OllamaLLM(model="llama2", temperature=0)

# === 2. Embedding Model (local HuggingFace) ===
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# === 3. Load PDF ===
pdf_path = "Stock_Market_Performance_2024.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF not found: {pdf_path}")

loader = PyPDFLoader(pdf_path)
pages = loader.load()
print(f"✅ PDF loaded with {len(pages)} pages.")

# === 4. Split PDF Text ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)

# === 5. ChromaDB Setup ===
persist_directory = r"C:\Vaibhav\LangGraph_Book\LangGraphCourse\Agents"
collection_name = "stock_market"
os.makedirs(persist_directory, exist_ok=True)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=persist_directory,
    collection_name=collection_name
)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# === 6. TOOL Definition ===
@tool
def retriever_tool(query: str) -> str:
    """Searches the Stock Market 2024 PDF for information."""
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant information found."
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

tools = [retriever_tool]
tools_dict = {tool_.name: tool_ for tool_ in tools}

# === 7. AgentState ===
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# === 8. System Prompt ===
system_prompt = """
You are an AI assistant answering questions from a document about Stock Market Performance 2024.
Use the tool `retriever_tool` if you need to look something up from the PDF.
Always cite from the document. Begin when the user asks a question.
"""

# === 9. LLM CALL NODE ===
def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm.invoke(messages)

    print(f"\n🤖 LLM Response:\n{response}")

    if response.lower().startswith("action:"):
        try:
            _, tool_name, query = response.split(":", 2)
            ai_msg = AIMessage(
                content="Calling tool...",
                tool_calls=[{"id": "tool1", "name": tool_name.strip(), "args": {"query": query.strip()}}]
            )
        except Exception:
            ai_msg = AIMessage(content=response)
    else:
        ai_msg = AIMessage(content=response)

    return {"messages": [ai_msg]}

# === 10. TOOL EXECUTION NODE ===
def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        tool_name = t["name"]
        query = t["args"].get("query", "")
        if tool_name in tools_dict:
            result = tools_dict[tool_name].invoke(query)
        else:
            result = "Invalid tool name."
        results.append(ToolMessage(tool_call_id=t["id"], name=tool_name, content=result))

    return {"messages": results}

# === 11. ROUTING ===
def should_continue(state: AgentState):
    last = state["messages"][-1]
    return hasattr(last, "tool_calls") and last.tool_calls

# === 12. BUILD GRAPH ===
graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("tool_node", take_action)
graph.set_entry_point("llm")
graph.add_edge("tool_node", "llm")
graph.add_conditional_edges("llm", should_continue, {True: "tool_node", False: END})
app = graph.compile()

# === 13. MAIN LOOP ===
def running_agent():
    print("\n📊 Welcome to the Stock Market RAG Agent (Offline)")
    while True:
        query = input("\nYour Question (type 'exit' to stop): ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting.")
            break
        result = app.invoke({"messages": [HumanMessage(content=query)]})
        print("\n🧠 Final Answer:\n" + result["messages"][-1].content)

if __name__ == "__main__":
    running_agent()
