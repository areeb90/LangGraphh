"""
Arbii- (Streamlit GUI): a minimal-but-capable LangGraph agent with chat UI

What you get (kept simple):
- Streamlit chat interface with history
- System persona injected every turn
- Intent router: "tools" | "chat" | "clarify"
- Two tools: calculator, current_time
- Short rolling summary memory (persisted in session)
- Model & prompt controls in the sidebar
- Tool-call traces shown inline (expanders)
- Clear conversation button

Run:
  pip install -U streamlit langchain langgraph python-dotenv pydantic typing_extensions
  export OPENAI_API_KEY=...  # or set in a .env file
  streamlit run app.py
"""

from __future__ import annotations
from typing import Annotated, Literal, Optional, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

import os
from dotenv import load_dotenv
import streamlit as st


import os, uuid, textwrap
from datetime import datetime
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


# LangChain / LangGraph
from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ──────────────────────────────────────────────────────────────────────────────
# Page Setup
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Aster • LangGraph Agent", page_icon="✨", layout="wide")
st.title("✨ Aster – Minimal LangGraph Agent (Streamlit GUI)")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    st.caption("Model & prompt can be adjusted at runtime.")

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Model
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

DEFAULT_LLM_MODEL = st.sidebar.text_input(
    "LLM model id", value=os.getenv("ASTER_LLM_MODEL", "openai:gpt-4o-mini")
)
SYSTEM_PROMPT = st.sidebar.text_area(
    "System prompt",
    value=(
        "You are Aster, a helpful, precise assistant. "
        "Prefer calling tools when math/time is involved."
        "Be concise and avoid hallucinations."
    ),
    height=120,
)

# Temperature knob (supported by many chat models)
TEMPERATURE = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

# Create the base and tool-enabled models lazily so they pick up current settings

def make_llm():
    return init_chat_model(DEFAULT_LLM_MODEL, model_kwargs={"temperature": TEMPERATURE})


MEMORY_DIR = os.getenv("ARBII_MEMORY_DIR", ".arbii_memory")
COLLECTION  = os.getenv("ARBII_MEMORY_COLLECTION", "arbii_mem")
EMBED_MODEL = os.getenv("ARBII_EMBED_MODEL", "text-embedding-3-small")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

@st.cache_resource(show_spinner=False)
def get_vectorstore():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=MEMORY_DIR,
    )

def memory_add(text: str, metadata: dict):
    if not text or not text.strip():
        return
    vs = get_vectorstore()
    vs.add_texts([text.strip()], metadatas=[metadata])
    vs.persist()

def memory_search(query: str, k: int = 4):
    if not query or not query.strip():
        return []
    vs = get_vectorstore()
    try:
        return vs.similarity_search(query.strip(), k=k)
    except Exception:
        return []



# ──────────────────────────────────────────────────────────────────────────────
# Tools (very small + safe)
# ──────────────────────────────────────────────────────────────────────────────
@tool("calculator")
def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression: +,-,*,/,**,%, parentheses.
    We strip unsafe characters and use a restricted eval.
    """
    import re

    expr = expression.replace("^", "**")
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/%\(\)]*", expr):
        return "Error: Only basic arithmetic is allowed."
    try:
        val = eval(expr, {"__builtins__": {}}, {})
        return str(val)
    except Exception as e:
        return f"Error: {e}"


@tool("current_time")
def current_time(_: str = "") -> str:
    """Return the current local time (ISO seconds)."""
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")

TOOLS = [calculator, current_time]

# ──────────────────────────────────────────────────────────────────────────────
# Intent classifier (structured output)
# ──────────────────────────────────────────────────────────────────────────────
class Intent(BaseModel):
    route: Literal["tools", "chat", "clarify"] = Field(
        ..., description="Use 'tools' for math/date/time; 'chat' for plain Q&A; 'clarify' if ambiguous."
    )


def classify_intent(llm, msgs: List[BaseMessage]) -> str:
    """Ask the model to return a tiny schema with a single 'route' string."""
    classifier = llm.with_structured_output(Intent)
    prompt: List[BaseMessage] = [
        SystemMessage(
            content="Classify the last user message. Prefer 'tools' for math/date/time/computation."
        )
    ] + msgs[-6:]
    return classifier.invoke(prompt).route


# ──────────────────────────────────────────────────────────────────────────────
# Rolling summary (tiny memory)
# ──────────────────────────────────────────────────────────────────────────────
SUMMARY_SYSTEM = (
    "You produce terse running summaries. Capture resolved facts, decisions, "
    "tasks, and user preferences. Keep under ~120 words. Omit chit-chat."
)


def update_summary(llm, summary: str, new_msgs: List[BaseMessage]) -> str:
    """Summarize the last few user/assistant messages into a short rolling memory."""
    if not new_msgs:
        return summary

    lines: List[str] = []
    for m in new_msgs:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    text = "\n".join(lines)

    out = make_llm().invoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM),
            HumanMessage(content=f"Existing summary:\n{summary or '(none)'}"),
            HumanMessage(content=f"New messages to fold in:\n{text}"),
            HumanMessage(content="Return only the updated concise summary."),
        ]
    )
    return out.content.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Graph state
# ──────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    intent: Optional[str]
    error: Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# Nodes (kept small and linear)
# ──────────────────────────────────────────────────────────────────────────────

def inject_system_node(state: AgentState) -> AgentState:
    """Put a fresh SystemMessage (persona + summary + retrieved memory) each turn."""
    summary = state.get("summary", "")

    # Query memories using the most recent user message
    last_user = next((m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)), None)
    retrieved_blurbs = []
    if last_user:
        docs = memory_search(last_user.content, k=4)
        for d in docs:
            snippet = d.page_content or ""
            if snippet:
                # keep it short to reduce prompt bloat
                retrieved_blurbs.append("• " + textwrap.shorten(snippet.replace("\n"," "), width=220))

    memory_block = ("\n\nRelevant memories:\n" + "\n".join(retrieved_blurbs)) if retrieved_blurbs else ""

    sys = SystemMessage(
        content=SYSTEM_PROMPT
        + (f"\n\nConversation summary: {summary}" if summary else "")
        + memory_block
    )

    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    trimmed = non_system[-12:]

    return {"messages": [sys] + trimmed, "summary": summary, "intent": state.get("intent"), "error": None}



def classify_node(state: AgentState) -> AgentState:
    try:
        route = classify_intent(make_llm(), state["messages"])
        return {"intent": route}
    except Exception as e:
        return {"intent": "chat", "error": f"classify_error: {e}"}


def tools_node(state: AgentState) -> AgentState:
    """
    Let the model call tools up to 3 times, returning all new messages:
    - AIMessage with tool_calls
    - ToolMessage with each result
    - Final AIMessage answer
    """
    llm = make_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    buffer: List[BaseMessage] = state["messages"]
    new_messages: List[BaseMessage] = []

    for _ in range(3):
        ai: AIMessage = llm_with_tools.invoke(buffer)
        if getattr(ai, "tool_calls", None):
            new_messages.append(ai)
            buffer = buffer + [ai]

            name_to_tool = {t.name: t for t in TOOLS}
            for call in ai.tool_calls:
                name = call["name"]
                args = call.get("args") or call.get("arguments") or {}
                if isinstance(args, dict) and len(args) == 1:
                    arg_val = next(iter(args.values()))
                else:
                    arg_val = args if isinstance(args, str) else str(args)

                if name not in name_to_tool:
                    result = f"Error: unknown tool '{name}'"
                else:
                    result = name_to_tool[name].invoke(arg_val)

                tm = ToolMessage(tool_call_id=call["id"], name=name, content=str(result))
                new_messages.append(tm)
                buffer = buffer + [tm]
            continue

        new_messages.append(ai)
        break

    if (
        not new_messages
        or not isinstance(new_messages[-1], AIMessage)
        or getattr(new_messages[-1], "tool_calls", None)
    ):
        new_messages.append(
            AIMessage(content="I hit a tool loop limit. Try rephrasing your request.")
        )

    return {"messages": new_messages}


def chat_node(state: AgentState) -> AgentState:
    ai: AIMessage = make_llm().invoke(state["messages"])
    return {"messages": [ai]}


def clarifier_node(state: AgentState) -> AgentState:
    last_user = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    prompt: List[BaseMessage] = [
        SystemMessage(content="Ask one concise clarifying question. No preamble."),
    ]
    if last_user:
        prompt.append(last_user)
    ai: AIMessage = make_llm().invoke(prompt)
    return {"messages": [ai]}


def summarize_node(state: AgentState) -> AgentState:
    """Update rolling summary, then store it to vector memory."""
    recent = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-4:]
    new_summary = update_summary(make_llm(), state.get("summary", ""), recent)

    if new_summary and new_summary.strip():
        memory_add(
            new_summary,
            {
                "type": "summary",
                "session_id": st.session_state.session_id,
                "ts": datetime.now().isoformat(timespec="seconds"),
            },
        )
    return {"summary": new_summary}



# ──────────────────────────────────────────────────────────────────────────────
# Graph wiring
# ──────────────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("inject_system", inject_system_node)
builder.add_node("classify", classify_node)
builder.add_node("tools_agent", tools_node)
builder.add_node("chat_agent", chat_node)
builder.add_node("clarifier", clarifier_node)
builder.add_node("summarize", summarize_node)

builder.add_edge(START, "inject_system")
builder.add_edge("inject_system", "classify")


def route(state: AgentState) -> str:
    intent = state.get("intent") or "chat"
    return {"tools": "tools_agent", "clarify": "clarifier"}.get(intent, "chat_agent")


builder.add_conditional_edges(
    "classify",
    route,
    {
        "tools_agent": "tools_agent",
        "chat_agent": "chat_agent",
        "clarifier": "clarifier",
    },
)

builder.add_edge("tools_agent", "summarize")
builder.add_edge("chat_agent", "summarize")
builder.add_edge("clarifier", "summarize")
builder.add_edge("summarize", END)

graph = builder.compile()

# ──────────────────────────────────────────────────────────────────────────────
# Streamlit Session State
# ──────────────────────────────────────────────────────────────────────────────
if "agent_state" not in st.session_state:
    st.session_state.agent_state: AgentState = {
        "messages": [],
        "summary": "",
        "intent": None,
        "error": None,
    }

if "chat_log" not in st.session_state:
    # Stores renderable dicts for Streamlit display
    st.session_state.chat_log: List[dict] = []

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: Memory & Controls
# ──────────────────────────────────────────────────────────────────────────────
# with st.sidebar:
#     st.subheader("Memory (rolling summary)")
#     st.write(st.session_state.agent_state.get("summary", "(empty)"))

#     st.divider()
#     if st.button("🗑️ Clear conversation", use_container_width=True):
#         st.session_state.agent_state = {
#             "messages": [],
#             "summary": "",
#             "intent": None,
#             "error": None,
#         }
#         st.session_state.chat_log = []
#         st.rerun()

st.subheader("Persistent Memory (Chroma)")
vs = get_vectorstore()
try:
    count = vs._collection.count()  # type: ignore[attr-defined]
except Exception:
    count = None
st.caption(f"Collection: {COLLECTION} • Dir: {MEMORY_DIR} • Items: {count if count is not None else '?'}")

q = st.text_input("🔎 Search memories", key="mem_query")
if q:
    docs = memory_search(q, k=5) or []
    for i, d in enumerate(docs, 1):
        with st.expander(f"Result {i}"):
            st.write(d.page_content)
            st.code(d.metadata, language="json")

if st.button("🧹 Wipe persistent memory", use_container_width=True):
    import shutil
    shutil.rmtree(MEMORY_DIR, ignore_errors=True)
    get_vectorstore()  # re-init
    st.success("Persistent memory wiped.")
    st.rerun()

# to check all the persisted memories stored in the vector store chromadb

# docs = vs.get()
# st.subheader("🔎 Persistent Memory Dump")
# for doc, meta in zip(docs["documents"], docs["metadatas"]):
#     st.write("•", doc)
#     st.caption(meta)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: render a batch of new messages from the graph
# ──────────────────────────────────────────────────────────────────────────────

def render_new_messages(new_msgs: List[BaseMessage]):
    """Append tool traces & AI/user messages to the visible chat_log."""
    for m in new_msgs:
        if isinstance(m, HumanMessage):
            st.session_state.chat_log.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            # If this AIMessage includes tool calls, show them in an expander
            content = m.content or ""
            tool_calls = getattr(m, "tool_calls", None) or []
            st.session_state.chat_log.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
        elif isinstance(m, ToolMessage):
            # Attach tool results as a separate assistant message with a tag
            st.session_state.chat_log.append(
                {
                    "role": "tool",
                    "name": m.name,
                    "content": m.content,
                    "tool_call_id": getattr(m, "tool_call_id", None),
                }
            )
        elif isinstance(m, SystemMessage):
            # Don't render system messages in the chat; skip
            continue
        else:
            st.session_state.chat_log.append(
                {"role": "assistant", "content": str(getattr(m, "content", m))}
            )


# ──────────────────────────────────────────────────────────────────────────────
# Display existing chat log
# ──────────────────────────────────────────────────────────────────────────────
for entry in st.session_state.chat_log:
    role = entry.get("role")
    if role == "user":
        with st.chat_message("user"):
            st.write(entry.get("content", ""))
    elif role == "assistant":
        with st.chat_message("assistant"):
            if entry.get("content"):
                st.write(entry["content"])
            # Tool call trace expander
            tool_calls = entry.get("tool_calls") or []
            for call in tool_calls:
                with st.expander(f"🔧 Tool call: {call.get('name')} (id: {call.get('id')})"):
                    st.json({k: v for k, v in call.items() if k != "id"} | {"id": call.get("id")})
    elif role == "tool":
        with st.chat_message("assistant"):
            st.info(f"Tool `{entry.get('name')}` result:")
            st.code(entry.get("content", ""), language="text")

# ──────────────────────────────────────────────────────────────────────────────
# Chat Input
# ──────────────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask Aster… (math/time will use tools)")

if user_input:
    # 1) Add user message to graph state
    st.session_state.agent_state["messages"] = st.session_state.agent_state["messages"] + [
        HumanMessage(content=user_input)
    ]

    # Also render immediately
    render_new_messages([HumanMessage(content=user_input)])

    # 2) Run the graph turn
    new_state = graph.invoke(st.session_state.agent_state)

    # 3) Diff out only the new messages added by nodes
    #    Because our nodes return deltas, the compiled graph merges them.
    #    We'll compute a naive diff based on lengths.
    old_len = len(st.session_state.agent_state["messages"])
    st.session_state.agent_state = new_state
    all_msgs = st.session_state.agent_state["messages"]
    new_msgs = all_msgs[old_len:]

    # 4) Render new messages (tool traces + ai)
    render_new_messages(new_msgs)

    # 5) Update sidebar summary (already in session state)
    st.rerun()  # Refresh to display the new messages neatly

# Footer / help
st.caption(
    "Tip: Try 'what time is it?', '12*(8+2)', or a normal chat question."
)
