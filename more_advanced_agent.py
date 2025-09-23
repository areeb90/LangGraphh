# from __future__ import annotations
# from typing import Annotated, Literal, Optional, List
# from typing_extensions import TypedDict
# from pydantic import BaseModel, Field
# import traceback

# from dotenv import load_dotenv

# from langchain.chat_models import init_chat_model
# from langchain_core.messages import (
#     BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
# )
# from langchain_core.tools import tool
# from langgraph.graph import StateGraph, START, END
# from langgraph.graph.message import add_messages

# # ----------------------------
# # Setup
# # ----------------------------
# load_dotenv()
# LLM_MODEL = "openai:gpt-4o-mini"
# SYSTEM_PROMPT = (
#     "You are Aster, a helpful, precise assistant. "
#     "Prefer calling tools when math/time is involved. "
#     "Be concise and avoid hallucinations."
# )
# llm = init_chat_model(LLM_MODEL)

# # ----------------------------
# # Tools
# # ----------------------------
# @tool("calculator", return_direct=False)
# def calculator(expression: str) -> str:
#     """Evaluate a basic arithmetic expression: +,-,*,/,**,%, parentheses."""
#     import re
#     expr = expression.replace("^", "**")
#     if not re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)\*]*", expr):
#         return "Error: Only basic arithmetic allowed."
#     try:
#         val = eval(expr, {"__builtins__": {}}, {})
#         return str(val)
#     except Exception as e:
#         return f"Error: {e}"

# @tool("current_time", return_direct=False)
# def current_time(_: str = "") -> str:
#     """Return current local time ISO string."""
#     from datetime import datetime
#     return datetime.now().isoformat(timespec="seconds")

# TOOLS = [calculator, current_time]
# llm_with_tools = llm.bind_tools(TOOLS)

# # ----------------------------
# # Classifier
# # ----------------------------
# class Intent(BaseModel):
#     route: Literal["tools", "chat", "clarify"] = Field(
#         ..., description="Pick 'tools' for math/date/time; 'chat' for plain Q&A; 'clarify' if ambiguous."
#     )

# def classify_intent(msgs: List[BaseMessage]) -> Intent:
#     classifier = llm.with_structured_output(Intent)
#     prompt: List[BaseMessage] = [
#         SystemMessage(content="Classify the last user message. Prefer 'tools' for math/date/time/computation.")
#     ] + msgs[-6:]  # small window is enough for intent
#     return classifier.invoke(prompt)

# # ----------------------------
# # Rolling summary
# # ----------------------------
# SUMMARY_SYSTEM = (
#     "You produce terse running summaries. "
#     "Capture resolved facts, decisions, tasks, and user preferences. "
#     "Keep under ~120 words. Omit chit-chat."
# )

# def update_summary(summary: str, new_msgs: List[BaseMessage]) -> str:
#     if not new_msgs:
#         return summary
#     summarizer = init_chat_model(LLM_MODEL)
#     text = []
#     for m in new_msgs:
#         if isinstance(m, HumanMessage):
#             text.append(f"User: {m.content}")
#         elif isinstance(m, AIMessage):
#             text.append(f"Assistant: {m.content}")
#     out = summarizer.invoke([
#         SystemMessage(content=SUMMARY_SYSTEM),
#         HumanMessage(content=f"Existing summary:\n{summary or '(none)'}"),
#         HumanMessage(content="New messages to fold in:\n" + "\n".join(text)),
#         HumanMessage(content="Return only the updated concise summary.")
#     ])
#     return out.content.strip()

# # ----------------------------
# # State
# # ----------------------------
# class AgentState(TypedDict):
#     messages: Annotated[List[BaseMessage], add_messages]  # use BaseMessage objects
#     summary: str
#     intent: Optional[str]
#     error: Optional[str]

# # ----------------------------
# # Nodes
# # ----------------------------
# def inject_system_node(state: AgentState) -> AgentState:
#     """Inject fresh system message with current summary; trim history."""
#     summary = state.get("summary", "")
#     summary_suffix = f"\n\nConversation summary: {summary}" if summary else ""
#     system = SystemMessage(content=SYSTEM_PROMPT + summary_suffix)

#     # Remove any prior system messages and keep last 12 non-system
#     non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
#     trimmed = non_system[-12:]

#     return {"messages": [system] + trimmed, "summary": summary, "intent": state.get("intent"), "error": None}

# def classify_node(state: AgentState) -> AgentState:
#     try:
#         route = classify_intent(state["messages"]).route
#         return {"intent": route}
#     except Exception as e:
#         return {"intent": "chat", "error": f"classify_error: {e}"}

# def tools_node(state: AgentState) -> AgentState:
#     """Tool loop: let the model call tools up to 3 times."""
#     msgs: List[BaseMessage] = state["messages"]  # already BaseMessage objects

#     try:
#         for _ in range(3):
#             ai: AIMessage = llm_with_tools.invoke(msgs)
#             tool_calls = getattr(ai, "tool_calls", None)
#             if tool_calls:
#                 # Append the AI tool-call message so the next step sees it
#                 msgs.append(ai)
#                 # Execute each call
#                 name_to_tool = {t.name: t for t in TOOLS}
#                 for call in tool_calls:
#                     name = call["name"]
#                     args = call.get("args") or call.get("arguments") or {}
#                     # Normalize single-arg to str if needed
#                     if isinstance(args, dict) and len(args) == 1:
#                         arg_val = next(iter(args.values()))
#                     else:
#                         arg_val = args if isinstance(args, str) else str(args)
#                     if name not in name_to_tool:
#                         result = f"Error: unknown tool '{name}'"
#                     else:
#                         result = name_to_tool[name].invoke(arg_val)
#                     msgs.append(ToolMessage(tool_call_id=call["id"], name=name, content=str(result)))
#                 # Continue loop to let the model use tool outputs
#                 continue
#             else:
#                 # Normal assistant answer
#                 msgs.append(ai)
#                 return {"messages": [AIMessage(content=ai.content)]}
#         return {"messages": [AIMessage(content="I hit a tool loop limit. Try rephrasing your request.")]}
#     except Exception as e:
#         tb = traceback.format_exc(limit=2)
#         return {"messages": [AIMessage(content=f"Tool error: {e}")], "error": tb}

# def chat_node(state: AgentState) -> AgentState:
#     try:
#         ai: AIMessage = llm.invoke(state["messages"])
#         return {"messages": [AIMessage(content=ai.content)]}
#     except Exception as e:
#         return {"messages": [AIMessage(content=f"Chat error: {e}")], "error": str(e)}

# def clarifier_node(state: AgentState) -> AgentState:
#     ai = llm.invoke([
#         SystemMessage(content="Ask one concise clarifying question. No preamble."),
#         # last user message only
#         *([state["messages"][-1]] if state["messages"] and isinstance(state["messages"][-1], HumanMessage) else [])
#     ])
#     return {"messages": [AIMessage(content=ai.content)]}

# def summarize_node(state: AgentState) -> AgentState:
#     # Fold last few into the rolling summary
#     recent = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-4:]
#     new_summary = update_summary(state.get("summary", ""), recent)
#     return {"summary": new_summary}

# # ----------------------------
# # Graph wiring
# # ----------------------------
# builder = StateGraph(AgentState)
# builder.add_node("inject_system", inject_system_node)
# builder.add_node("classify", classify_node)
# builder.add_node("tools_agent", tools_node)
# builder.add_node("chat_agent", chat_node)
# builder.add_node("clarifier", clarifier_node)
# builder.add_node("summarize", summarize_node)

# builder.add_edge(START, "inject_system")
# builder.add_edge("inject_system", "classify")

# def route(state: AgentState) -> str:
#     intent = state.get("intent", "chat")
#     if intent == "tools":
#         return "tools_agent"
#     if intent == "clarify":
#         return "clarifier"
#     return "chat_agent"

# builder.add_conditional_edges("classify", route, {
#     "tools_agent": "tools_agent",
#     "chat_agent": "chat_agent",
#     "clarifier": "clarifier",
# })

# builder.add_edge("tools_agent", "summarize")
# builder.add_edge("chat_agent", "summarize")
# builder.add_edge("clarifier", "summarize")
# builder.add_edge("summarize", END)

# graph = builder.compile()

# # ----------------------------
# # Minimal CLI
# # ----------------------------
# def run_cli():
#     state: AgentState = {"messages": [], "summary": "", "intent": None, "error": None}
#     print("Aster ready. Type 'exit' to quit.")
#     while True:
#         user = input("\nYou: ").strip()
#         if user.lower() in {"exit", "quit"}:
#             print("Bye!")
#             break
#         state["messages"] = state["messages"] + [HumanMessage(content=user)]
#         state = graph.invoke(state)
#         # Print last assistant message
#         last = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
#         if last:
#             print(f"Aster: {last.content}")
#         if state.get("error"):
#             print(f"[debug] {state['error'][:200]}...")

# if __name__ == "__main__":
#     run_cli()



"""
Aster: a minimal-but-capable LangGraph agent

What you get (kept simple):
- System persona injected every turn
- Intent router: "tools" | "chat" | "clarify"
- Two tools: calculator, current_time
- Short rolling summary memory
- Minimal CLI

Focus: the functions are short, linear, and well-commented.
"""

from __future__ import annotations
from typing import Annotated, Literal, Optional, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

LLM_MODEL = "openai:gpt-4o-mini"  # uses OPENAI_API_KEY from .env
SYSTEM_PROMPT = (
    "You are Aster, a helpful, precise assistant. "
    "Prefer calling tools when math/time is involved. "
    "Be concise and avoid hallucinations."
)

llm = init_chat_model(LLM_MODEL)  # base chat model (no tools bound)

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
    if not re.fullmatch(r"[0-9\.\s\+\-\*\/\%\(\)]*", expr):
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
llm_with_tools = llm.bind_tools(TOOLS)  # same model, but tool-capable

# ──────────────────────────────────────────────────────────────────────────────
# Intent classifier (structured output)
# ──────────────────────────────────────────────────────────────────────────────
class Intent(BaseModel):
    route: Literal["tools", "chat", "clarify"] = Field(
        ..., description="Use 'tools' for math/date/time; 'chat' for plain Q&A; 'clarify' if ambiguous."
    )

def classify_intent(msgs: List[BaseMessage]) -> str:
    """Ask the model to return a tiny schema with a single 'route' string."""
    classifier = llm.with_structured_output(Intent)
    prompt: List[BaseMessage] = [
        SystemMessage(content="Classify the last user message. Prefer 'tools' for math/date/time/computation.")
    ] + msgs[-6:]  # last few are enough
    return classifier.invoke(prompt).route

# ──────────────────────────────────────────────────────────────────────────────
# Rolling summary (tiny memory)
# ──────────────────────────────────────────────────────────────────────────────
SUMMARY_SYSTEM = (
    "You produce terse running summaries. Capture resolved facts, decisions, "
    "tasks, and user preferences. Keep under ~120 words. Omit chit-chat."
)

def update_summary(summary: str, new_msgs: List[BaseMessage]) -> str:
    """
    Summarize the last few user/assistant messages into a short rolling memory.
    """
    if not new_msgs:
        return summary

    # Turn messages into a simple text block for summarization
    lines: List[str] = []
    for m in new_msgs:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            lines.append(f"Assistant: {m.content}")
    text = "\n".join(lines)

    out = llm.invoke([
        SystemMessage(content=SUMMARY_SYSTEM),
        HumanMessage(content=f"Existing summary:\n{summary or '(none)'}"),
        HumanMessage(content=f"New messages to fold in:\n{text}"),
        HumanMessage(content="Return only the updated concise summary.")
    ])
    return out.content.strip()

# ──────────────────────────────────────────────────────────────────────────────
# Graph state
# ──────────────────────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # The reducer `add_messages` APPENDS new messages automatically.
    messages: Annotated[List[BaseMessage], add_messages]
    summary: str
    intent: Optional[str]
    error: Optional[str]

# ──────────────────────────────────────────────────────────────────────────────
# Nodes (kept small and linear)
# ──────────────────────────────────────────────────────────────────────────────
def inject_system_node(state: AgentState) -> AgentState:
    """
    Put a fresh SystemMessage (persona + summary) at the top each turn.
    Also trim to last 12 non-system messages to keep context small.
    """
    summary = state.get("summary", "")
    sys = SystemMessage(
        content=SYSTEM_PROMPT + (f"\n\nConversation summary: {summary}" if summary else "")
    )

    # Remove old system messages; keep last 12 others
    non_system = [m for m in state["messages"] if not isinstance(m, SystemMessage)]
    trimmed = non_system[-12:]

    # Return a *replacement* message list (system + trimmed history)
    return {"messages": [sys] + trimmed, "summary": summary, "intent": state.get("intent"), "error": None}

def classify_node(state: AgentState) -> AgentState:
    """Decide where to go next: tools / chat / clarify."""
    try:
        route = classify_intent(state["messages"])
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
    buffer: List[BaseMessage] = state["messages"]
    new_messages: List[BaseMessage] = []

    for _ in range(3):  # hard stop to avoid infinite loops
        ai: AIMessage = llm_with_tools.invoke(buffer)
        if ai.tool_calls:
            new_messages.append(ai)
            buffer = buffer + [ai]  # include the tool-call message

            # Run each tool and append the ToolMessage result
            name_to_tool = {t.name: t for t in TOOLS}
            for call in ai.tool_calls:
                name = call["name"]
                args = call.get("args") or call.get("arguments") or {}
                # Normalize single-argument dict to a simple string if possible
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
                buffer = buffer + [tm]  # model will see the tool result next loop
            continue

        # No tool calls -> final answer
        new_messages.append(ai)
        break

    # If exhausted loops with no final answer, add a gentle notice
    if not new_messages or not isinstance(new_messages[-1], AIMessage) or new_messages[-1].tool_calls:
        new_messages.append(AIMessage(content="I hit a tool loop limit. Try rephrasing your request."))

    return {"messages": new_messages}

def chat_node(state: AgentState) -> AgentState:
    """Plain assistant reply (no tools)."""
    ai: AIMessage = llm.invoke(state["messages"])
    return {"messages": [ai]}

def clarifier_node(state: AgentState) -> AgentState:
    """Ask one short clarifying question (no preamble)."""
    last_user = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    prompt: List[BaseMessage] = [SystemMessage(content="Ask one concise clarifying question. No preamble.")]
    if last_user:
        prompt.append(last_user)
    ai: AIMessage = llm.invoke(prompt)
    return {"messages": [ai]}

def summarize_node(state: AgentState) -> AgentState:
    """Update the short rolling summary from the last few exchanges."""
    recent = [m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))][-4:]
    return {"summary": update_summary(state.get("summary", ""), recent)}

# ──────────────────────────────────────────────────────────────────────────────
# Graph wiring
# ──────────────────────────────────────────────────────────────────────────────
builder = StateGraph(AgentState)

builder.add_node("inject_system", inject_system_node)
builder.add_node("classify",       classify_node)
builder.add_node("tools_agent",    tools_node)
builder.add_node("chat_agent",     chat_node)
builder.add_node("clarifier",      clarifier_node)
builder.add_node("summarize",      summarize_node)

builder.add_edge(START, "inject_system")
builder.add_edge("inject_system", "classify")

def route(state: AgentState) -> str:
    """Map intent -> node name."""
    intent = state.get("intent") or "chat"
    return {"tools": "tools_agent", "clarify": "clarifier"}.get(intent, "chat_agent")

builder.add_conditional_edges("classify", route, {
    "tools_agent": "tools_agent",
    "chat_agent": "chat_agent",
    "clarifier": "clarifier",
})

# After responding, always summarize then END
builder.add_edge("tools_agent", "summarize")
builder.add_edge("chat_agent", "summarize")
builder.add_edge("clarifier",  "summarize")
builder.add_edge("summarize",  END)

graph = builder.compile()

# ──────────────────────────────────────────────────────────────────────────────
# Minimal CLI
# ──────────────────────────────────────────────────────────────────────────────
def run_cli():
    state: AgentState = {"messages": [], "summary": "", "intent": None, "error": None}
    print("Aster ready. Type 'exit' to quit.")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        state["messages"] = state["messages"] + [HumanMessage(content=user)]
        state = graph.invoke(state)

        # Print last assistant message
        last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
        if last_ai:
            print(f"Aster: {last_ai.content}")

if __name__ == "__main__":
    run_cli()
