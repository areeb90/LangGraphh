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


