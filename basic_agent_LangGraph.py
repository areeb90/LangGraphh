# # # simple_aster.py
# # import os
# # import uuid
# # import textwrap
# # from datetime import datetime
# # from difflib import SequenceMatcher
# # from dotenv import load_dotenv

# # load_dotenv()  # take environment variables from .env file

# # import streamlit as st

# # # Optional: real LLM via OpenAI. If you don't have the key, app falls back to a simple echo model.
# # try:
# #     import openai
# # except Exception:
# #     openai = None

# # st.set_page_config(page_title="Simple Aster", layout="wide")
# # st.title("Simple Aster — Minimal RAG + Tools demo")

# # # --------------------------
# # # Simple config & session
# # # --------------------------
# # if "session_id" not in st.session_state:
# #     st.session_state.session_id = str(uuid.uuid4())
# # if "chat" not in st.session_state:
# #     st.session_state.chat = []  # list of (role, text)
# # if "summaries" not in st.session_state:
# #     st.session_state.summaries = []  # list of {"text":..., "ts":...}

# # # --------------------------
# # # Tools (simple)
# # # --------------------------
# # def tool_calculator(expr: str) -> str:
# #     import re
# #     expr = expr.replace("^", "**")
# #     if not re.fullmatch(r"[0-9\.\s\+\-\*\/%\(\)]*", expr):
# #         return "Error: Only basic arithmetic allowed."
# #     try:
# #         return str(eval(expr, {"__builtins__": {}}, {}))
# #     except Exception as e:
# #         return f"Error: {e}"

# # def tool_time(_: str = "") -> str:
# #     return datetime.now().isoformat(timespec="seconds")

# # TOOLS = {"calculator": tool_calculator, "current_time": tool_time}

# # # --------------------------
# # # Tiny LLM wrapper
# # # --------------------------
# # OPENAI_KEY = os.getenv("OPENAI_API_KEY")


# # def call_llm(prompt: str) -> str:
# #     # Show visible debug in the app (remove or lower verbosity afterwards)
# #     st.write("DEBUG: OPENAI_KEY present?", bool(OPENAI_KEY))
# #     st.write("DEBUG: openai package present?", openai is not None)
# #     st.write("DEBUG: prompt preview:", prompt[:400].replace("\n", "\\n"))

# #     if OPENAI_KEY and openai:
# #         try:
# #             openai.api_key = OPENAI_KEY
# #             resp = openai.ChatCompletion.create(
# #                 model="gpt-3.5-turbo",   # change if you want another model
# #                 messages=[
# #                     {"role": "system", "content": "You are Aster. Be concise but helpful."},
# #                     {"role": "user", "content": prompt},
# #                 ],
# #                 temperature=0.5,
# #                 max_tokens=800,
# #                 n=1,
# #             )
# #             # robust extraction
# #             try:
# #                 out = resp.choices[0].message.content
# #             except Exception:
# #                 out = resp["choices"][0]["message"]["content"]
# #             out = out.strip() if isinstance(out, str) else str(out)
# #             st.write("DEBUG: LLM out preview:", out[:400].replace("\n","\\n"))
# #             return out
# #         except Exception as e:
# #             st.write("DEBUG: OpenAI API call failed:", repr(e))
# #             return f"[LLM ERROR] {e}"
# #     # fallback
# #     st.write("DEBUG: Using local fallback LLM (no OPENAI key or openai package).")
# #     pl = prompt.lower()
# #     if "calculate:" in pl:
# #         arg = prompt.split("calculate:",1)[1].strip().splitlines()[0]
# #         return f"[TOOL_CALL calculator] {arg}"
# #     if "what time" in pl or "time is it" in pl:
# #         return "[TOOL_CALL current_time] "
# #     return "I hear you. Summary: " + (prompt[:120].strip().replace("\n", " "))

# # # --------------------------
# # # Simple retrieval (fuzzy)
# # # --------------------------
# # def retrieve_relevant(query: str, k=3):
# #     # Use SequenceMatcher ratio to find similar summaries as a naive "RAG"
# #     candidates = []
# #     for s in st.session_state.summaries:
# #         score = SequenceMatcher(None, query.lower(), s["text"].lower()).ratio()
# #         candidates.append((score, s))
# #     candidates.sort(reverse=True, key=lambda x: x[0])
# #     return [c for sc,c in candidates[:k] if sc > 0.1]

# # # --------------------------
# # # Rolling summary update
# # # --------------------------
# # def update_summary(new_messages: list):
# #     """
# #     Very small summarizer: join last few user/assistant messages and
# #     produce a one-line 'summary' using the LLM (or naive fallback).
# #     """
# #     text = "\n".join([f"{r}: {t}" for r,t in new_messages[-6:]])
# #     prompt = f"Produce a one-line concise summary (<=40 words) of:\n{text}"
# #     out = call_llm(prompt)
# #     summary_text = out.splitlines()[0][:500]
# #     summary = {"text": summary_text, "ts": datetime.now().isoformat(timespec="seconds")}
# #     st.session_state.summaries.append(summary)
# #     return summary_text

# # # --------------------------
# # # Intent classifier (simple rule-based)
# # # --------------------------
# # def classify_intent(user_message: str) -> str:
# #     lw = user_message.lower()
# #     if any(tok in lw for tok in ["calculate", "+", "-", "*", "/", "what is", "times", "^"]):
# #         return "tools"
# #     if any(tok in lw for tok in ["clarify", "which", "what do you mean", "did you mean"]):
# #         return "clarify"
# #     return "chat"

# # # --------------------------
# # # UI: sidebar & memory view
# # # --------------------------
# # with st.sidebar:
# #     st.header("Memory (rolling summaries)")
# #     for i,s in enumerate(reversed(st.session_state.summaries[-10:]), 1):
# #         st.write(f"{i}. {s['text']}  — {s['ts']}")
# #     if st.button("Clear memory"):
# #         st.session_state.summaries = []
# #     st.divider()
# #     st.caption("This demo uses a naive local memory (no vector DB).")

# # # --------------------------
# # # Chat display
# # # --------------------------
# # for role, text in st.session_state.chat:
# #     if role == "user":
# #         with st.chat_message("user"):
# #             st.write(text)
# #     else:
# #         with st.chat_message("assistant"):
# #             st.write(text)

# # # --------------------------
# # # Input handling
# # # --------------------------
# # user_input = st.chat_input("Ask Aster (try 'what time is it', or 'calculate: 12*(8+2)')")

# # if user_input:
# #     # 1) append user
# #     st.session_state.chat.append(("user", user_input))
# #     # 2) retrieve relevant summaries (RAG-style)
# #     retrieved = retrieve_relevant(user_input, k=3)
# #     retrieved_block = ""
# #     if retrieved:
# #         retrieved_block = "\n\nRelevant memories:\n" + "\n".join("• " + textwrap.shorten(r["text"], width=120) for r in retrieved)
# #     # 3) build prompt (system + retrieved + conversation excerpt)
# #     convo_excerpt = "\n".join([f"{r}: {t}" for r,t in st.session_state.chat[-6:]])
# #     prompt = f"You are Aster. Keep concise.\n\nConversation excerpt:\n{convo_excerpt}{retrieved_block}\n\nUser asks:\n{user_input}"
# #     # 4) call LLM (or fallback)
# #     raw_reply = call_llm(prompt)
# #     # 5) handle TOOL_CALL markers (simple protocol)
# #     if raw_reply.startswith("[TOOL_CALL"):
# #         # extract name and arg if present
# #         parts = raw_reply.strip("[]").split()
# #         tool_name = parts[1]
# #         # some fallbacks parse arg after the tag
# #         arg = raw_reply.split("]",1)[-1].strip()
# #         if tool_name in TOOLS:
# #             tool_res = TOOLS[tool_name](arg)
# #             assistant_text = f"Tool `{tool_name}` result: {tool_res}"
# #         else:
# #             assistant_text = f"Unknown tool requested: {tool_name}"
# #     else:
# #         assistant_text = raw_reply

# #     # 6) append assistant
# #     st.session_state.chat.append(("assistant", assistant_text))

# #     # 7) update a rolling summary from the latest few messages
# #     summary = update_summary(st.session_state.chat[-6:])

# #     # 8) re-run to display new messages
# #     st.rerun()

    




































# # simple_agent.py
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import ToolNode, tools_condition
# from langchain_openai import ChatOpenAI
# from langchain_core.messages import HumanMessage
# from langchain_core.tools import tool

# # 1) Define a simple tool
# @tool
# def add_numbers(a: int, b: int) -> int:
#     """Add two numbers."""
#     return a + b

# # 2) Define the model (replace with your model, e.g. "gpt-4o-mini")
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# # 3) Wrap tools in a ToolNode
# tools = [add_numbers]
# tool_node = ToolNode(tools)

# # 4) Build the graph
# workflow = StateGraph(HumanMessage)  # State: conversation
# workflow.add_node("llm", llm)        # Node: LLM
# workflow.add_node("tools", tool_node) # Node: Tools
# workflow.set_entry_point("llm")      # Start from the LLM
# workflow.add_conditional_edges("llm", tools_condition)  # If tool needed, go to tools
# workflow.add_edge("tools", "llm")    # After tool, return to LLM
# workflow.add_edge("llm", END)        # Otherwise, finish
# graph = workflow.compile()

# # 5) Run a test
# if __name__ == "__main__":
#     response = graph.invoke(HumanMessage(content="Please add 7 and 11"))
#     print(response.content)





























# basic_agent_LangGraph.py
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool

import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  # take environment variables from .env file

st.set_page_config(page_title="Basic Agent with LangGraph", layout="wide")
st.title("Basic Agent with LangGraph")

# 1) Define a simple tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

# 2) Define the model (replace with your model, e.g. "gpt-4o-mini")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=OPENAI_KEY)

# 3) Wrap tools in a ToolNode
tools = [add_numbers, multiply_numbers]
tool_node = ToolNode(tools)

# 4) Build the graph
workflow = StateGraph(HumanMessage)  # State: conversation
workflow.add_node("llm", llm)        # Node: LLM
workflow.add_node("tools", tool_node) # Node: Tools
workflow.set_entry_point("llm")      # Start from the LLM
workflow.add_conditional_edges("llm", tools_condition)  # If tool needed, go to tools
workflow.add_edge("tools", "llm")    # After tool, return to LLM
workflow.add_edge("llm", END)        # Otherwise, finish
graph = workflow.compile()

# 5) Run a test 
if __name__ == "__main__":
    user_input = st.text_input("Enter your query (e.g., 'Please add 7 and 11'):")
    if user_input:
        # FIX: pass as list
        response = graph.invoke([HumanMessage(content=user_input)])
        st.write("Response:", response[0].content)  # take first message
        st.write("DEBUG: Full response object:", response)


        st.write("Try queries like 'Please multiply 6 and 9' or 'What is 15 plus 27?'")
        st.write("This demo uses LangGraph to create a basic agent that can use tools.")
        st.write("Make sure your OPENAI_API_KEY is set in the environment variables.")
        st.write("Replace the model name in the code to experiment with different LLMs.")
        st.write("The agent can decide when to use the tools based on the input.")
        st.write("This is a minimal example; you can expand it with more tools and complex logic.")
        st.write("Check the console for any errors or debug information.")
        st.write("Enjoy experimenting with LangGraph and LangChain!")
        st.write("For more information, visit the LangGraph and LangChain documentation.")
        st.write("This is a simple demo app built with Streamlit.")
        st.write("Feel free to modify the code and add more functionality.")
        st.write("Thank you for trying out this basic agent example!")
        st.write("Have fun building more advanced agents with LangGraph!")
        st.write("Remember to handle any exceptions or errors in a production app.")
        st.write("This app is for educational purposes and may not be suitable for production use.")
        st.write("You can add more tools by defining new functions and decorating them with @tool.")
        st.write("The agent's behavior can be customized by changing the prompt templates.")
        st.write("You can also integrate with other APIs or services as needed.")
        st.write("Happy coding!")
        st.write("This concludes the basic agent demo with LangGraph.")
        st.write("Goodbye!")
        st.write("This is the end of the demo.")
        st.write("Feel free to close the app or refresh to start over.")
        st.write("Thank you for using the basic agent demo!")
        st.write("This is a simple demonstration of LangGraph's capabilities.")
        st.write("You can build more complex agents by adding more nodes and edges.")
        st.write("Experiment with different tools and see how the agent responds.")
        st.write("This app showcases the integration of LangGraph with LangChain.")
        st.write("You can deploy this app on a server or cloud platform for wider access.")
        st.write("This demo is open-source; feel free to share and contribute.")
        st.write("Check out the LangGraph GitHub repository for more examples and documentation.")
        st.write("Stay tuned for more updates and features in LangGraph!")
        st.write("End of the demo. Have a great day!")


