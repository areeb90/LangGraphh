# test_arbii.py
import types
import unittest
from datetime import datetime, timedelta

import advanced_agent_v2 as arbii  # <-- your Streamlit app module (rename if file name differs)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage

# ---------------------------
# Fake LLM for deterministic tests
# ----------------------------
class FakeStructuredOut:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, prompt):
        # Last human content drives routing
        last_user = None
        for m in reversed(prompt):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break

        class IntentShim:
            def __init__(self, route):
                self.route = route

        if not last_user:
            return IntentShim("chat")

        msg = last_user.lower()
        # crude classifier: prefer tools for math/time
        if any(op in msg for op in ["+", "-", "*", "/", "%", "^", "(", ")"]) or any(ch.isdigit() for ch in msg):
            return IntentShim("tools")
        if "time" in msg or "current time" in msg:
            return IntentShim("tools")
        if "book" in msg and "flight" in msg:
            return IntentShim("clarify")
        return IntentShim("chat")


class FakeLLM:
    """
    A minimal stand-in that supports:
      - .with_structured_output(...)
      - .bind_tools([...])
      - .invoke([...])
    Behavior:
      * If tools bound and last user asks math => emit calculator tool_call
      * If tools bound and last user asks time => emit current_time tool_call
      * If a ToolMessage is already in the buffer => emit final AIMessage summarizing tool results
      * For clarifier node => emits one short question
      * For chat => emits a fixed reply
    """
    def __init__(self, tools=None):
        self.tools = {t.name: t for t in (tools or [])}

    def with_structured_output(self, _schema):
        return FakeStructuredOut(self)

    def bind_tools(self, tools):
        return FakeLLM(tools=tools)

    def _latest_tool_result(self, buffer):
        for m in reversed(buffer):
            if isinstance(m, ToolMessage):
                return m.name, m.content
        return None, None

    def invoke(self, buffer):
        # Clarifier prompt detection
        if buffer and isinstance(buffer[0], SystemMessage) and "Ask one concise clarifying question" in buffer[0].content:
            return AIMessage(content="Which dates and destination do you want?")

        # If we've already called a tool, produce a final answer using its result
        name, content = self._latest_tool_result(buffer)
        if name:
            return AIMessage(content=f"[final] Used `{name}` → {content}")

        # Get last user message
        last_user = None
        for m in reversed(buffer):
            if isinstance(m, HumanMessage):
                last_user = m.content
                break

        # If tools are bound, create tool calls when appropriate
        if self.tools and last_user:
            q = last_user.lower().strip()
            # very naive math detector
            if any(op in q for op in ["+", "-", "*", "/", "%", "^", "(", ")"]) or any(ch.isdigit() for ch in q):
                # Emit a calculator tool call with the raw expression (strip label if present)
                expr = last_user.replace("calc:", "").strip()
                return AIMessage(content="", tool_calls=[{
                    "name": "calculator",
                    "id": "tc_calculator_1",
                    "arguments": {"expression": expr}
                }])

            if "time" in q:
                return AIMessage(content="", tool_calls=[{
                    "name": "current_time",
                    "id": "tc_time_1",
                    "arguments": {}
                }])

        # Otherwise: basic chat
        return AIMessage(content=f"(fake-chat) I heard: {last_user or '[no user message]'}")


# ----------------------------
# TestCase
# ----------------------------
class TestArbiiAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Monkeypatch the app's make_llm to return our FakeLLM
        arbii.make_llm = lambda: FakeLLM()
        # Rebuild the compiled graph if needed (usually not necessary)
        # arbii.graph = arbii.graph  # placeholder to show where you'd recompile if required

    def new_state(self):
        return {"messages": [], "summary": "", "intent": None, "error": None}

    def last_ai(self, state):
        for m in reversed(state["messages"]):
            if isinstance(m, AIMessage):
                return m
        return None

    def tool_msgs(self, state, name=None):
        out = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        if name:
            out = [m for m in out if m.name == name]
        return out

    def test_routes_math_tool(self):
        state = self.new_state()
        state["messages"] += [HumanMessage(content="12*(8+2)")]
        state = arbii.graph.invoke(state)

        # Ensure calculator was called
        tms = self.tool_msgs(state, "calculator")
        self.assertTrue(len(tms) >= 1, "Calculator tool should have been called")
        # Verify final AIMessage present
        ai = self.last_ai(state)
        self.assertIsNotNone(ai)
        self.assertIn("[final] Used `calculator`", ai.content)

    def test_routes_time_tool(self):
        state = self.new_state()
        state["messages"] += [HumanMessage(content="what is the current time?")]
        state = arbii.graph.invoke(state)

        tms = self.tool_msgs(state, "current_time")
        self.assertTrue(len(tms) >= 1, "current_time tool should have been called")

        ai = self.last_ai(state)
        self.assertIsNotNone(ai)
        self.assertIn("[final] Used `current_time`", ai.content)

    def test_chat_route(self):
        state = self.new_state()
        state["messages"] += [HumanMessage(content="Tell me a joke")]
        state = arbii.graph.invoke(state)

        # No tools should be used
        self.assertEqual(len(self.tool_msgs(state)), 0)
        ai = self.last_ai(state)
        self.assertIsNotNone(ai)
        self.assertIn("(fake-chat)", ai.content)

    def test_clarifier_route(self):
        # First turn ambiguous
        state = self.new_state()
        state["messages"] += [HumanMessage(content="Book a flight")]
        state = arbii.graph.invoke(state)

        ai = self.last_ai(state)
        self.assertIsNotNone(ai)
        self.assertRegex(ai.content, r"Which .* dates .* destination", "Should ask a concise clarifying question")

    def test_summary_updates(self):
        state = self.new_state()
        # Turn 1
        state["messages"] += [HumanMessage(content="hello there")]
        state = arbii.graph.invoke(state)
        self.assertTrue(isinstance(state.get("summary"), str))

        # Turn 2
        state["messages"] += [HumanMessage(content="tell me more about yourself")]
        state = arbii.graph.invoke(state)
        self.assertTrue(len(state.get("summary","")) > 0, "Summary should accumulate")

    def test_single_system_message(self):
        state = self.new_state()
        for i in range(15):
            state["messages"] += [HumanMessage(content=f"msg {i}")]
            state = arbii.graph.invoke(state)
        # Ensure only 1 SystemMessage at head (trim works)
        sys_count = sum(isinstance(m, SystemMessage) for m in state["messages"])
        self.assertEqual(sys_count, 1, "There should be exactly one SystemMessage in the buffer")

if __name__ == "__main__":
    unittest.main()
