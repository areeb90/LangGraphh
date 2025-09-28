# Title: Aster – LangGraph Agent with Tools, Intent Routing & Memory

## Tech: Python · LangGraph · LangChain · OpenAI (GPT-4o-mini) · Pydantic · dotenv

Built a production-style conversational agent using LangGraph that routes user intents, calls tools (calculator, current time), and maintains a rolling summary memory. The agent injects a fresh system persona every turn, trims context to control cost, and reliably handles tool loops with guards.

# Highlights:

- Intent router (tools | chat | clarify) using structured output

- Tool calling with automatic loop handling (up to 3 tool steps)

- Rolling summary memory merged into the system prompt each turn

- Token/cost control via message trimming (last 12 non-system messages)


<img width="1805" height="484" alt="image" src="https://github.com/user-attachments/assets/0800b7c8-cc14-4aa1-a2c3-fe16b85747a2" />


<img width="1799" height="874" alt="image" src="https://github.com/user-attachments/assets/02dc53a8-6b0a-4cc5-b203-644ab3463c00" />


<img width="1787" height="233" alt="image" src="https://github.com/user-attachments/assets/e58d4370-ad8f-448f-ba08-32828d5e226f" />


# 2__advanced_agent.py 

## Added GUI using streamlit

<img width="1905" height="920" alt="image" src="https://github.com/user-attachments/assets/48a77e06-9997-4088-ad4a-ab6b36ecea7a" />

# advanced_agent_v2.py

## Added persistent memory

🧠 What is Persistent Memory (Chroma)?

Normal LangGraph memory in your agent = short rolling summary (only keeps the last few turns).

Persistent memory with Chroma = long-term memory stored on disk so your agent can recall things even after you stop and restart the app.

It works like this:

After each turn, you take a summary, fact, or preference and embed it (turn into a vector).

Store that vector + the text in ChromaDB (a local vector database).

Later, when the user asks something new, you query Chroma to fetch relevant old info → inject it into the system prompt.

So persistent memory ≈ “a searchable notebook” of past conversations.

🗂️ Where is it stored?

By default, in a folder on disk (e.g. ./.arbii_memory/).

Inside that folder, Chroma manages SQLite + Parquet files that hold vectors and metadata.

## TL;DR Summary

You built a Streamlit chat app powered by a LangGraph state machine.

Every user turn:

1- A fresh SystemMessage is injected with the system prompt, a rolling summary, and retrieved Chroma memories relevant to the latest user message.

2- An intent classifier routes to: tools, chat, or clarify.

3- Tools (calculator/time) are orchestrated safely for up to 3 call rounds.

4- The turn is summarized and that short summary is persisted into Chroma for future retrieval.

- The UI shows:

chat bubbles,

tool-call traces,

a persistent memory panel (queryable, count shown, wipe button).

- All state (messages, intent, summary) lives in AgentState; UI rendering uses a separate chat_log for clean display.

<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/ee6c125a-92a6-4d42-83e4-0c3d857f8da2" />

<img width="1920" height="894" alt="image" src="https://github.com/user-attachments/assets/e475b34f-d10d-4772-8c0b-59648558a3c1" />


## folder structure

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/716fde07-68db-47c7-83cf-1dfcd900b751" />


# advanced_agent_v3.py
<img width="1918" height="900" alt="image" src="https://github.com/user-attachments/assets/d40ad54d-48af-496f-804e-d954d1b47092" />


## folder structure
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/6e303638-75d8-40b1-aecf-2bc57edbf4ad" />
