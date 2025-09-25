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

<img width="1920" height="912" alt="image" src="https://github.com/user-attachments/assets/ee6c125a-92a6-4d42-83e4-0c3d857f8da2" />

<img width="1920" height="894" alt="image" src="https://github.com/user-attachments/assets/e475b34f-d10d-4772-8c0b-59648558a3c1" />


## folder structure

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/716fde07-68db-47c7-83cf-1dfcd900b751" />


