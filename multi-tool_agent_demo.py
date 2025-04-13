from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Define tools
@tool
def summarize_meeting(transcript: str) -> str:
    """Summarize the meeting transcript."""
    return llm.invoke(f"Summarize this meeting: {transcript}")

@tool
def extract_tasks(transcript: str) -> str:
    """Extract action items from the meeting transcript."""
    return llm.invoke(f"Extract key action items from the meeting: {transcript}")

@tool
def add_task_to_notion(task: str) -> str:
    """Mock function to simulate adding a task to Notion."""
    return f"✅ Task '{task}' added to Notion (simulated)."

@tool
def schedule_next_meeting(date: str) -> str:
    """Mock function to simulate scheduling a meeting."""
    return f"📅 Meeting scheduled for {date} (simulated)."

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent with the tools
agent = initialize_agent(
    tools=[summarize_meeting, extract_tasks, add_task_to_notion, schedule_next_meeting],
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=True
)

# Simulate user input
meeting_transcript = """
Today we discussed the new marketing campaign. Sarah will design the draft, Tom will review by Friday, and we aim to launch next Monday. Let's follow up on Thursday.
"""

# Agent runs
print("--- Summarizing Meeting ---")
agent.invoke(f"Summarize this meeting: {meeting_transcript}")

print("\n--- Extracting Tasks ---")
agent.invoke(f"Extract tasks from this: {meeting_transcript}")

print("\n--- Adding a Task to Notion ---")
agent.invoke("Add this task to Notion: Sarah will design the draft")

print("\n--- Scheduling Next Meeting ---")
agent.invoke("Schedule the next meeting for Thursday at 3PM")