from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.tools import tool
from dotenv import load_dotenv
import random
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@tool
def get_python_tip(input: str) -> str:
    """Gives a random Python tip."""
    tips = [
        "Use list comprehensions instead of loops when possible!",
        "Use `enumerate()` when you need both index and value in a loop.",
        "Use f-strings for cleaner and faster string formatting.",
        "Use `with open(...)` to automatically close files.",
        "Avoid using mutable default arguments like lists or dicts.",
        "Leverage built-in functions like `sum()`, `max()`, and `any()` for cleaner code.",
        "Use `zip()` to iterate over multiple sequences at once.",
        "Use type hints to improve readability and catch bugs early.",
        "Use virtual environments to manage dependencies cleanly.",
        "Remember: `is` is for identity, `==` is for equality."
    ]
    return f"Here's a Python tip: {random.choice(tips)}"

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

agent = initialize_agent(
    tools=[get_python_tip],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

agent.invoke("Hi, I'm trying to learn Python.")

agent.invoke("What did I just say?")