from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

@tool
def get_bitcoin_price() -> str:
    """Returns the current price of Bitcoin in USD."""
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=idr")
    data = response.json()
    return f"Bitcoin price is {data}"
    print(f"Bitcoin price is {data}")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [get_bitcoin_price]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

response = agent.invoke("What’s the current price of Bitcoin?")
print(response)
