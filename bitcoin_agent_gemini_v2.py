from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import requests
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


# Step 2: Define a simple external tool (real-time Bitcoin price fetcher)
def get_bitcoin_price():
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=idr")
    data = response.json()
    return data

# Gemini chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp", temperature=0)

# Custom prompt
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You're a helpful crypto assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create agent
agent = create_openai_functions_agent(llm=llm, tools=[get_bitcoin_price], prompt=prompt)

# Agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[get_bitcoin_price],
    verbose=True,
    handle_parsing_errors=True,  # Important for Gemini
)

# Call agent
response = agent_executor.invoke({
    "input": "What's the price of Bitcoin?",
    "chat_history": [],
    "agent_scratchpad": []  # safe init — LangChain fills it after tool call
})

print(response["output"])
