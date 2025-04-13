# LangChain Lesson: Using Gemini with External Tools (Manual Agent Simulation)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv
import requests
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# Step 1: Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp" , temperature=0.3)

# Step 2: Define a simple external tool (real-time Bitcoin price fetcher)
def get_bitcoin_price():
    response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=idr")
    data = response.json()
    return data

# Step 3: First Prompt - User asks a question
initial_question = "What is the current price of Bitcoin?"

# Step 4: Send the question to Gemini
first_response = llm.invoke(initial_question)

# Step 5: Simulate tool routing manually
# (In real use, you'd use regex/NLP to detect tool intent)
needs_tool = "bitcoin" in initial_question.lower() and "price" in initial_question.lower()

if needs_tool:
    # Step 6: Call the tool manually
    tool_result = get_bitcoin_price()

    # Step 7: Construct a follow-up input with the tool result
    followup_input = f"You asked earlier: '{initial_question}'. Here's the real-time data: {tool_result}. Summarize this in one sentence."

    # Step 8: Get the final response from Gemini
    final_response = llm.invoke(followup_input)

    print("--- Final Response ---")
    print(final_response.content)
else:
    print("--- Gemini Response ---")
    print(first_response.content)
