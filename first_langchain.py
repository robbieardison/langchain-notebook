# Import the Google LLM wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# For loading env variables
from dotenv import load_dotenv
import os

# LangChain building blocks
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load your .env file
load_dotenv()
# Set up your key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Initialize Gemini Pro
llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp", temperature=0.7)

# Define a reusable prompt with a variable
prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?"
)

# Wrap model + prompt into a chain
chain = LLMChain(llm=llm, prompt=prompt)

# Run it!
print(chain.run("toothbrushes"))
