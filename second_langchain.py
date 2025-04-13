from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableSequence, RunnableLambda
from dotenv import load_dotenv
import os

# Load your Gemini API Key from .env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Load Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-pro-exp", temperature=0.7)

# First prompt: generate a startup idea
startup_prompt = PromptTemplate(
    input_variables=["product"],
    template="Give me a creative startup idea for a product that sells {product}."
)

# Second prompt: generate a slogan from the idea
slogan_prompt = PromptTemplate(
    input_variables=["idea"],
    template="Create a catchy slogan for this startup idea: {idea}"
)

# Step 1: startup idea (product -> idea)
step_1 = startup_prompt | llm

# Step 2: map output from step_1 to input for step_2
extract_idea = RunnableLambda(lambda output: {"idea": output.content.strip()})

step_2 = slogan_prompt | llm

# Combine into a full chain
full_chain = RunnableSequence(first=step_1, middle=[extract_idea], last=step_2)

# Run the chain
result = full_chain.invoke({"product": "self-driving cars"})

# Access the content of the AIMessage object
print(result.content)
