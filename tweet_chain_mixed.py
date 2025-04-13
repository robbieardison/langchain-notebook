from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

llm_gpt4o_mini = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

llm_claude = ChatAnthropic(
    model="claude-3.7-sonnet",
    temperature=0.7,
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Define prompt templates
fun_fact_prompt = PromptTemplate(
    input_variables=["weather"],
    template="Tell me a fun and surprising fact about the weather today: {weather}."
)

story_prompt = PromptTemplate(
    input_variables=["fact"],
    template="Write a historical story based on this fact: {fact}"
)

tweet_prompt = PromptTemplate(
    input_variables=["story"],
    template="Turn the following story into a fun, concise tweet:\n\n{story}"
)

# Step 1: Generate a fun fact using Gemini
step_1 = fun_fact_prompt | llm_gemini

# Step 2: Create a story using GPT-4o Mini
step_2 = (
    RunnableLambda(lambda output: {"fact": output.content.strip()})
    | story_prompt
    | llm_gpt4o_mini
)

# Step 3: Generate a tweet using Claude
step_3 = (
    RunnableLambda(lambda output: {"story": output.content.strip()})
    | tweet_prompt
    | llm_claude
)

# Combine steps into a sequence
full_chain = RunnableSequence(
    first=step_1,
    middle=[step_2],
    last=step_3
)

# Execute the chain
result = full_chain.invoke({"weather": "rainy day"})
print(result.content)
