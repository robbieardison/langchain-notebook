from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-pro-exp",
    temperature=0.5
)

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

step_1 = fun_fact_prompt | llm
step_2 = RunnableLambda(lambda output: {"fact": output.content.strip()}) | story_prompt | llm
step_3 = RunnableLambda(lambda output: {"story": output.content.strip()}) | tweet_prompt | llm

full_chain = RunnableSequence(
    first=step_1,
    middle=[step_2],
    last=step_3
)

result = full_chain.invoke({"weather": "rainy day"})
print(result.content)


