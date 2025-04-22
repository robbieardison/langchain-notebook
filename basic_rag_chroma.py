import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA


# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load your documents
loader = TextLoader("./sample_meeting.txt")  # or PDFLoader, etc.
docs = loader.load()

# Split the docs into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Embed and store in vector database (Chroma)
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(chunks, embedding_model)

# Create the retrieval-based QA chain
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Main loop
print("Ask me anything about your docs (type 'exit' to quit):")
while True:
    query = input("\n> ")
    if query.lower() in ['exit', 'quit']:
        break
    answer = qa_chain.invoke(query)
    result = answer.get("result", answer)
    print(f"\n📘 Answer:\n{result}")
