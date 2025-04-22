from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

# Load API keys
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load PDF document
pdf_docs = PyMuPDFLoader("logic_test_hc.pdf").load()
print(f"PDF Content: {pdf_docs}")  # Debug PDF content

# Load website content
web_docs = WebBaseLoader(["https://www.geeksforgeeks.org/architecture-of-8085-microprocessor/"]).load()
print(f"Website Content: {web_docs}")  # Debug website content

# Combine documents
all_docs = pdf_docs + web_docs
print(f"Loaded {len(all_docs)} documents.")  # Debug loaded documents

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = splitter.split_documents(all_docs)
print(f"Created {len(docs)} document chunks.")  # Debug document chunks

# Embed and store in vector database
embedding_model = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embedding=embedding_model)
print("Vectorstore initialized with embeddings.")  # Debug vectorstore

# Create retriever and QA chain
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model='gpt-4o-mini', temperature=0), retriever=retriever)

# Query the documents
query = "How much bits of register does the website talk about?"
response = qa.invoke({"query": query})  # Use invoke instead of run
result = response.get("result", "No answer found.")  # Safely extract the result
print(f"\n📘 Answer:\n{result}")
