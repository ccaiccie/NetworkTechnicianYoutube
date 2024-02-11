#!/usr/bin/python3
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain

# Prompt user for URL and question
user_url = input("Please enter the URL you'd like to analyze: ")
user_question = input("What is your question? ")

# Initialize Ollama model
llm = Ollama(model="llama2")

# Load documents from the web using the user-provided URL
loader = WebBaseLoader(user_url)
docs = loader.load()

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

# Create chat prompt template
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retrieval chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Invoke the retrieval chain with the user's question
response = retrieval_chain.invoke({"input": user_question})
print(response["answer"])
