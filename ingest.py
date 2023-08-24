import streamlit as st
import openai
from langchain.document_loaders import NotionDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings

# Load OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Load the notion content located in the notion_content folder
loader = NotionDirectoryLoader("notion_docs")
documents = loader.load()

# Split Notion content into smaller chunks
markdown_splitter = RecursiveCharacterTextSplitter(
    separators=["#","##", "###", "\n\n","\n","."],
    chunk_size=1000,
    chunk_overlap=100)
docs = markdown_splitter.split_documents(documents)

# Initialize OpenAI embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") 
embedded_docs = embeddings.embed_documents(docs)
# embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", show_progress_bar = True)

# Convert all chunks into vectors embeddings using OpenAI embedding model
# Store all vectors in FAISS index and save locally to 'faiss_index'
db = FAISS.from_embeddings(embedded_docs, embeddings)
db.save_local("faiss_index")

print('Local FAISS index has been successfully saved.')