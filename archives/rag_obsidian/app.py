import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings 


VAULT_PATH = "./2026"
print(f"🔍 Checking path: {VAULT_PATH}")

#  Load the markdown files

loader = DirectoryLoader(VAULT_PATH, 
                         glob="**/*.md", 
                         loader_cls=TextLoader, 
                         recursive=True)
docs = loader.load()

print(f"📄 Found {len(docs)} markdown files.")

if len(docs) == 0:
    print("❌ Error: No .md files found! Check your VAULT_PATH.")
else:
    # Split notes into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"🧠 Indexing {len(chunks)} chunks...")

    #Create the Vector Database
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=OllamaEmbeddings(model="gemma"),
        persist_directory="./rag_obsidian/chroma_db"
    )
    print("✅ Success! Your brain is ready.")
