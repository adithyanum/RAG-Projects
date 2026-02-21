import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder
from ddgs import DDGS 
from config import SOURCE_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP

# Initialize Embeddings using config
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Vector DB Logic
if not os.path.exists(VECTOR_DB_DIR):
    print("🧠 Building the brain for the first time...")
    loader = DirectoryLoader(SOURCE_DIR, glob='**/*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(documents)
    vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=VECTOR_DB_DIR)
else:
    print("✅ Brain found! Loading from disk...")
    vector_db = Chroma(persist_directory=VECTOR_DB_DIR, embedding_function=embeddings)

judge_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def re_rank(query, retrieved_docs):
    pairs = [[query, doc['context'] if isinstance(doc, dict) else doc.page_content] for doc in retrieved_docs]
    scores = judge_model.predict(pairs)
    sorted_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in sorted_docs[:3]]

def local_search(query: str):
    initial_docs = vector_db.similarity_search(query, k=15)
    ranked_docs = re_rank(query, initial_docs)
    return "\n*********\n".join([f"Source: {c.metadata.get('source')} \nContext: {c.page_content}" for c in ranked_docs])

def web_search(query: str):
    print(f"🌐 Searching the web for: {query}")
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=10):
            results.append({'title': r['title'], 'url': r['href'], 'context': r['body']})
    ranked_docs = re_rank(query, results)
    return "\n*******\n".join([f"### {c['title']}\nLink: {c['url']}\nContent: {c['context']}" for c in ranked_docs])

AVAILABLE_TOOLS = {"local_search": local_search, "web_search": web_search}