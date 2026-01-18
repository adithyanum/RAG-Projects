import ollama
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

# 1. Load the database we just created
vector_db = Chroma(
    persist_directory="./rag_obsidian/chroma_db", 
    embedding_function=OllamaEmbeddings(model="gemma")
)

def ask_my_notes(question):
    # 2. Find the most relevant notes
    results = vector_db.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in results])

    # 3. Ask Gemma using the notes as context
    prompt = f"Using ONLY the following context, answer the question: {question}\n\nContext:\n{context}"
    
    response = ollama.chat(model='gemma', messages=[
        {'role': 'user', 'content': prompt},
    ])
    
    return response['message']['content']

query = "Explain the Tortoise and Hare algorithm based on my notes."
print(f"🤔 Question: {query}")
print(f"🤖 Answer:\n{ask_my_notes(query)}")