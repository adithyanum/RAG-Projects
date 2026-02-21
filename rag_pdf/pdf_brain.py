import os
import ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_ollama import OllamaEmbeddings

pdf_path = "./my_library/ilide.info-two-page-story-pr_551d0bfe3037842f0f827abe98ddf52a.pdf"
print(f'Checking path : {pdf_path}')

#Loading The PDF....

print('Loading PDF..')
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# Force-print the first 1000 characters of the PDF to see the raw text
print("--- RAW TEXT START ---")
print(pages[0].page_content[:1000])
print("--- RAW TEXT END ---")

print(f'found {len(pages)} PDF...')

#Splitting to small Chunks..

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 300)
chunks = splitter.split_documents(pages)

print(f'Reduced the PDF to {len(chunks)} byte-sized pieces..')

#Creations of the Embeddings..

embeddings = OllamaEmbeddings(model = 'gemma')

#vector store 

print('Turning the pdf chunks to math (vectors) and storing them to pdf_db')

vector_db = Chroma.from_documents(
    documents = chunks,
    embedding = embeddings,
    persist_directory = './rag_pdf/pdf_db'
)

print('Success!')

def chat_with_pdf() :
    #connect to PDF database 
    vector_db = Chroma(
        persist_directory = './rag_pdf/pdf_db',
        embedding_function = OllamaEmbeddings(model = 'gemma')
    )
    
    print("\n AI is ready! (Type 'exit' to stop)")

    while True :
        query = input("\n Ask the question..")

        if query.lower() == 'exit' :
            break
        
        #semantic search pdf for answer 
        docs = vector_db.similarity_search(query, k=7)
        
        #keyword search 
        clean_query = query.lower()
        keywords = [word.strip("?,.") for word in clean_query.split() if len(word) > 3]

        manual_hits = []
        for chunk in chunks:
           # Check if ANY keyword is in the chunk
            if any(word in chunk.page_content.lower() for word in keywords):
                manual_hits.append(chunk.page_content)
       
        #combine them 

        context = "\n\n---\n\n".join(manual_hits[:4] + [d.page_content for d in docs[:3]])

        #prompt
        prompt = f"Use the context below to answer.\nContext: {context}\n\nQuestion: {query}"

        response = ollama.chat(model='gemma', messages=[{'role': 'user', 'content': prompt}])
        print(f"\n🤖 Answer: {response['message']['content']}")

chat_with_pdf()