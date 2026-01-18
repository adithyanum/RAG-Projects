import os
import ollama
import nltk
from nltk.corpus import stopwords
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_ollama import OllamaEmbeddings
from sentence_transformers import CrossEncoder

dir_path = "./my_library"

print('Loading Directory...')

director = DirectoryLoader(
    dir_path,
    glob = '**/*.pdf',
    loader_cls = PyPDFLoader
)

dirt = director.load()
print(f"Files Read : {len(dirt)}")

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
chunks = splitter.split_documents(dirt)

print(f'Reduced the PDF to {len(chunks)} byte-sized pieces..')

#Embeddings 

embeddings = OllamaEmbeddings(model = 'gemma')

#vectorization

db_folder = './agentic_rag/agent_db'

if not os.path.exists(db_folder):
    print("🧠 Building the brain for the first time...")
    # ... (Your Loading, Splitting, and DB Creation code goes here) ...
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_folder
    )
else:
    print("✅ Brain found! Loading from disk instantly...")
    # This just connects to the existing files without reading the PDFs
    vector_db = Chroma(
        persist_directory=db_folder,
        embedding_function=embeddings
    )

print('Success')

#Building Re-Ranker

judge_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def re_rank(query, retreived_docs) :
    
    pairs = [[query, doc.page_content] for doc in retreived_docs]

    scores = judge_model.predict(pairs)
    scored_docs = zip(retreived_docs, scores)

    sorted_docs = sorted(scored_docs, key = lambda x:x[1], reverse = True)
    return [doc for doc, score in sorted_docs[:3]]

def agent_query() :

    while True :
        query = input('Enter your question , Type exit to end : \t')
        if query.lower() == 'exit' :
            break

        docs = vector_db.similarity_search(query, k = 15 )

        stop_words = set(stopwords.words('english'))

        clean_query = query.lower().split()

        keywords = [word.strip("?,.") for word in clean_query if word not in stop_words and len(word) > 3]


        print(f"\n\n🎯 NLP Detective is looking for: {keywords}")


        manual_hits = []
        for chunk in chunks :
            if any(word in chunk.page_content.lower() for word in keywords) :
                if len(manual_hits) < 5 :
                    manual_hits.append(chunk)
        
        all_docs = manual_hits + docs
        
        ranked_docs = re_rank(query, all_docs)

        formatted_chunks = []
        for c in ranked_docs:
            source = os.path.basename(c.metadata.get('source', 'Unknown'))
            formatted_chunks.append(f"[Source: {source}]\n{c.page_content}")

        context = "\n\n---\n\n".join(formatted_chunks)

        prompt = f"""
        <role>You are a strict AI Professor who must cite sources.</role>

        <rules>
        1. EVERY sentence must include a [Source: filename] tag.
        2. If the source is missing, you must state: "Source not found."
        </rules>

        <context>
        {context}
        </context>

        <question>
        {query}
        </question>

        EXAMPLE RESPONSE:
        According to [Source: vanka.pdf], Vanka is nine years old and lives with a shoemaker.
        
        REMINDER: You are NOT allowed to answer without citing the source name.
        """

        # 4. SEND TO GEMMA
        response = ollama.chat(model='gemma', messages=[{'role': 'user', 'content': prompt}])
        print(f"\n🤖 Answer: {response['message']['content']}")

agent_query()