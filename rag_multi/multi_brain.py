import os
import ollama
import nltk
from nltk.corpus import stopwords
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma 
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader

multi_path = "./my_library"
print(f"Checking path.. : {multi_path}")
print('Success!!')

#Loading the directory :

print('Loading directory..!')
loader = PyPDFLoader
director = DirectoryLoader(
    multi_path, 
    glob = '**/*.pdf', 
    loader_cls = loader)

dirt = director.load()

print(f"Found {len(dirt)} files...!")

#splitting

splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 250)
chunks = splitter.split_documents(dirt)

print(f'Reduced the PDF to {len(chunks)} byte-sized pieces..')

#Embeddings 

embeddings = OllamaEmbeddings(model = 'gemma')

#vectorization

db_folder = './rag_multi/multi_db'

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

def question_hour() :

    print('AI is ready...!')

    while True :
        query = input('\n\nHow Can I Assist You ? , Type exit to end! \t')

        if query.lower() == 'exit' :
            break

        docs = vector_db.similarity_search(query, k=5)

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
        
        formatted_chunks = []
        seen_content = set()

        for c in all_docs :
            # getting source name
            if c.page_content not in seen_content :
                source = os.path.basename(c.metadata.get('source', 'Unknown'))
                formatted_chunks.append(f"[Source : {source}]\n{c.page_content}")
                seen_content.add(c.page_content)

        context = "\n\n--\n\n".join(formatted_chunks)
       
       
        print(f"DEBUG: The Librarian found chunks from: {[os.path.basename(c.metadata.get('source', '')) for c in all_docs]}")
        print('\n\n',context[:150])
       
        prompt = f"""
        You are an AI Professor. Always cite the source which is provided in the context and Use the context provided to answer the question.
        
        RULES:
        1.You MUST start by mentioning its [Source] name which is also provided with the context.
        2. If the answer is not in the context, say you don't know.
        3. Don't ever forget to cite the source! (Important)

        CONTEXT:
        {context}

        QUESTION: 
        {query}
        """

        # 5. Send to Gemma
        response = ollama.chat(model='gemma', messages=[{'role': 'user', 'content': prompt}])
        print(f"\n🤖 Answer: {response['message']['content']}")

question_hour()