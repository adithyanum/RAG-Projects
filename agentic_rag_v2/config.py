import os

# --- Model Settings ---
LLM_MODEL = "gemma" 
EMBEDDING_MODEL = "gemma"

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Points to your specific SCT college or project data folder
SOURCE_DIR = os.path.join(BASE_DIR, "source_data")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_store")

# --- RAG Parameters ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 250
RE_RANK_TOP_K = 3

SYSTEM_PROMPT = """
You are a Research Assistant. You MUST use this EXACT format:

THOUGHT: <your reasoning here>
ACTION: tool_name("search_query")

Wait for an OBSERVATION, then:
FINAL ANSWER: <your summarized answer with sources>

Example:
User: Who is Vanka?
THOUGHT: I need to check the local documents for Vanka.
ACTION: local_search("Vanka")

RULES:
1. Always use local_search first and if the info received dont answer the question use web_search.
2. [IMPORTANT] If answer not provided in given context use web_search.
1. Only use local_search or web_search.
2. Do not explain yourself outside the tags.
3. If you have the answer, use FINAL ANSWER.
"""