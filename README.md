# RAG Projects — Hybrid Retrieval Pipeline

A multi-stage Retrieval-Augmented Generation system built with a focus on retrieval quality over basic vector search. Each version improves on the last — from naive retrieval to a fully agentic system with autonomous web fallback.

## Versions

### V1 — Naive RAG
Basic vector search implementation using FAISS embeddings.
- Dense vector retrieval only
- Simple similarity search
- Directories: `rag_pdf/`, `rag_obsidian/`

### V2 — Advanced RAG
Significant upgrade to retrieval quality using hybrid search and reranking.
- Hybrid Search — combines dense vector retrieval (FAISS) with BM25 keyword matching
- Cross-Encoder Reranking — reranks retrieved chunks for improved context relevance
- Local LLM backend — fully offline using Ollama with Gemma, no external API dependency
- Directory: `rag_multi/`

### V3 — Agentic RAG
ReAct agent with autonomous tool use and a strict local-before-web search protocol.
- Agent reasons through queries using THOUGHT → ACTION → OBSERVATION loops
- Always checks local documents first before escalating to web search
- Web fallback powered by DuckDuckGo — cross-encoder reranking applied to web results too
- ChromaDB for persistent local vector storage with Ollama embeddings
- Directory: `agentic_rag_v2/`

## Tech Stack
- LLM: Ollama + Gemma (local, offline)
- Embeddings: Ollama (Gemma)
- Vector Store: ChromaDB (V3), FAISS (V1, V2)
- Keyword Search: BM25 (rank-bm25) — V2
- Reranking: Cross-encoder (ms-marco-MiniLM-L-6-v2)
- Web Search: DuckDuckGo (ddgs)
- Framework: LangChain
- Language: Python

## Why Hybrid Search?
Standard RAG uses only dense vector search — great for semantic similarity but misses exact keyword matches. BM25 catches what vector search misses. Combining both and reranking with a cross-encoder gives significantly better retrieval precision than either method alone.

## Why Agentic RAG?
Static RAG pipelines always retrieve — even when the local corpus has nothing relevant. The agentic approach lets the model decide: check local documents first, and only fall back to live web search when local results are insufficient. This keeps responses grounded in your documents while staying useful for out-of-domain queries.

## Setup
```bash
git clone https://github.com/adithyanum/RAG-Projects
cd RAG-Projects
pip install -r requirements.txt
```

Make sure Ollama is running locally:
```bash
ollama pull gemma
ollama serve
```

## Results
Hybrid search + cross-encoder reranking consistently retrieves more relevant chunks compared to naive vector search, especially for domain-specific queries where exact terminology matters. The agentic layer adds adaptability — handling both in-corpus and open-domain questions in a single pipeline.

---
Built by Adithyan U M — [github.com/adithyanum](https://github.com/adithyanum)
