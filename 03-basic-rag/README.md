# 📚 Basic RAG: Zero-Hallucination PDF Assistant

## ☁️ Overview
A Retrieval-Augmented Generation (RAG) prototype built to interact with local PDF documents. This application allows users to upload technical manuals or dense documents and query them in real-time. By leveraging a strict prompting architecture, the assistant is constrained to answer *only* using the provided context, effectively eliminating LLM hallucinations.

👉 **[Try the app here on Streamlit](https://stateless-basic-rag.streamlit.app/)**

## 🎯 Objective
To demonstrate the foundational mechanics of a RAG pipeline; from document ingestion and vector embedding, to similarity search and context-injected text generation; prioritizing factual accuracy over creative liberty.

## 🛠️ Techniques
* **Local Vectorization:** Utilizing `faiss-cpu` for efficient, in-memory vector storage and similarity search.
* **Fast Embeddings:** Implementing HuggingFace's lightweight `all-MiniLM-L6-v2` model for lightning-fast, CPU-friendly document chunk embedding.
* **Anti-Hallucination Prompting:** Designing strict system prompts that force the LLM to admit "I don't know" rather than generating statistically probable but factually incorrect responses when information is missing from the source text.
* **LCEL Orchestration:** Using LangChain Expression Language to build a readable, modular, and piping-based RAG chain (`Context + Question -> Prompt -> LLM -> Parser`).

## 🧰 Skills and tools
* **Languages:** Python
* **Frameworks:** LangChain Core/Community, Streamlit, FAISS, PyMuPDF
* **LLM Provider:** Groq API (Llama-3-70b-versatile)
* **Embeddings:** HuggingFace `sentence-transformers`
* **Concepts:** Document Chunking, Vector Space, Semantic Search, UI State Management

## 🚧 Known limitations & Next steps
This prototype is intentionally basic to serve as a stepping stone. Current limitations include:
1. **Stateless Chat:** The system evaluates each query independently without memory of previous turns, making follow-up questions unavailable.
2. **Single Format Bottleneck:** It currently only accepts one PDF at a time, lacking multi-document or multi-format (URLs, CSVs) capabilities.

---
*Note: The web application is in Spanish. The description here is in English for portfolio presentation.*