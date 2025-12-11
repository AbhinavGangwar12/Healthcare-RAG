# 🏥 Enterprise Healthcare RAG System

A production-ready, full-stack Retrieval-Augmented Generation (RAG) system built with LangChain, FastAPI, FAISS, and Streamlit. This system processes medical PDFs, performs hybrid retrieval with reranking, and answers queries with grounded, multi-document reasoning.

---

## 🔍 Features

- 📄 PDF Upload and Ingestion (Text Splitting + BGE Embeddings)
- 🔍 Hybrid Retrieval (Dense FAISS + Sparse BM25)
- 🎯 Reranking with BGE-Reranker
- 🧠 Multi-hop Reasoning via ReAct-style LangChain agent
- ✅ Medical Safety Filter (blocks unsafe or out-of-context questions)
- 📚 Citation with document name, page, chunk, and confidence
- ⚡ FastAPI Backend + Streamlit Frontend
- 💬 Local LLM via Ollama (e.g., Llama3, Mistral)

---

## 📸 Demo / UI Preview

<!-- Replace this with your screenshot or demo GIF -->
![Sample Output](path/to/your/demo_screenshot.png)

---

## 🛠️ Tech Stack

| Component    | Technology                         |
|--------------|-------------------------------------|
| Backend      | FastAPI, LangChain                 |
| Frontend     | Streamlit                          |
| Embeddings   | BGE Large v1.5                     |
| Reranker     | BGE-Reranker-Large                 |
| Vector Store | FAISS                              |
| LLM          | Llama3 / Mistral via Ollama        |
| Safety       | Regex Filtering + Evidence Checking|

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
