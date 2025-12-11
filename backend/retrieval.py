import os
from typing import List, Tuple
import math

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-large"
FAISS_INDEX_DIR = os.path.join("data", "processed", "faiss_index")

def hybrid_retrieve(query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
    """
    Perform hybrid retrieval: dense (FAISS) + sparse (BM25) and rerank results.
    Returns top_k documents with confidence scores.
    """
    # Load FAISS vectorstore
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    if not os.path.exists(FAISS_INDEX_DIR):
        return []
    vs = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    # Dense retrieval (embedding search)
    dense_retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": top_k * 3})
    dense_docs = dense_retriever.invoke(query)
    # BM25 retrieval: build on all docs from FAISS
    # Assuming FAISS docstore holds all docs
    all_texts = [doc.page_content for doc in vs.docstore._dict.values()]
    all_docs = list(vs.docstore._dict.values())

    bm25_retriever = BM25Retriever.from_documents(all_docs)
    bm25_docs = bm25_retriever.invoke(query)
    # Combine unique docs preserving density of relevance
    combined = { (doc.metadata["source"], doc.metadata["page"], doc.metadata["chunk"]): doc 
                 for doc in dense_docs + bm25_docs }
    unique_docs = list(combined.values())
    if not unique_docs:
        return []
    # Rerank using cross-encoder (BGE reranker)
    tokenizer = AutoTokenizer.from_pretrained(RERANK_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(RERANK_MODEL)
    model.eval()
    # Prepare input pairs for reranker
    pairs = [[query, doc.page_content] for doc in unique_docs]
    with torch.no_grad():
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        logits = model(**inputs).logits.view(-1)
        scores = logits.tolist()
    # Attach scores, compute confidence
    doc_scores = []
    for doc, score in zip(unique_docs, scores):
        # Convert raw score to [0,1] confidence via sigmoid
        conf = 1 / (1 + math.exp(-score))
        doc_scores.append((doc, conf))
    # Sort by confidence descending
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    return doc_scores[:top_k]
