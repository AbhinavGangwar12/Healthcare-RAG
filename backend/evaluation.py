from typing import List, Dict, Any

from backend.retrieval import hybrid_retrieve
from backend.reasoning import generate_answer
from backend.safety import is_query_safe

def process_query(question: str):
    """
    Combine retrieval, safety, and reasoning to produce final answer.
    Returns answer string, list of citations, and confidence.
    """
    # Retrieve relevant docs with scores
    doc_scores = hybrid_retrieve(question, top_k=5)
    # Safety check
    safe = is_query_safe(question, [doc for doc, _ in doc_scores])
    if not safe:
        # Fallback message for unsafe queries
        return "Insufficient medical evidence in the provided documents.", [], 0.0
    # Prepare citations list from top docs
    citations = []
    for doc, score in doc_scores:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        chunk = doc.metadata.get("chunk", "")
        citations.append({
            "source": source,
            "page": page,
            "chunk": chunk,
            "confidence": score
        })
    # Generate answer using reasoning agent
    answer = generate_answer(question)
    # Use highest score as overall confidence
    confidence = doc_scores[0][1] if doc_scores else 0.0
    return answer, citations, confidence
