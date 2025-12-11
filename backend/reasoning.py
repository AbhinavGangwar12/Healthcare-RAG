from backend.retrieval import hybrid_retrieve
from langchain_ollama import ChatOllama

LLM = ChatOllama(model="mistral", temperature=0)

def generate_answer(question: str) -> str:
    """
    Retrieves relevant document chunks and passes them with the question to the LLM.
    Returns a grounded answer based only on retrieved content.
    """
    doc_scores = hybrid_retrieve(question, top_k=5)

    if not doc_scores:
        return "No relevant content was found in the documents."

    context = ""
    for doc, score in doc_scores:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        chunk = doc.metadata.get("chunk", "")
        context += f"[{source} | page {page} | chunk {chunk}]\n{doc.page_content}\n\n"

    prompt = (
        "You are a helpful medical assistant. Use only the context below to answer the user's question.\n"
        "If the answer cannot be found in the context, respond with:\n"
        "\"I don't know based on the provided documents.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    response = LLM.invoke(prompt)
    return response.content.strip() if hasattr(response, "content") else str(response).strip()

