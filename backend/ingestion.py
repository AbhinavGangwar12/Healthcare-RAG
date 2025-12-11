# import os
# from typing import List
# from pathlib import Path

# from langchain_core.documents import Document
# from langchain_community.document_loaders import PyMuPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200
# FAISS_INDEX_DIR = os.path.join("data", "processed", "faiss_index")

# def ingest_pdf(file_path: str):
#     """
#     Ingest a PDF file: load, split into chunks, embed, and store in FAISS.
#     """
#     # Load PDF and extract pages
#     loader = PyMuPDFLoader(file_path)
#     docs = loader.load()  # List of Documents, one per page
#     if not docs:
#         raise ValueError(f"No content extracted from {file_path}.")
#     # Prepare text splitter
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
#     )
#     # Split pages into chunks and collect Document objects
#     chunks: List[Document] = []
#     for page_doc in docs:
#         page_content = page_doc.page_content
#         page_num = page_doc.metadata.get("page", 0) + 1  # 1-index pages
#         texts = splitter.split_text(page_content)
#         for i, chunk_text in enumerate(texts):
#             metadata = {
#                 "source": os.path.basename(file_path),
#                 "page": page_num,
#                 "chunk": i
#             }
#             chunks.append(Document(page_content=chunk_text, metadata=metadata))
#     if not chunks:
#         raise ValueError(f"No chunks created from {file_path}.")
#     # Create embeddings model
#     embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
#     # Load or create FAISS vector store
#     if os.path.exists(FAISS_INDEX_DIR):
#         vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
#         vectorstore.add_documents(chunks)
#     else:
#         vectorstore = FAISS.from_documents(chunks, embeddings)
#     # Ensure directory exists and save FAISS index
#     os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
#     vectorstore.save_local(FAISS_INDEX_DIR)


import os
from typing import List
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
FAISS_INDEX_DIR = os.path.join("data", "processed", "faiss_index")
FAISS_INDEX_FILE = os.path.join(FAISS_INDEX_DIR, "index.faiss")

def ingest_pdf(file_path: str):
    """
    Ingest a PDF file: load, split into chunks, embed, and store in FAISS.
    """
    # Load PDF and extract pages
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    if not docs:
        raise ValueError(f"No content extracted from {file_path}.")

    # Prepare text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    # Split pages into chunks
    chunks: List[Document] = []
    for page_doc in docs:
        page_content = page_doc.page_content
        page_num = page_doc.metadata.get("page", 0) + 1
        texts = splitter.split_text(page_content)
        for i, chunk_text in enumerate(texts):
            metadata = {
                "source": os.path.basename(file_path),
                "page": page_num,
                "chunk": i
            }
            chunks.append(Document(page_content=chunk_text, metadata=metadata))

    if not chunks:
        raise ValueError(f"No chunks created from {file_path}.")

    # Create embedding model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load or create FAISS index
    if os.path.exists(FAISS_INDEX_FILE):
        vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        vectorstore.add_documents(chunks)
    else:
        os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
        vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index
    vectorstore.save_local(FAISS_INDEX_DIR)
