from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel  # ✅ Added for request model
import os
import shutil
import traceback


from backend.ingestion import ingest_pdf
from backend.rag_pipeline import process_query

app = FastAPI(title="Healthcare RAG Backend")

# Ensure directories exist
RAW_PDFS_DIR = os.path.join("data", "raw_pdfs")
os.makedirs(RAW_PDFS_DIR, exist_ok=True)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload endpoint to accept a PDF file and run ingestion pipeline.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    # Save uploaded file to data/raw_pdfs/
    file_path = os.path.join(RAW_PDFS_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Run ingestion pipeline on saved PDF
    try:
        ingest_pdf(file_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")
    return {"status": "success", "detail": f"File '{file.filename}' uploaded and ingested."}

# ✅ Define a request body model
class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def query(request: QueryRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Question must be provided.")
    try:
        answer, citations, confidence = process_query(request.query)
    except Exception as e:
        traceback.print_exc()  # 🔴 This logs the full traceback to the console
        raise HTTPException(status_code=500, detail=f"Query processing failed: {e}")
    return JSONResponse(content={
        "answer": answer,
        "citations": citations,
        "confidence": confidence
    })
