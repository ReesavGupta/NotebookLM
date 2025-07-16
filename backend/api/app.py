from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from services.ingestion import process_and_store_chunks, qdrant, clip_embd, sparse_embd, COLLECTION
from services.retrieval import contextual_compression
from services.llm_service import multimodal_query, llm, decompose_query
from services.models import SessionLocal, Document
from sqlalchemy.orm import Session
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance


vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION,
    embedding=clip_embd,
    sparse_embedding=sparse_embd, 
    retrieval_mode=RetrievalMode.HYBRID,
    distance=Distance.COSINE
)

app = FastAPI()

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    process_and_store_chunks(file_path)
    return {"status": "Document uploaded and ingested."}

@app.post("/query")
async def query_doc(query: str = Form(...), k: int = Form(3)):
    # --- Use Groq for query decomposition ---
    subqueries = decompose_query(query)
    enhanced_query = " ; ".join(subqueries)
    docs = contextual_compression(vector_store, enhanced_query, k, llm)
    text_chunks = [doc.page_content for doc in docs if doc.metadata.get("modality") == "text"]
    image_paths = [doc.page_content for doc in docs if doc.metadata.get("modality") == "image"]
    # Multimodal answer synthesis
    answer = multimodal_query(text_chunks, image_paths, enhanced_query)
    return {"answer": answer, "subqueries": subqueries}

query_history = []

@app.get("/history")
async def get_history():
    return {"history": query_history}

# --- Document Management APIs ---
@app.get("/documents")
def list_documents():
    db: Session = SessionLocal()
    try:
        docs = db.query(Document).all()
        return [
            {
                "id": d.id,
                "filename": d.filename,
                "filetype": d.filetype,
                "title": d.title,
                "author": d.author,
                "upload_time": d.upload_time,
                "meta": d.meta,
            }
            for d in docs
        ]
    finally:
        db.close()

@app.get("/documents/{doc_id}")
def get_document(doc_id: int):
    db: Session = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return {
            "id": doc.id,
            "filename": doc.filename,
            "filetype": doc.filetype,
            "title": doc.title,
            "author": doc.author,
            "upload_time": doc.upload_time,
            "meta": doc.meta,
            "status": "uploaded"
        }
    finally:
        db.close()