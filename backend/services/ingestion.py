import os
import numpy as np
import open_clip
from langchain_docling.loader import DoclingLoader, ExportType
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from services.models import Document, DocumentChunk, SessionLocal
from langchain_community.document_loaders import NotebookLoader
from langchain_nomic import NomicEmbeddings

COLLECTION = "docling_chunks"
VECTOR_SIZE = 1024
model_name = "ViT-g-14"
checkpoint = "laion2b_s34b_b88k"

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

model, _, preprocess = open_clip.create_model_and_transforms(model_name, checkpoint)
tokenizer = open_clip.get_tokenizer(model_name)
# --- Embedding models ---
clip_embd = OpenCLIPEmbeddings(
    model_name=model_name,
    checkpoint=checkpoint,
    preprocess=preprocess,
    model=model,
    tokenizer=tokenizer
)
sparse_embd = FastEmbedSparse(model_name="Qdrant/bm25")
# Nomic Embeddings for table and code
nomic_api_key = os.getenv("NOMIC_API_KEY")
nomic_embd = NomicEmbeddings(
    model="nomic-embed-text-v1.5"
)

qdrant = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

# --- Qdrant named vectors config ---
NAMED_VECTORS = {
    "text": VectorParams(size=1024, distance=Distance.COSINE),   # CLIP text
    "image": VectorParams(size=1024, distance=Distance.COSINE),  # CLIP image
    "table": VectorParams(size=768, distance=Distance.COSINE),   # Nomic
    "code": VectorParams(size=768, distance=Distance.COSINE),    # Nomic
}

if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=NAMED_VECTORS
    )

# ---- INGESTION ----
def process_and_store_chunks(file_path: str):
    # Detect file type
    ext = os.path.splitext(file_path)[1].lower()
    docs = []
    if ext == ".ipynb":
        # Use NotebookLoader for Jupyter notebooks
        loader = NotebookLoader(
            file_path,
            include_outputs=True,
            max_output_length=50,
            remove_newline=True,
        )
        loaded_chunks = loader.load()
        # Wrap in list if not already
        if not isinstance(loaded_chunks, list):
            loaded_chunks = [loaded_chunks]
        # For consistency, add modality and meta fields
        for idx, chunk in enumerate(loaded_chunks):
            meta = dict(chunk.metadata) if hasattr(chunk, "metadata") else {}
            meta["modality"] = "notebook"
            meta["chunk_idx"] = idx
            meta["text_preview"] = chunk.page_content[:200]
            docs.append(Document(page_content=chunk.page_content, metadata=meta))
    else:
        # Chunk the document (text, images, tables, etc.)
        hybrid_chunker = HybridChunker()
        loader = DoclingLoader(file_path=file_path, chunker=hybrid_chunker, export_type=ExportType.DOC_CHUNKS)
        loaded_chunks = loader.load()
        doc_id = np.random.randint(int(1e6))
        for idx, chunk in enumerate(loaded_chunks):
            meta = dict(chunk.metadata)
            meta["doc_id"] = doc_id
            meta["chunk_idx"] = idx
            meta["text_preview"] = getattr(chunk, "page_content", "")[:200]
            # --- Robust modality detection ---
            modality = None
            # Prefer explicit label from doc_items if available
            doc_items = meta.get("doc_items")
            if doc_items and isinstance(doc_items, list) and len(doc_items) > 0:
                label = doc_items[0].get("label")
                if label:
                    modality = label.lower()  # e.g., 'text', 'table', 'code', 'image', etc.
            # Fallback to blob detection for images
            blob = getattr(chunk, "blob", None)
            if not modality:
                if blob:
                    modality = "image"
                else:
                    modality = "text"
            meta["modality"] = modality
            # --- Routing logic for future extensibility ---
            if modality == "image":
                if blob is not None:
                    img_path = f"temp_img_{doc_id}_{idx}.png"
                    with open(img_path, "wb") as f:
                        f.write(blob)
                    meta["image_path"] = img_path
                    docs.append(Document(page_content=img_path, metadata=meta))
                else:
                    # If no blob, skip this chunk
                    continue
            elif modality == "text":
                docs.append(Document(page_content=getattr(chunk, "page_content", ""), metadata=meta))
            elif modality == "table":
                # Table chunk: for now treat as text, but ready for table embedding
                docs.append(Document(page_content=getattr(chunk, "page_content", ""), metadata=meta))
            elif modality == "code":
                # Code chunk: for now treat as text, but ready for code embedding
                docs.append(Document(page_content=getattr(chunk, "page_content", ""), metadata=meta))
            else:
                # Default fallback
                docs.append(Document(page_content=getattr(chunk, "page_content", ""), metadata=meta))
    # --- Save Document metadata to DB ---
    db = SessionLocal()
    try:
        filename = os.path.basename(file_path)
        filetype = ext[1:] if ext.startswith(".") else ext
        document = Document(
            filename=filename,
            filetype=filetype,
            title=None,  # You can extract from metadata if available
            author=None, # You can extract from metadata if available
            meta={},     # Add more metadata if needed
        )
        db.add(document)
        db.commit()
        db.refresh(document)
        for idx, doc in enumerate(docs):
            chunk_obj = DocumentChunk(
                document_id=document.id,
                chunk_idx=idx,
                content=doc.page_content,
                modality=doc.metadata.get("modality", "text"),
                meta=doc.metadata
            )
            db.add(chunk_obj)
        db.commit()
        print(f"Saved document and {len(docs)} chunks to DB.")
    finally:
        db.close()
    print(f"Prepared {len(docs)} chunks for upsert.")
    # --- Store with Qdrant named vectors ---
    points = []
    for idx, doc in enumerate(docs):
        modality = doc.metadata.get("modality", "text")
        vectors = {}
        if modality == "text":
            vectors["text"] = clip_embd.embed_query(doc.page_content)
        elif modality == "image":
            vectors["image"] = clip_embd.embed_query(doc.page_content)
        elif modality == "table":
            vectors["table"] = nomic_embd.embed_query(doc.page_content)
        elif modality == "code":
            vectors["code"] = nomic_embd.embed_query(doc.page_content)
        else:
            # Default to text
            vectors["text"] = clip_embd.embed_query(doc.page_content)
        points.append({
            "id": f"{filename}_{idx}",
            "payload": doc.metadata,
            "vectors": vectors
        })
    qdrant.upsert(
        collection_name=COLLECTION,
        points=points
    )
    print(f"Stored {len(points)} chunks in Qdrant with named vectors.")
    return None
# ---- MAIN ----
if __name__ == "__main__":
    file_path = "PATH_TO_YOUR_DOC.pdf"  # <-- change this!
    vector_store = process_and_store_chunks(file_path)
    # Example hybrid search
    # hybrid_search(vector_store, "What is shown in the diagram?", k=3)
    # hybrid_search(vector_store, "Summarize the introduction section.", k=3)
