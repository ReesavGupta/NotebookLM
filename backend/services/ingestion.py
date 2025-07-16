import os
import numpy as np
import clip
import torch
import uuid
from PIL import Image
from langchain_docling.loader import DoclingLoader, ExportType
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from services.models import async_session_maker, Document, DocumentChunk
from langchain_community.document_loaders import NotebookLoader
from langchain_nomic import NomicEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import torchvision.transforms as T
import datetime

COLLECTION = "docling_chunks"
device = "cuda"  # or "cuda" if you have a GPU
model, preprocess = clip.load("ViT-B/32", device=device)

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

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
    "text": VectorParams(size=512, distance=Distance.COSINE),   # CLIP text
    "image": VectorParams(size=512, distance=Distance.COSINE),  # CLIP image
    "table": VectorParams(size=768, distance=Distance.COSINE),   # Nomic
    "code": VectorParams(size=768, distance=Distance.COSINE),    # Nomic
}

if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=NAMED_VECTORS
    )

def get_text_embedding(text):
    # Binary search for the longest prefix that fits in 77 tokens for CLIP
    max_tokens = 77
    left, right = 0, len(text)
    best = ""
    while left <= right:
        mid = (left + right) // 2
        candidate = text[:mid]
        try:
            tokens = clip.tokenize([candidate])
            if tokens.shape[1] <= max_tokens:
                best = candidate
                left = mid + 1
            else:
                right = mid - 1
        except RuntimeError:
            right = mid - 1
    tokens = clip.tokenize([best])
    tokens = tokens.to(device)
    with torch.no_grad():
        return model.encode_text(tokens).cpu().numpy()[0]

# --- CHUNKING CONFIG SUGGESTION ---
# For CLIP: aim for ~200-250 chars per chunk.
# For transformers: aim for ~1000-1500 chars per chunk, but always truncate to 512 tokens before embedding.

def get_image_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    image = preprocess(img)
    # Ensure image is a torch.Tensor and add batch dimension
    if isinstance(image, Image.Image):
        import torchvision.transforms as T
        image = T.ToTensor()(image)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        return model.encode_image(image).cpu().numpy()[0]

# ---- INGESTION ----
async def process_and_store_chunks(file_path: str):
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
            docs.append(Document(content=chunk.page_content, meta=meta))
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
                    docs.append(Document(content=img_path, meta=meta))
                else:
                    # If no blob, skip this chunk
                    continue
            elif modality == "text":
                docs.append(Document(content=getattr(chunk, "page_content", ""), meta=meta))
            elif modality == "table":
                # Table chunk: for now treat as text, but ready for table embedding
                docs.append(Document(content=getattr(chunk, "page_content", ""), meta=meta))
            elif modality == "code":
                # Code chunk: for now treat as text, but ready for code embedding
                docs.append(Document(content=getattr(chunk, "page_content", ""), meta=meta))
            else:
                # Default fallback
                docs.append(Document(content=getattr(chunk, "page_content", ""), meta=meta))
    # --- Save Document metadata to DB ---
    try:
        await save_metadata_to_db(file_path, docs)
    finally:
        pass
    print(f"Prepared {len(docs)} chunks for upsert.")
    # --- Store with Qdrant named vectors ---
    points = []
    for idx, doc in enumerate(docs):
        modality = doc.meta.get("modality", "text")
        vector = {}
        if modality == "text":
            vector["text"] = np.array(get_text_embedding(doc.content), dtype=np.float32)
        elif modality == "image":
            vector["image"] = np.array(get_image_embedding(doc.content), dtype=np.float32)
        elif modality in {"table", "code"}:
            vector[modality] = np.array(nomic_embd.embed_query(doc.content), dtype=np.float32)
        else:
            vector["text"] = np.array(get_text_embedding(doc.content), dtype=np.float32)
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            payload=doc.meta,
            vector=vector
        ))
    qdrant.upload_points(
        collection_name=COLLECTION,
        points=points,
        batch_size=8
    )
    print(f"Stored {len(points)} chunks in Qdrant with named vectors.")
    return None

async def save_metadata_to_db(file_path, docs):
    async with async_session_maker() as session:
        document = Document(
            filename=os.path.basename(file_path),
            filetype=os.path.splitext(file_path)[1][1:] or "unknown",  # e.g., 'pdf'
            upload_time=datetime.datetime.utcnow(),
            meta={"num_chunks": len(docs)},
            content="\n\n".join([doc.content for doc in docs if hasattr(doc, "content")])
        )
        session.add(document)
        await session.commit()

if __name__ == "__main__":
    file_path = "PATH_TO_YOUR_DOC.pdf"  # <-- change this!
    # vector_store = await process_and_store_chunks(file_path)  # Only use await in async context
    # To run this, use: asyncio.run(process_and_store_chunks(file_path)) in an async context
