import os
import numpy as np
import clip
import torch
from PIL import Image
from langchain_docling.loader import DoclingLoader, ExportType
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from services.models import async_session_maker, Document, DocumentChunk
from langchain_community.document_loaders import NotebookLoader
from langchain_nomic import NomicEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession
import asyncio
import torchvision.transforms as T

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
    text_tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        return model.encode_text(text_tokens).cpu().numpy()[0]

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
    try:
        pass
    finally:
        pass
    print(f"Prepared {len(docs)} chunks for upsert.")
    # --- Store with Qdrant named vectors ---
    points = []
    for idx, doc in enumerate(docs):
        modality = doc.metadata.get("modality", "text")
        vectors = {}
        if modality == "text":
            vectors["text"] = get_text_embedding(doc.page_content)
        elif modality == "image":
            vectors["image"] = get_image_embedding(doc.page_content)
        elif modality == "table":
            vectors["table"] = nomic_embd.embed_query(doc.page_content)
        elif modality == "code":
            vectors["code"] = nomic_embd.embed_query(doc.page_content)
        else:
            vectors["text"] = get_text_embedding(doc.page_content)
        points.append({
            "id": f"{os.path.basename(file_path)}_{idx}",
            "payload": doc.metadata,
            "vectors": vectors
        })
    qdrant.upsert(
        collection_name=COLLECTION,
        points=points
    )
    print(f"Stored {len(points)} chunks in Qdrant with named vectors.")
    return None

async def save_metadata_to_db(file_path, docs):
    async with async_session_maker() as session:
        document = Document(
            filename=os.path.basename(file_path),
        )
        session.add(document)
        await session.commit()
        await session.refresh(document)
        for idx, doc in enumerate(docs):
            chunk_obj = DocumentChunk(
                document_id=document.id,
                chunk_idx=idx,
                content=doc.page_content,
                modality=doc.metadata.get("modality", "text"),
                meta=doc.metadata
            )
            session.add(chunk_obj)
        await session.commit()

if __name__ == "__main__":
    file_path = "PATH_TO_YOUR_DOC.pdf"  # <-- change this!
    vector_store = process_and_store_chunks(file_path)
