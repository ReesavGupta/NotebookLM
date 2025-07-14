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

COLLECTION = "docling_chunks"
VECTOR_SIZE = 1024
model_name = "ViT-g-14"
checkpoint = "laion2b_s34b_b88k"

QDRANT_HOST = os.getenv("QDRANT_HOST", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)

model, _, preprocess = open_clip.create_model_and_transforms(model_name, checkpoint)
tokenizer = open_clip.get_tokenizer(model_name)
clip_embd = OpenCLIPEmbeddings(
    model_name=model_name,
    checkpoint=checkpoint,
    preprocess=preprocess,
    model=model,
    tokenizer=tokenizer
)
sparse_embd = FastEmbedSparse(model_name="Qdrant/bm25")

qdrant = QdrantClient(
    url=QDRANT_HOST,
    api_key=QDRANT_API_KEY
)

if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )

# ---- INGESTION ----
def process_and_store_chunks(file_path: str):
    # Chunk the document (text, images, tables, etc.)
    hybrid_chunker = HybridChunker()
    loader = DoclingLoader(file_path=file_path, chunker=hybrid_chunker,export_type=ExportType.DOC_CHUNKS)
    loaded_chunks = loader.load()
    doc_id = np.random.randint(int(1e6))
    
    docs = []
    for idx, chunk in enumerate(loaded_chunks):
        meta = dict(chunk.metadata)
        meta["doc_id"] = doc_id
        meta["chunk_idx"] = idx
        meta["text_preview"] = getattr(chunk, "page_content", "")[:200]
        blob = getattr(chunk, "blob", None)  # Only Docling chunks have .blob
        if blob:
            # Image chunk (from Docling)
            img_path = f"temp_img_{doc_id}_{idx}.png"
            with open(img_path, "wb") as f:
                f.write(blob)
            meta["image_path"] = img_path
            meta["modality"] = "image"
            docs.append(Document(page_content=img_path, metadata=meta))
        else:
            # Text chunk (from Docling)
            meta["modality"] = "text"
            docs.append(Document(page_content=getattr(chunk, "page_content", ""), metadata=meta))
    print(f"Prepared {len(docs)} chunks for upsert.")

    # Store with hybrid mode (dense + sparse)
    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=clip_embd,
        sparse_embedding=sparse_embd,
        client=qdrant,
        collection_name=COLLECTION,
        retrieval_mode=RetrievalMode.HYBRID,
        distance=Distance.COSINE
    )
    print(f"Stored {len(docs)} chunks in Qdrant.")
    return vector_store
# ---- MAIN ----
if __name__ == "__main__":
    file_path = "PATH_TO_YOUR_DOC.pdf"  # <-- change this!
    vector_store = process_and_store_chunks(file_path)
    # Example hybrid search
    # hybrid_search(vector_store, "What is shown in the diagram?", k=3)
    # hybrid_search(vector_store, "Summarize the introduction section.", k=3)
