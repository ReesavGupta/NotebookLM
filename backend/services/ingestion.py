import os
import open_clip
import numpy as np
from langchain_docling.loader import DoclingLoader, ExportType
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_experimental.open_clip import OpenCLIPEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

COLLECTION = "docling_chunks"
VECTOR_SIZE = 1024

model_name = "ViT-g-14"
checkpoint = "laion2b_s34b_b88k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, checkpoint)
tokenizer = open_clip.get_tokenizer(model_name)

clip_embd = OpenCLIPEmbeddings(model_name=model_name, checkpoint=checkpoint, preprocess=preprocess ,model= model, tokenizer=tokenizer)

qdrant = QdrantClient(
    url= os.getenv("QDRANT_HOST"), 
    api_key= os.getenv("QDRANT_API_KEY")
)

if COLLECTION not in [c.name for c in qdrant.get_collections().collections]:
    qdrant.create_collection(collection_name=COLLECTION, vectors_config=VectorParams(size=VECTOR_SIZE  ,distance=Distance.COSINE))
    

def process_and_store_chunks(chunks, doc_id):
    points= []
    for idx, chunk in enumerate(chunks):
        meta = dict(chunk.metadata)        
        meta["doc_id"] = doc_id
        meta["chunk_idx"] = idx
        meta["text_preview"] = chunk.page_content[:200] if hasattr(chunk, "page_content") else ""

        modality = meta.get("modality", "text")

        if hasattr(chunk, "blob") and chunk.blob:
            img_path = f"temp_img_{doc_id}_{idx}.png"
            with open(img_path, "wb") as f:
                f.write(chunk.blob)
            meta["image_path"] = img_path
            meta["modality"] = "image"
            embedding = clip_embd.embed_image([img_path])[0]
        else:
            meta["modality"] = "text"
            embedding = clip_embd.embed_documents([chunk.page_content])[0]

        points.append(
            PointStruct(
                id=int(f"{doc_id}{idx}"),
                vector=embedding,
                payload=meta
            )
        )
    qdrant.upsert(collection_name=COLLECTION, points=points)
    print(f"Stored {len(points)} chunks in Qdrant.")


def process_document(file_path: str):
    hybrid_chunker = HybridChunker()
    loader = DoclingLoader(file_path=file_path, chunker= hybrid_chunker, export_type=ExportType.DOC_CHUNKS)    
    loaded_docs = loader.load()
    doc_id = np.random.randint(int(1e6))
    process_and_store_chunks(loaded_docs, doc_id)
    return {"status": "processed", "chunks": len(loaded_docs), "doc_id": doc_id}
    
if __name__ == "__main__":
    process_document("C:/Users/REESAV/Desktop/misogi-assignments/notebookLLM/backend/data/doc.pdf")
