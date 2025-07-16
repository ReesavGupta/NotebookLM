import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ingestion import NAMED_VECTORS, COLLECTION, qdrant
from qdrant_client.http.models import VectorParams, SparseVectorParams

if __name__ == "__main__":
    from time import sleep
    print("Collections before:", qdrant.get_collections())
    try:
        qdrant.delete_collection(collection_name=COLLECTION)
        print(f"Deleted collection '{COLLECTION}'.")
    except Exception as e:
        print(f"Delete failed (may not exist): {e}")
    sleep(2)  # Wait for deletion to propagate
    SPARSE_VECTORS = {
        "text_sparse": SparseVectorParams()
    }
    print(f"Recreating Qdrant collection '{COLLECTION}' with dense and sparse support...")
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=NAMED_VECTORS,
        sparse_vectors_config=SPARSE_VECTORS
    )
    print("Collections after:", qdrant.get_collections())
    print("Collection recreated successfully.") 