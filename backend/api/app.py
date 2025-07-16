from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from services.ingestion import process_and_store_chunks, qdrant, sparse_embd, COLLECTION, nomic_embd, get_text_embedding, get_image_embedding
from services.retrieval import contextual_compression
from services.llm_service import multimodal_query, llm, decompose_query, classify_query_modality
from services.models import Document, User, engine, Base, UserRead, UserCreate, UserUpdate
from sqlalchemy.orm import Session
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy, AuthenticationBackend, BearerTransport
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from pydantic import EmailStr
from fastapi import Depends, Request
from sqlalchemy.orm import Session as SyncSession
import uuid
import os
from sqlalchemy.ext.asyncio import AsyncSession
from services.models import async_session_maker, User
from typing import AsyncGenerator
from sqlalchemy.future import select
from langchain_core.embeddings import Embeddings
import numpy as np
from qdrant_client.http.models import SparseVectorParams
from services.ingestion import NAMED_VECTORS, COLLECTION, qdrant
from fastapi.middleware.cors import CORSMiddleware

class DummyEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [np.zeros(512, dtype=np.float32) for _ in texts]
    def embed_query(self, text):
        return np.zeros(512, dtype=np.float32)

class LangchainEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [get_text_embedding(text) for text in texts]
    def embed_query(self, text: str) -> list[float]:
        return get_text_embedding(text)

SECRET = os.getenv("JWT_SECRET", "SECRET")

# User DB setup
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

async def get_user_db(session: AsyncSession = Depends(get_async_session)) -> AsyncGenerator[SQLAlchemyUserDatabase, None]:
    yield SQLAlchemyUserDatabase(session, User)

from fastapi_users.manager import BaseUserManager, UserManagerDependency

class UserManager(BaseUserManager[User, uuid.UUID]):
    user_db_model = User
    reset_password_token_secret = SECRET
    verification_token_secret = SECRET

    def parse_id(self, value: str) -> uuid.UUID:
        return uuid.UUID(value)

def get_user_manager(user_db=Depends(get_user_db)):
    yield UserManager(user_db)

bearer_transport = BearerTransport(tokenUrl="auth/jwt/login")

def get_jwt_strategy() -> JWTStrategy:
    return JWTStrategy(secret=SECRET, lifetime_seconds=3600)

auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

fastapi_users = FastAPIUsers[
    User,
    uuid.UUID
](
    get_user_manager,
    [auth_backend],
)

current_active_user = fastapi_users.current_user(active=True)

# Remove global vector_store initialization
# vector_store = QdrantVectorStore(
#     client=qdrant,
#     collection_name=COLLECTION,
#     embedding=DummyEmbeddings(),
#     sparse_embedding=sparse_embd, 
#     retrieval_mode=RetrievalMode.HYBRID,
#     distance=Distance.COSINE,
#     vector_name="text",
# )

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # React default
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/auth/jwt",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
    prefix="/auth",
    tags=["auth"],
)
app.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)

@app.on_event("startup")
def ensure_qdrant_collection():
    SPARSE_VECTORS = {
        "text_sparse": SparseVectorParams()
    }
    try:
        qdrant.create_collection(
            collection_name=COLLECTION,
            vectors_config=NAMED_VECTORS,
            sparse_vectors_config=SPARSE_VECTORS
        )
    except Exception as e:
        print(f"Qdrant collection creation error: {e}")

# Utility function to get a QdrantVectorStore for a given vector name

def get_vector_store(vector_name="text"):
    return QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION,
        embedding=LangchainEmbeddings(),  # Use real embedding
        sparse_embedding=sparse_embd,
        retrieval_mode=RetrievalMode.HYBRID,
        distance=Distance.COSINE,
        vector_name=vector_name,
        sparse_vector_name="text_sparse"
    )

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    await process_and_store_chunks(file_path)  # process_and_store_chunks is async, so await is correct
    return {"status": "Document uploaded and ingested."}

@app.post("/query")
async def query_doc(query: str = Form(...), k: int = Form(3)):
    subqueries = decompose_query(query)
    enhanced_query = " ; ".join(subqueries)
    modality = classify_query_modality(query)
    if modality in ["text", "image"]:
        embedding_model = get_text_embedding
        vector_name = modality
    elif modality in ["table", "code"]:
        embedding_model = nomic_embd
        vector_name = modality
    else:
        embedding_model = get_text_embedding
        vector_name = "text"
    vector_store = get_vector_store(vector_name=vector_name)
    docs = await contextual_compression(vector_store, enhanced_query, k, llm)  # contextual_compression is async, so await is correct
    print("\n\nthese are the docs that i am recieving: ", docs, "\n\n")
    text_chunks = [doc.page_content for doc in docs if doc.metadata.get("modality") == "text"]
    image_paths = [doc.page_content for doc in docs if doc.metadata.get("modality") == "image"]
    answer = multimodal_query(text_chunks, image_paths, enhanced_query)
    return {"answer": answer, "subqueries": subqueries, "modality": modality, "results": [doc.page_content for doc in docs]}

query_history = []

@app.get("/history")
async def get_history():
    return {"history": query_history}

@app.get("/me", response_model=UserRead)
async def get_me(user: User = Depends(current_active_user)):
    return user

# --- Document Management APIs ---
@app.get("/documents", dependencies=[Depends(current_active_user)])
async def list_documents():
    async with async_session_maker() as session:
        try:
            result = await session.execute(select(Document))
            docs = result.scalars().all()
            return [
                 {
                    "id": doc.id,
                    "filename": doc.filename,
                    "filetype": doc.filetype,
                    "title": doc.title,
                    "author": doc.author,
                    "upload_time": doc.upload_time,
                    "meta": doc.meta,
                    "content": doc.content,
                 }
                for doc in docs
            ]
        finally:
            await session.close()

@app.get("/documents/{doc_id}", dependencies=[Depends(current_active_user)])
async def get_document(doc_id: int):
    async with async_session_maker() as session:
        try:
            result = await session.execute(select(Document).where(Document.id == doc_id))
            doc = result.scalars().first()
            if not doc:
                raise HTTPException(status_code=404, detail="Document not found")
            return {
                "id": doc.id,
                "filename": doc.filename,
                "uploaded_by": doc.uploaded_by,
                "created_at": doc.created_at,
            }
        finally:
            await session.close()

@app.post("/recreate_collection")
async def recreate_collection():
    # Define dense and sparse vector configs
    from services.ingestion import NAMED_VECTORS, COLLECTION, qdrant
    SPARSE_VECTORS = {
        "text_sparse": SparseVectorParams()
    }
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=NAMED_VECTORS,
        sparse_vectors_config=SPARSE_VECTORS
    )
    return {"status": "Collection recreated with dense and sparse support."}