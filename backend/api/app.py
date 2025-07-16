from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from services.ingestion import process_and_store_chunks, qdrant, clip_embd, sparse_embd, COLLECTION, nomic_embd
from services.retrieval import contextual_compression
from services.llm_service import multimodal_query, llm, decompose_query, classify_query_modality
from services.models import Document, User, engine, Base, UserRead, UserCreate, UserUpdate
from sqlalchemy.orm import Session
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance
from fastapi_users import FastAPIUsers
from fastapi_users.authentication import JWTStrategy, AuthenticationBackend, BearerTransport
from fastapi_users.db import SQLAlchemyUserDatabase
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from pydantic import EmailStr
from fastapi import Depends, Request
from sqlalchemy.orm import Session as SyncSession
import uuid
import os
from sqlalchemy.ext.asyncio import AsyncSession
from services.models import async_session_maker, User
from typing import AsyncGenerator
from sqlalchemy.future import select

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

vector_store = QdrantVectorStore(
    client=qdrant,
    collection_name=COLLECTION,
    embedding=clip_embd,
    sparse_embedding=sparse_embd, 
    retrieval_mode=RetrievalMode.HYBRID,
    distance=Distance.COSINE
)

app = FastAPI()

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

@app.post("/upload")
async def upload_doc(file: UploadFile = File(...)):
    file_path = f"data/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    process_and_store_chunks(file_path)
    return {"status": "Document uploaded and ingested."}

@app.post("/query")
async def query_doc(query: str = Form(...), k: int = Form(3)):
    # --- Use Groq for query decomposition and modality classification ---
    subqueries = decompose_query(query)
    enhanced_query = " ; ".join(subqueries)
    modality = classify_query_modality(query)
    # Select embedding model and named vector
    if modality in ["text", "image"]:
        embedding_model = clip_embd
        vector_name = modality
    elif modality in ["table", "code"]:
        embedding_model = nomic_embd
        vector_name = modality
    else:
        embedding_model = clip_embd
        vector_name = "text"
    # Create a QdrantVectorStore for the correct named vector
    vector_store = QdrantVectorStore(
        client=qdrant,
        collection_name=COLLECTION,
        embedding=embedding_model,
        sparse_embedding=sparse_embd,
        retrieval_mode=RetrievalMode.HYBRID,
        distance=Distance.COSINE,
        vector_name=vector_name
    )
    docs = contextual_compression(vector_store, enhanced_query, k, llm)
    # Filter results by modality for answer synthesis
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
                    "uploaded_by": doc.uploaded_by,
                    "created_at": doc.created_at,
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