from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import datetime
import os
from dotenv import load_dotenv
from fastapi_users_db_sqlalchemy import SQLAlchemyBaseUserTableUUID
from pydantic import EmailStr
from fastapi_users import schemas
import uuid
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

load_dotenv()

Base = declarative_base()

class Document(Base):
    __tablename__ = 'documents'
    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    filetype = Column(String, nullable=False)
    title = Column(String)
    author = Column(String)
    upload_time = Column(DateTime, default=datetime.datetime.utcnow)
    meta = Column(JSON)  # Renamed from 'metadata' to 'meta'
    # Relationship to chunks
    chunks = relationship('DocumentChunk', back_populates='document')

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    chunk_idx = Column(Integer)
    content = Column(Text)
    modality = Column(String)  # e.g., 'text', 'image', etc.
    meta = Column(JSON)  # Renamed from 'metadata' to 'meta'
    # Relationship back to document
    document = relationship('Document', back_populates='chunks')

class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = 'users'
    # id, email, hashed_password, is_active, is_superuser, is_verified are provided by base

class UserRead(schemas.BaseUser[uuid.UUID]):
    pass

class UserCreate(schemas.BaseUserCreate):
    pass

class UserUpdate(schemas.BaseUserUpdate):
    pass

# --- Database setup ---
DATABASE_URL = os.getenv("DATABASE_URL")  # Update as needed
if DATABASE_URL:
    engine = create_async_engine(DATABASE_URL, echo=True)
    async_session_maker = async_sessionmaker(engine, expire_on_commit=False)
else:
    print("database initialization miserably :C")

import asyncio

async def async_init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

def init_db():
    asyncio.run(async_init_db())