from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
import datetime
import os
from dotenv import load_dotenv

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

# --- Database setup ---
DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://user:password@localhost:5432/notebookllm")
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine) 