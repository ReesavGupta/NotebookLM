# Notebook LLM: Complete Multimodal Research Assistant

## Objective

Notebook LLM is a production-ready, multimodal Retrieval-Augmented Generation (RAG) system designed to ingest, process, and reason over complex documents containing text, images, tables, charts, and code snippets. Inspired by NotebookLM, it extends capabilities to support advanced multimodal understanding, intelligent document structure, and robust research workflows.

---

## Features

- **Comprehensive Document Processing:**
  - Supports 10+ file formats: PDF, DOCX, HTML, CSV, Excel, PowerPoint, Jupyter notebooks, and image files (PNG, JPG, etc.)
- **Advanced Multimodal Understanding:**
  - Processes and relates text, images, tables, charts, and code within and across documents
- **Intelligent Document Structure:**
  - Maintains hierarchy (sections, subsections) and relationships between document elements
- **Advanced Query Capabilities:**
  - Handles complex, cross-modal queries requiring reasoning and synthesis
- **Production-Grade Features:**
  - User authentication, document management, query history, and real-time collaboration
- **Custom Embedding Strategy:**
  - Domain-specific embeddings for technical and multimodal content
- **Export & Integration:**
  - Export insights and integrate with external tools
- **Smart Summarization:**
  - Generates executive summaries across multiple documents
- **Relationship Mapping:**
  - Identifies and visualizes connections between concepts

---

## System Architecture

- **Backend:** FastAPI (Python) for API, document processing, and orchestration
- **Frontend:** React (with Vite) for a modern, real-time user interface
- **Document Processing:** Custom pipeline using Unstructured, Docling, and specialized parsers
- **Multimodal AI:** Integrates vision models (e.g., GPT-4V, Claude Vision) for image and chart understanding
- **Vector Database:** Qdrant for hybrid search (dense + sparse), metadata filtering, and scalable retrieval
- **Embeddings:** Custom pipeline using CLIP, Nomic, and transformer models for text, code, and images
- **Authentication:** FastAPI Users for secure user management

---

## Design Choices & Rationale

### 1. **Multimodal Document Processing**
- **Why?** Real-world research documents are rarely unimodal. By supporting text, images, tables, and code, the system can answer richer, more complex queries.
- **How?**
  - Uses Unstructured and Docling for robust parsing of diverse formats
  - Custom chunking logic ensures context windows fit model/tokenizer limits (e.g., CLIP's 77-token limit)
  - Maintains document hierarchy and cross-references for context-aware retrieval

### 2. **Custom Embedding Strategy**
- **Why?** Off-the-shelf embeddings often underperform on technical, code-heavy, or multimodal content.
- **How?**
  - Text: Transformer-based models fine-tuned for technical domains
  - Images: CLIP and vision models for semantic image embeddings
  - Code: Code-specific models (e.g., CodeBERT)
  - Hybrid: Combines dense (neural) and sparse (BM25) vectors for best-in-class retrieval

### 3. **Vector Database: Qdrant**
- **Why?** Qdrant supports named vectors, hybrid search, and fast metadata filtering, which are essential for multimodal and production-scale RAG.
- **How?**
  - Each document chunk is stored with both dense and sparse vectors
  - Metadata (e.g., section, filetype, author) enables fine-grained filtering
  - Batched upserts and robust collection management for reliability

### 4. **Query Decomposition & Reasoning**
- **Why?** Complex queries often require breaking down into sub-questions and aggregating results.
- **How?**
  - Implements query decomposition pipeline
  - Uses LLMs to orchestrate sub-query generation and answer synthesis

### 5. **Production-Ready Backend**
- **Why?** Reliability, security, and extensibility are critical for real-world use.
- **How?**
  - FastAPI for async, scalable APIs
  - FastAPI Users for authentication and user management
  - Alembic for database migrations
  - Modular service structure for easy extension

---

## Setup Instructions

### Prerequisites
- Python 3.9+
- Node.js 18+
- Qdrant (Docker recommended)
- (Optional) GPU for faster embedding/model inference

### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
# Set up environment variables (see .env.example)
# Run Alembic migrations
alembic upgrade head
# Start FastAPI server
uvicorn api.app:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

### Qdrant Setup (Docker)
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

---

## Usage

1. **Login/Register** via the frontend
2. **Upload Documents** (PDF, DOCX, images, etc.)
3. **Ingest & Process**: Backend parses, chunks, and embeds content
4. **Query**: Ask questions; system retrieves relevant chunks and generates answers
5. **Export**: Download insights or integrate with external tools

---

## Extensibility
- **Add new file formats:** Implement a parser and register in the ingestion pipeline
- **Add new embedding models:** Extend the embedding service and update vector schema
- **Integrate new LLMs/vision models:** Plug into the multimodal pipeline
- **Custom UI features:** Extend React components or add Streamlit apps

---

## Technical Documentation

- **Backend:** See `backend/api/app.py` and `backend/services/`
- **Frontend:** See `frontend/src/`
- **Database Models:** See `backend/services/models.py`
- **Document Ingestion:** See `backend/services/ingestion.py`
- **Vector Store Logic:** See `backend/services/retrieval.py`
- **Embeddings:** See `backend/services/llm_service.py`

---

## Sample Dataset
- Research papers with embedded images/charts
- Technical manuals
- Financial reports
- Code documentation
- Presentation slides

---

## Contribution Guidelines

1. Fork the repository
2. Create a new branch (`feature/your-feature`)
3. Commit your changes with clear messages
4. Submit a pull request with a detailed description

---

## License

MIT License

---

## Acknowledgements
- [LangChain](https://github.com/langchain-ai/langchain)
- [Qdrant](https://qdrant.tech/)
- [Unstructured](https://github.com/Unstructured-IO/unstructured)
- [CLIP](https://github.com/openai/CLIP)
- [Docling](https://github.com/docling/docling)
- [FastAPI](https://fastapi.tiangolo.com/)

---

## Contact
For questions or support, please open an issue or contact the maintainer. 