# AI-Powered Research Assistant Using LangChain & Mistral AI

A production-grade intelligent research assistant that uses Retrieval Augmented Generation (RAG) architecture to retrieve information from multiple documents and generate accurate answers.

## 🔬 Overview

This application implements a complete RAG pipeline that:

- Ingests documents (PDF, TXT, MD, HTML) and web URLs
- Processes documents into semantic chunks
- Generates embeddings using Mistral AI
- Stores embeddings in ChromaDB vector database
- Retrieves relevant context through semantic search
- Generates answers using Mistral AI LLM
- Provides document summarization capabilities

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Documents     │────▶│   RAG Pipeline  │────▶│   ChromaDB      │
│ (PDF, URL, TXT)│     │                 │     │   Vector Store  │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │   FastAPI       │◀────│   Mistral AI    │
                        │   Backend + UI  │     │   (LLM + Embed) │
                        └─────────────────┘     └─────────────────┘
```

## 🚀 Features

### Document Processing
- **Multiple Format Support**: PDF, TXT, MD, HTML files
- **URL Fetching**: Load content from web pages
- **Smart Chunking**: Configurable text chunking with overlap
- **Metadata Tracking**: Source, title, timestamps

### Search & Retrieval
- **Semantic Search**: Vector-based similarity search
- **MMR Support**: Maximum Marginal Relevance for diverse results
- **Source Filtering**: Filter by document source
- **Configurable Top-K**: Retrieve top N relevant chunks

### Question Answering
- **RAG Architecture**: Context-aware AI responses
- **Source Citations**: Track answer sources
- **Confidence Scoring**: Show answer confidence
- **Streaming Support**: Real-time response generation

### Summarization
- **Multiple Types**: Short, bullet, detailed, executive summaries
- **Document-level**: Summarize individual documents
- **Cross-document**: Generate research summaries

## 📋 Requirements

- Python 3.11+
- Mistral API Key ([Get from console.mistral.ai](https://console.mistral.ai/))
- Docker & Docker Compose (optional)

## 🛠️ Installation

### Option 1: Local Development

1. **Clone and Navigate**
```bash
cd /Users/kundanvarma/Research Assistant
```

2. **Create Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your Mistral API key
```

5. **Run the API Server**
```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

6. **Access the Application**
Open your browser and navigate to `http://localhost:8000`

### Option 2: Docker Deployment

1. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your Mistral API key
```

2. **Build and Run**
```bash
docker-compose up --build
```

3. **Access the Application**
3. **Access the Application**
- Unified Interface: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MISTRAL_API_KEY` | Mistral AI API key | Required |
| `CHROMA_DB_PATH` | ChromaDB storage path | `./database/chroma_db` |
| `EMBEDDING_MODEL` | Embedding model | `mistral-embed` |
| `LLM_MODEL` | LLM model | `mistral-large-latest` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `TOP_K_RESULTS` | Default K for retrieval | `5` |
| `MAX_FILE_SIZE_MB` | Max upload size | `50` |

## 📡 API Endpoints

### Health Check
```bash
GET /health
```

### Upload Document
```bash
POST /api/upload
Content-Type: multipart/form-data
Body: file (PDF, TXT, MD, HTML)
```

### Upload from URL
```bash
POST /api/upload/url?source_url=<url>
```

### Query
```bash
POST /api/query
Content-Type: application/json
Body: {
  "query": "Your research question",
  "top_k": 5,
  "return_sources": true
}
```

### Generate Summary
```bash
POST /api/summary
Content-Type: application/json
Body: {
  "source": "document_source or null for all",
  "summary_type": "short|bullet|detailed|executive"
}
```

### List Documents
```bash
GET /api/documents
```

### Delete Document
```bash
DELETE /api/documents
Content-Type: application/json
Body: { "source": "document_source" }
```

### Statistics
```bash
GET /api/stats
```

## 📁 Project Structure

```
project/
├── app/
│   ├── api/
│   │   └── main.py          # FastAPI application
│   ├── models/
│   │   └── schemas.py       # Pydantic models
│   └── rag/
│       ├── document_loader.py    # Document loading
│       ├── text_chunker.py       # Text chunking
│       ├── embedding_generator.py # Embeddings
│       ├── vector_store.py       # ChromaDB storage
│       ├── retriever.py          # Semantic search
│       ├── qa_chain.py           # Question answering
│       └── summarizer.py         # Document summarization
├── config/
│   └── settings.py          # Configuration
├── frontend/
│   └── web/                 # Modern JS Frontend
├── database/
│   └── chroma_db/          # Vector database
├── .env.example            # Environment template
├── Dockerfile              # Docker build
├── docker-compose.yml      # Docker compose
├── requirements.txt        # Dependencies
└── README.md               # This file
```

## 🔧 Usage Examples

### Python API Usage

```python
import requests

API_URL = "http://localhost:8000"

# Upload a document
with open("research.pdf", "rb") as f:
    files = {"file": ("research.pdf", f, "application/pdf")}
    response = requests.post(f"{API_URL}/api/upload", files=files)

# Query
response = requests.post(
    f"{API_URL}/api/query",
    json={"query": "What are the main findings?", "top_k": 5}
)
result = response.json()
print(result["answer"])

# Generate summary
response = requests.post(
    f"{API_URL}/api/summary",
    json={"summary_type": "executive"}
)
```

### Direct Python Usage

```python
from app.rag import (
    DocumentLoader,
    TextChunker,
    VectorStore,
    Retriever,
    QAChain,
    Summarizer,
)
from config.settings import settings

# Initialize components
loader = DocumentLoader()
chunker = TextChunker()
vector_store = VectorStore(api_key=settings.mistral_api_key)
retriever = Retriever(vector_store)
qa = QAChain(retriever, api_key=settings.mistral_api_key)
summarizer = Summarizer(retriever, api_key=settings.mistral_api_key)

# Load and process documents
documents = loader.load_file("document.pdf")
chunks = chunker.chunk_documents(documents)
vector_store.add_documents(chunks)

# Query
response = qa.answer("Your question here")
print(response.answer)

# Summarize
summary = summarizer.summarize_all_documents()
print(summary.summary)
```

## 🧪 Testing

```bash
# Run with test queries
python -c "
from app.api.main import app
# Test your setup
"
```

## 📝 License

MIT License

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [Mistral AI](https://mistral.ai/) - LLM and embeddings
- [ChromaDB](https://www.trychroma.com/) - Vector database

## 📞 Support

For issues and questions:
1. Check the API documentation at `/docs`
2. Review logs in the console
3. Ensure Mistral API key is valid and has sufficient quota
