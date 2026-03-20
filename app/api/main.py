"""
AI Research Assistant - FastAPI Backend
Production-grade API for research assistant functionality
"""

import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import aiofiles
from fastapi import FastAPI, File, HTTPException, UploadFile, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from app.models import (
    DeleteRequest,
    DeleteResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievalSource,
    SourceInfo,
    StatsResponse,
    UploadResponse
)
from app.services.research_service import ResearchService
from config.settings import settings

# Logger configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Global service state
research_service: Optional[ResearchService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the Research Engine."""
    global research_service
    logger.info("Initializing Nexus Synthesis Engine...")
    settings.ensure_directories()
    try:
        research_service = ResearchService()
        logger.info("Engine components initialized successfully.")
    except Exception as e:
        logger.critical(f"Engine initialization failure: {e}", exc_info=True)
        raise
    yield
    logger.info("Deactivating Nexus Synthesis Engine...")

app = FastAPI(
    title="Nexus AI Assistant API",
    description="Local RAG-powered research infrastructure",
    version="3.4.0",
    lifespan=lifespan,
)

# CORS Infrastructure
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Diagnostic pulse check for the neural engine."""
    status_map = {"ok": "healthy", "error": "unhealthy"}
    vs_status = "healthy"
    try:
        if research_service:
            research_service.get_stats()
    except Exception:
        vs_status = "unhealthy"

    return HealthResponse(
        status="healthy" if vs_status == "healthy" else "degraded",
        version="3.4.0",
        timestamp=datetime.now().isoformat(),
        vector_store_status=vs_status,
    )

@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Ingest a local document node into the neural index."""
    start_time = time.time()
    ext = Path(file.filename).suffix.lower().lstrip(".")
    
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail=f"Incompatible format: {ext}"
        )

    try:
        content = await file.read()
        if len(content) > settings.max_file_size_mb * 1024 * 1024:
             raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Payload exceeds {settings.max_file_size_mb}MB threshold"
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        node_count = await research_service.ingest_file(tmp_path, file.filename)
        if os.path.exists(tmp_path): os.unlink(tmp_path)

        return UploadResponse(
            success=True,
            message="Node indexed successfully",
            source=file.filename,
            chunk_count=node_count or 0,
            processing_time_seconds=time.time() - start_time
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failure: {e}")
        raise HTTPException(status_code=500, detail="Internal neural error during ingestion")

@app.post("/api/upload/url", response_model=UploadResponse)
async def upload_from_url(source_url: str):
    """Sync a remote knowledge node via synapse link."""
    start_time = time.time()
    try:
        chunk_count = await research_service.ingest_url(source_url)
        return UploadResponse(
            success=True,
            message="URL indexed",
            source=source_url,
            chunk_count=chunk_count or 0,
            processing_time_seconds=time.time() - start_time
        )
    except Exception as e:
        logger.error(f"URL Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{source:path}")
async def get_document(source: str):
    """Retrieve a raw document artifact from storage."""
    path = research_service.get_document_path(source)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(path, filename=os.path.basename(path), content_disposition_type="inline")

@app.delete("/api/documents/{source:path}")
async def delete_document(source: str):
    """Exise a specific knowledge node and its vector fragments."""
    if not research_service.delete_document(source):
        raise HTTPException(status_code=500, detail="De-indexing operation failed")
    return {"message": f"Successfully de-indexed {source}"}

@app.delete("/api/documents")
async def purge_all_documents():
    """Nuclear purge of the entire knowledge index."""
    if not research_service.purge_all():
        raise HTTPException(status_code=500, detail="Neural purge failed")
    return {"message": "Entire neural index purged successfully"}

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Perform a structured neural synthesis query."""
    try:
        res = research_service.search_and_answer(
            query=request.query,
            top_k=request.top_k,
            search_type=request.search_type,
        )
        
        sources = [RetrievalSource(
            source=s["source"], chunk_index=s["chunk_index"], 
            score=float(s["score"]), content_preview=s["content_preview"]
        ) for s in res.sources]

        research_service.add_to_history(request.query, res.answer)

        return QueryResponse(
            answer=res.answer, query=res.query, sources=sources,
            context=res.context, confidence_score=res.confidence_score,
            processing_time_seconds=res.processing_time_seconds,
            model_used=res.model_used
        )
    except Exception as e:
        logger.error(f"Synthesis error: {e}")
        raise HTTPException(status_code=500, detail="Neural synthesis disrupted")

@app.post("/api/query/stream")
async def query_documents_streaming(request: QueryRequest):
    """Stream a neural synthesis response in real-time."""
    try:
        def history_wrapper():
            full_answer = ""
            for chunk in research_service.search_and_answer_streaming(
                query=request.query,
                top_k=request.top_k,
                search_type=request.search_type,
            ):
                full_answer += chunk
                yield chunk
            research_service.add_to_history(request.query, full_answer)

        return StreamingResponse(history_wrapper(), media_type="text/event-stream")
    except Exception as e:
        logger.error(f"Stream synthesis error: {e}")
        raise HTTPException(status_code=500, detail="Streaming synthesis disrupted")

@app.get("/api/history", response_model=List[Dict[str, Any]])
async def get_history():
    """Retrieve global neural interaction history."""
    return research_service.get_history()

@app.delete("/api/history/{history_id}")
async def delete_history_item(history_id: str):
    """Delete a specific history node."""
    if not research_service.delete_history_item(history_id):
        raise HTTPException(status_code=404, detail="History node not found")
    return {"message": "Node de-indexed from history"}

@app.get("/api/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    """Enumerate all active knowledge nodes."""
    docs = research_service.list_all_documents()
    return [{
        "source": d.source,
        "source_type": d.source_type,
        "title": d.title,
        "chunk_count": d.chunk_count,
        "created_at": d.created_at,
        "metadata": d.metadata
    } for d in docs]

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Diagnostic system metrics."""
    s = research_service.get_stats()
    return StatsResponse(
        total_documents=s.total_documents,
        total_chunks=s.total_chunks,
        storage_size_bytes=s.storage_size_bytes,
        last_updated=s.last_updated,
    )

app.mount("/", StaticFiles(directory="frontend/web", html=True), name="static")
