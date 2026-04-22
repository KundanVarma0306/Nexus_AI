"""
AI Research Assistant - API Models
Pydantic models for request/response validation
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Request Models
class UploadRequest(BaseModel):
    """Request to upload a document."""
    source_type: str = Field(..., description="Type of source: file, url")
    source: Optional[str] = Field(None, description="URL or file path")


class QueryRequest(BaseModel):
    """Request to query the knowledge base."""
    query: str = Field(..., description="Research question or query")
    top_k: int = Field(5, description="Number of documents to retrieve")
    search_type: str = Field("hybrid", description="Type of search: hybrid, similarity, mmr")
    return_sources: bool = Field(True, description="Whether to return source documents")
    return_context: bool = Field(True, description="Whether to return the context")
    model: Optional[str] = Field(None, description="Mistral model to use")


class SummaryRequest(BaseModel):
    """Request to generate a summary."""
    source: Optional[str] = Field(None, description="Source to summarize (all if not provided)")
    summary_type: str = Field("short", description="Type of summary: short, bullet, detailed, executive")


class DeleteRequest(BaseModel):
    """Request to delete documents."""
    source: str = Field(..., description="Source identifier to delete")


class UserCreate(BaseModel):
    """Request to sign up a new user."""
    username: str = Field(..., min_length=3)
    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    password: str = Field(..., min_length=6)


class UserLogin(BaseModel):
    """Request to login a user."""
    email: str = Field(...)
    password: str = Field(...)


class Token(BaseModel):
    """Authentication token response."""
    access_token: str
    token_type: str = "bearer"
    username: str


# Response Models
class DocumentInfo(BaseModel):
    """Information about a stored document."""
    doc_id: str
    source: str
    source_type: str
    title: str
    chunk_count: int
    created_at: str


class SourceInfo(BaseModel):
    """Information about a document source."""
    source: str
    source_type: str
    title: str
    chunk_count: int
    created_at: str


class RetrievalSource(BaseModel):
    """A retrieved source document."""
    source: str
    chunk_index: int
    score: float
    content_preview: str


class QueryResponse(BaseModel):
    """Response from a query."""
    answer: str
    query: str
    sources: List[RetrievalSource]
    context: str
    confidence_score: float
    processing_time_seconds: float
    model_used: str
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0


class SummaryResponse(BaseModel):
    """Response from a summary request."""
    summary: str
    summary_type: str
    source: str
    key_points: List[str]
    word_count: int
    processing_time_seconds: float
    model_used: str
    input_tokens: Optional[int] = 0
    output_tokens: Optional[int] = 0


class UploadResponse(BaseModel):
    """Response from document upload."""
    success: bool
    message: str
    source: str
    chunk_count: int
    processing_time_seconds: float


class DeleteResponse(BaseModel):
    """Response from document deletion."""
    success: bool
    message: str
    source: str


class StatsResponse(BaseModel):
    """System statistics."""
    total_documents: int
    total_chunks: int
    storage_size_bytes: int
    last_updated: str
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    model_usage: Dict[str, Any] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: str
    vector_store_status: str


class UserResponse(BaseModel):
    """Information about a user."""
    user_id: str
    username: str
    email: str
    created_at: str


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
