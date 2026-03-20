"""
AI Research Assistant - Vector Store Module
ChromaDB-based vector storage and retrieval
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentInfo(BaseModel):
    """Information about a stored document."""
    source: str
    source_type: str
    title: str
    chunk_count: int
    created_at: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class VectorStoreStats(BaseModel):
    """Statistics for the vector store."""
    total_documents: int
    total_chunks: int
    collections: List[str]
    storage_size_bytes: int
    embedding_model: str
    last_updated: str


class VectorStore:
    """
    Production-grade vector storage infrastructure utilizing ChromaDB.
    Orchestrates semantic indexing, metadata persistence, and lifecycle metrics.
    """

    def __init__(
        self,
        persist_directory: str = "./database/chroma_db",
        collection_name: str = "research_documents",
        api_key: Optional[str] = None,
        embedding_model: str = "mistral-embed",
    ):
        self.persist_dir = Path(persist_directory)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        
        # Neural synapse configuration
        self.embeddings = MistralAIEmbeddings(
            api_key=api_key or os.getenv("MISTRAL_API_KEY"),
            model=embedding_model,
        )

        # Persistent Vector Fabric
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_dir),
            collection_metadata={"hnsw:space": "cosine"}
        )

        # Metadata Registry
        self._registry: Dict[str, DocumentInfo] = {}
        self._hydrate_registry()

    @property
    def collection(self): return self.vectorstore

    def _hydrate_registry(self) -> None:
        """Hydrate document metadata from persistent storage."""
        registry_path = self.persist_dir / "doc_metadata.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r") as f:
                    data = json.load(f)
                    self._registry = {k: DocumentInfo(**v) for k, v in data.items()}
            except Exception as e: logger.error(f"Registry hydration failure: {e}")

    def _synch_registry(self) -> None:
        """Atomic synchronization of the metadata registry to disk."""
        registry_path = self.persist_dir / "doc_metadata.json"
        try:
            with open(registry_path, "w") as f:
                json.dump({k: v.model_dump() for k, v in self._registry.items()}, f, indent=2)
        except Exception as e: logger.error(f"Registry synchronization failure: {e}")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Index research nodes and register metadata."""
        if not documents: return []
        
        ids = self.vectorstore.add_documents(documents)
        
        # Update metadata state
        for doc in documents:
            source = doc.metadata.get("source", "unknown")
            if source not in self._registry:
                self._registry[source] = DocumentInfo(
                    source=source,
                    source_type=doc.metadata.get("source_type", "file"),
                    title=doc.metadata.get("title", Path(source).name),
                    chunk_count=0,
                    created_at=datetime.now().isoformat(),
                    metadata=doc.metadata
                )
            self._registry[source].chunk_count += 1
            
        self._synch_registry()
        return ids

    def delete_document(self, source: str) -> bool:
        """Excise a research node and its associated neural fragments."""
        try:
            results = self.vectorstore.get(where={"source": source})
            if results["ids"]: self.vectorstore.delete(ids=results["ids"])
            
            if source in self._registry:
                del self._registry[source]
                self._synch_registry()
            return True
        except Exception as e:
            logger.error(f"Excision failure for {source}: {e}")
            return False

    def delete_all(self) -> bool:
        """Full reset of the neural memory fabric."""
        try:
            self.vectorstore.delete_collection()
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir)
            )
            self._registry.clear()
            self._synch_registry()
            return True
        except Exception as e:
            logger.error(f"Fabric reset failure: {e}")
            return False

    def get_stats(self) -> VectorStoreStats:
        """Synthesize system-level operational metrics."""
        count = self.vectorstore._collection.count()
        size = sum(f.stat().st_size for f in self.persist_dir.rglob('*') if f.is_file())
        
        return VectorStoreStats(
            total_documents=len(self._registry),
            total_chunks=count,
            collections=[self.collection_name],
            storage_size_bytes=size,
            embedding_model=self.embeddings.model,
            last_updated=datetime.now().isoformat()
        )

    def get_all_sources(self) -> List[str]: return list(self._registry.keys())
    def get_document_info(self, source: str) -> Optional[DocumentInfo]: return self._registry.get(source)
    def exists(self, source: str) -> bool: return source in self._registry
