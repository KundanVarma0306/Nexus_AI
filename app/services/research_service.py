"""
AI Research Assistant - Research Service
Handles high-level business logic for research operations
"""

import logging
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.rag import (
    DocumentLoader,
    QAChain,
    Retriever,
    Summarizer,
    TextChunker,
    VectorStore,
)
from config.settings import settings

logger = logging.getLogger(__name__)

class ResearchService:
    """
    Core orchestration layer for the AI Research Assistant.
    Integrates RAG components with persistence and business logic.
    """

    def __init__(self):
        # Neural components initialization
        self.vector_store = VectorStore(
            persist_directory=settings.chroma_db_path,
            api_key=settings.mistral_api_key,
            embedding_model=settings.embedding_model,
        )

        self.retriever = Retriever(vector_store=self.vector_store)
        self.qa_chain = QAChain(
            retriever=self.retriever,
            api_key=settings.mistral_api_key,
            model=settings.llm_model,
        )
        self.summarizer = Summarizer(
            retriever=self.retriever,
            api_key=settings.mistral_api_key,
            model=settings.llm_model,
        )
        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        
        # History & Persistence
        self.history_file = os.path.join(settings.cache_path, "query_history.json")
        self.history: List[Dict[str, Any]] = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Safely load neural interaction history."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"History corruption detected: {e}")
        return []

    def _persist_history(self) -> None:
        """Atomic write of neural history."""
        try:
            temp_file = f"{self.history_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(self.history, f, indent=2)
            os.replace(temp_file, self.history_file)
        except Exception as e:
            logger.error(f"Atomic persistence failed: {e}")

    async def ingest_file(self, source_path: str, filename: str) -> int:
        """Ingest, fragment, and index a local filesystem node."""
        docs = self.loader.load_file(source_path)
        if not docs: return 0
        
        # Collision-resistant path resolution
        dest_path = os.path.join(settings.uploads_path, filename)
        if os.path.exists(dest_path):
             ts = int(datetime.now().timestamp())
             filename = f"{Path(filename).stem}_{ts}{Path(filename).suffix}"
             dest_path = os.path.join(settings.uploads_path, filename)

        shutil.copy2(source_path, dest_path)

        for doc in docs:
            doc.metadata.update({"source": filename, "file_path": str(dest_path)})
            
        nodes = self.chunker.chunk_documents(docs)
        self.vector_store.add_documents(nodes)
        return len(nodes)

    async def ingest_url(self, url: str) -> int:
        """Index a remote knowledge node via URL synapse."""
        docs = self.loader.load_url(url)
        if not docs: return 0
        
        for doc in docs: doc.metadata["source_type"] = "url"
        nodes = self.chunker.chunk_documents(docs)
        self.vector_store.add_documents(nodes)
        return len(nodes)

    def add_to_history(self, query: str, answer: str):
        """Register a new research synthesis node in history."""
        self.history.insert(0, {
            "id": f"node_{hex(int(datetime.now().timestamp()))[2:]}",
            "query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })
        self.history = self.history[:100]
        self._persist_history()

    def delete_history_item(self, node_id: str) -> bool:
        """Excise a specific history node."""
        original_count = len(self.history)
        self.history = [n for n in self.history if n["id"] != node_id]
        if len(self.history) < original_count:
            self._persist_history()
            return True
        return False

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history

    def list_all_documents(self) -> List[Any]:
        """Enumerate all active knowledge nodes in the vector fabric."""
        return [self.vector_store.get_document_info(s) for s in self.vector_store.get_all_sources() if s]

    def get_document_path(self, source: str) -> Optional[str]:
        """Resolve a knowledge node ID to a physical filesystem coordinate."""
        info = self.vector_store.get_document_info(source)
        if info and "file_path" in info.metadata:
            path = Path(info.metadata["file_path"])
            if path.exists(): return str(path)
            
            # Fallback for environment migrations (same-dir fallback)
            fallback = settings.uploads_path / path.name
            if fallback.exists(): return str(fallback)

        direct = settings.uploads_path / source
        return str(direct) if direct.exists() else None

    def delete_document(self, source: str) -> bool:
        """De-index fragments and exise source artifact."""
        path = self.get_document_path(source)
        if self.vector_store.delete_document(source):
            if path and os.path.exists(path):
                try: os.remove(path)
                except OSError as e: logger.error(f"Cleanup failure for {path}: {e}")
            return True
        return False

    def purge_all(self) -> bool:
        """Nuclear resets of the entire neural memory."""
        if self.vector_store.delete_all():
            if settings.uploads_path.exists():
                for item in settings.uploads_path.iterdir():
                    try:
                        if item.is_file(): item.unlink()
                        elif item.is_dir(): shutil.rmtree(item)
                    except Exception as e: logger.error(f"Purge error at {item}: {e}")
            return True
        return False

    def search_and_answer(self, query: str, top_k: int = 5, search_type: str = "hybrid"):
        self.qa_chain.retrieval_config.k = top_k
        self.qa_chain.retrieval_config.search_type = search_type
        return self.qa_chain.answer(query)

    def search_and_answer_streaming(self, query: str, top_k: int = 5, search_type: str = "hybrid"):
        self.qa_chain.retrieval_config.k = top_k
        self.qa_chain.retrieval_config.search_type = search_type
        return self.qa_chain.answer_streaming(query)

    def get_stats(self):
        return self.vector_store.get_stats()
