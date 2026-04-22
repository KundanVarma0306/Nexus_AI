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
        self.usage_file = os.path.join(settings.cache_path, "usage_stats.json")
        self.history: List[Dict[str, Any]] = self._load_history()
        self.usage: Dict[str, Any] = self._load_usage()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Safely load neural interaction history."""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"History corruption detected: {e}")
        return []

    def _load_usage(self) -> Dict[str, Any]:
        """Load cumulative token usage statistics."""
        default_usage = {
            "total_input_tokens": 0, 
            "total_output_tokens": 0, 
            "total_cost_usd": 0.0,
            "model_usage": {}
        }
        if os.path.exists(self.usage_file):
            try:
                with open(self.usage_file, "r") as f:
                    usage = json.load(f)
                    # Migrate old usage format if necessary
                    if "model_usage" not in usage:
                        usage["model_usage"] = {}
                    
                    # Key Migration: mistral-small-latest -> codestral-latest
                    if "mistral-small-latest" in usage["model_usage"]:
                        old_data = usage["model_usage"].pop("mistral-small-latest")
                        if "codestral-latest" not in usage["model_usage"]:
                            usage["model_usage"]["codestral-latest"] = old_data
                        else:
                            usage["model_usage"]["codestral-latest"]["input"] += old_data.get("input", 0)
                            usage["model_usage"]["codestral-latest"]["output"] += old_data.get("output", 0)
                            usage["model_usage"]["codestral-latest"]["cost"] += old_data.get("cost", 0.0)
                    return usage
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Usage file corruption: {e}")
        return default_usage

    def _persist_usage(self) -> None:
        """Atomic write of usage statistics."""
        try:
            temp_file = f"{self.usage_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(self.usage, f, indent=2)
            os.replace(temp_file, self.usage_file)
        except Exception as e:
            logger.error(f"Usage persistence failed: {e}")

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
        # Generate a smart contextual title based on the synthesis
        smart_title = self.qa_chain.generate_title(answer)
        
        self.history.insert(0, {
            "id": f"node_{hex(int(datetime.now().timestamp()))[2:]}",
            "query": smart_title,
            "original_query": query,
            "answer": answer,
            "timestamp": datetime.now().isoformat(),
        })
        self.history = self.history[:100]
        self._persist_history()

    def migrate_history_titles(self):
        """Retroactively rename all history items based on their contextual synthesis."""
        for node in self.history:
            new_title = self.qa_chain.generate_title(node["answer"])
            if new_title:
                node["query"] = new_title
        self._persist_history()

    def delete_history_item(self, node_id: str) -> bool:
        """Excise a specific history node."""
        original_count = len(self.history)
        self.history = [n for n in self.history if n["id"] != node_id]
        if len(self.history) < original_count:
            self._persist_history()
            return True
        return False

    def rename_history_item(self, node_id: str, new_query: str) -> bool:
        """Update the label of a history node."""
        for node in self.history:
            if node["id"] == node_id:
                node["query"] = new_query
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
            
            # Reset usage stats as well
            self.usage = {
                "total_input_tokens": 0, 
                "total_output_tokens": 0, 
                "total_cost_usd": 0.0,
                "model_usage": {}
            }
            self._persist_usage()
            return True
        return False

    def search_and_answer(self, query: str, top_k: int = 5, search_type: str = "hybrid", model: Optional[str] = None):
        self.qa_chain.retrieval_config.k = top_k
        self.qa_chain.retrieval_config.search_type = search_type
        if model: self.qa_chain.set_model(model)
        res = self.qa_chain.answer(query)
        self.record_usage(res.model_used, res.input_tokens, res.output_tokens)
        return res

    def search_and_answer_streaming(self, query: str, top_k: int = 5, search_type: str = "hybrid", model: Optional[str] = None):
        self.qa_chain.retrieval_config.k = top_k
        self.qa_chain.retrieval_config.search_type = search_type
        if model: self.qa_chain.set_model(model)
        
        full_answer = ""
        try:
            # We need to capture the full answer to estimate tokens
            for chunk in self.qa_chain.answer_streaming(query):
                full_answer += chunk
                yield chunk
        finally:
            # Estimate tokens for streaming (Rule of thumb: 4 chars per token)
            # This ensures usage is recorded even if the stream is partially consumed
            if full_answer or query:
                # Add a base context overhead (e.g. 500 tokens for retrieved documents)
                # In a real app, we'd count tokens of the actual retrieved context
                est_input_tokens = max(1, len(query) // 4 + 800) 
                est_output_tokens = max(1, len(full_answer) // 4)
                self.record_usage(model or settings.llm_model, est_input_tokens, est_output_tokens)

    def record_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Track and update token usage and costs."""
        # Updated Pricing based on User Request
        # Mistral Large 3: $0.5 / $1.5
        # Devstral 2 / Small: $0.4 / $2.0
        
        # Determine pricing category
        if "large" in model.lower():
            cost_config = {"input": 0.5, "output": 1.5}
            category = "mistral-large-latest"
        elif any(k in model.lower() for k in ["small", "devstral", "codestral"]):
            cost_config = {"input": 0.4, "output": 2.0}
            category = "codestral-latest"
        elif "embed" in model.lower():
            cost_config = {"input": 0.1, "output": 0.0}
            category = "mistral-embed"
        else:
            cost_config = {"input": 0.5, "output": 1.5}
            category = "mistral-large-latest"
        
        input_cost = (input_tokens / 1_000_000) * cost_config["input"]
        output_cost = (output_tokens / 1_000_000) * cost_config["output"]
        total_call_cost = input_cost + output_cost

        # Global usage
        self.usage["total_input_tokens"] += input_tokens
        self.usage["total_output_tokens"] += output_tokens
        self.usage["total_cost_usd"] += total_call_cost
        
        # Per-model usage (Using category as the key for consistent UI display)
        if "model_usage" not in self.usage:
            self.usage["model_usage"] = {}
        
        if category not in self.usage["model_usage"]:
            self.usage["model_usage"][category] = {"input": 0, "output": 0, "cost": 0.0}
            
        self.usage["model_usage"][category]["input"] += input_tokens
        self.usage["model_usage"][category]["output"] += output_tokens
        self.usage["model_usage"][category]["cost"] += total_call_cost
        
        self._persist_usage()

    def get_stats(self):
        stats = self.vector_store.get_stats()
        return {
            "total_documents": stats.total_documents,
            "total_chunks": stats.total_chunks,
            "storage_size_bytes": stats.storage_size_bytes,
            "last_updated": stats.last_updated,
            "total_input_tokens": self.usage["total_input_tokens"],
            "total_output_tokens": self.usage["total_output_tokens"],
            "total_cost_usd": round(self.usage["total_cost_usd"], 6),
            "model_usage": self.usage.get("model_usage", {})
        }
