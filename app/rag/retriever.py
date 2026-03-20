"""
AI Research Assistant - Retriever Module
Semantic search and retrieval with multiple strategies
"""

import logging
from typing import Any, Dict, List, Optional, Union

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

from app.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalResult(BaseModel):
    """Result from retrieval operation."""
    document: Document
    score: float
    rank: int
    source: str
    chunk_index: int = 0


class RetrievalConfig(BaseModel):
    """Configuration for retrieval."""
    search_type: str = Field(default="mmr", description="Type of search: similarity, mmr, similarity_threshold, or hybrid")
    k: int = Field(default=5, description="Number of documents to retrieve")
    score_threshold: float = Field(default=0.0, description="Minimum score threshold")
    fetch_k: int = Field(default=20, description="Number of documents to fetch for MMR/Hybrid")
    lambda_mult: float = Field(default=0.5, description="Lambda for MMR (0 = diversity, 1 = relevance) or Hybrid (0 = BM25, 1 = Vector)")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filter")


class Retriever:
    """
    High-fidelity retrieval engine for neural research synthesis.
    Supports similarity, MMR, and RRF-based hybrid search strategies.
    """

    def __init__(self, vector_store: VectorStore, config: Optional[RetrievalConfig] = None):
        self.vector_store = vector_store
        self.config = config or RetrievalConfig(k=5, search_type="mmr")
        logger.info(f"Retriever online: search_mode={self.config.search_type}")

    def retrieve(self, query: str, config: Optional[RetrievalConfig] = None) -> List[RetrievalResult]:
        """Execute a retrieval operation based on the configured neural strategy."""
        cfg = config or self.config
        mode = cfg.search_type

        # Dispatch retrieval strategy
        if mode == "similarity":
            results = self.vector_store.collection.similarity_search_with_score(query, k=cfg.k, filter=cfg.filter)
        elif mode == "mmr":
            # Using Chroma's native MMR for performance
            results = self.vector_store.collection.max_marginal_relevance_search(
                query, k=cfg.k, fetch_k=cfg.fetch_k, lambda_mult=cfg.lambda_mult, filter=cfg.filter
            )
            # Normalize scores for MMR (Chroma MMR returns docs only, we assign 1.0 for rank-based scoring)
            results = [(doc, 1.0 - (i / cfg.k * 0.1)) for i, doc in enumerate(results)]
        elif mode == "hybrid":
            results = self._hybrid_fusion(query, cfg)
        else:
            results = self.vector_store.collection.similarity_search_with_score(query, k=cfg.k, filter=cfg.filter)

        return [
            RetrievalResult(
                document=doc, score=float(score), rank=i+1,
                source=doc.metadata.get("source", "N/A"),
                chunk_index=doc.metadata.get("chunk_index", 0)
            ) for i, (doc, score) in enumerate(results)
        ]

    def _hybrid_fusion(self, query: str, cfg: RetrievalConfig) -> List[tuple[Document, float]]:
        """Reciprocal Rank Fusion (RRF) for semantic + structural hybrid search."""
        # Note: True hybrid would need a keyword index. For now, we use dual-scale vector search.
        v_results = self.vector_store.collection.similarity_search_with_score(query, k=cfg.fetch_k)
        m_results = self.vector_store.collection.max_marginal_relevance_search(query, k=cfg.fetch_k)
        
        ranks = {}
        c = 60 # RRF Constant
        
        for i, (doc, _) in enumerate(v_results):
            ranks[doc.page_content] = {"score": 1.0 / (c + i + 1), "doc": doc}
        
        for i, doc in enumerate(m_results):
            if doc.page_content in ranks:
                ranks[doc.page_content]["score"] += 1.0 / (c + i + 1)
            else:
                ranks[doc.page_content] = {"score": 1.0 / (c + i + 1), "doc": doc}
        
        sorted_nodes = sorted(ranks.values(), key=lambda x: x["score"], reverse=True)
        return [(node["doc"], node["score"]) for node in sorted_nodes[:cfg.k]]

    def get_context_from_results(self, results: List[RetrievalResult], max_chars: int = 12000) -> str:
        """
        Assemble a structured, cross-document knowledge fabric for high-fidelity synthesis.
        Groups fragments by source and provides a hierarchical index for the LLM.
        """
        if not results: return ""
        
        # Group fragments by source for coherent cross-document reasoning
        source_groups: Dict[str, List[str]] = {}
        for r in results:
            if r.source not in source_groups: source_groups[r.source] = []
            source_groups[r.source].append(f"[Fragment {r.chunk_index}]: {r.document.page_content}")

        # Construct the knowledge landscape
        fabric = ["### DOCUMENT LANDSCAPE INDEX"]
        for i, source in enumerate(source_groups.keys()):
            fabric.append(f"{i+1}. SOURCE_NODE_{i}: {source}")
        
        fabric.append("\n### CORE KNOWLEDGE SEGMENTS")
        current_len = sum(len(f) for f in fabric)
        
        for i, (source, fragments) in enumerate(source_groups.items()):
            node_header = f"\n>> DOCUMENT_ID: SOURCE_NODE_{i} ({source})"
            if current_len + len(node_header) > max_chars: break
            fabric.append(node_header)
            current_len += len(node_header)
            
            for frag in fragments:
                if current_len + len(frag) > max_chars: break
                fabric.append(frag)
                current_len += len(frag)
            
        return "\n".join(fabric)
