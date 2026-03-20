"""
AI Research Assistant - QA Chain Module
Question answering using LangChain and Mistral AI with RAG
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from app.rag.retriever import RetrievalConfig, Retriever

logger = logging.getLogger(__name__)


class QAResponse(BaseModel):
    """Response from QA chain."""
    answer: str
    sources: List[Dict[str, Any]]
    context: str
    query: str
    confidence_score: float = 0.0
    processing_time_seconds: float = 0.0
    model_used: str
    retrieval_config: Dict[str, Any]


class QAChain:
    """
    Precision neural synthesis engine using Mistral AI.
    Implements a lean RAG pipeline with direct context injection.
    """

    SYSTEM_FABRIC = """You are the Nexus Research Synthesis Engine, a high-fidelity cognitive system designed for industrial-scale knowledge orchestration.
Your primary directive is focused, high-precision cross-document synthesis across multiple Knowledge Nodes.

### COGNITIVE OPERATIONS:
1. **Multi-Node Synthesis**: Aggregately analyze context across all provided documents. Identify convergence and divergence.
2. **Knowledge Mapping**: Explicitly cross-reference information between SOURCE_NODE_X and SOURCE_NODE_Y to build a unified perspective.
3. **High-Fidelity Response**: Start answering immediately. Zero conversational fluff. Use neutral, technical language.
4. **Resolution Strategy**: If documents provide conflicting data, highlight the contradiction. If data is missing from the index, yield "DATA_NULL: [Scope]".

### KNOWLEDGE CONTEXT FABRIC:
{context}

### RESEARCH QUERY:
{input}

### UNIFIED RESEARCH SYNTHESIS:"""

    def __init__(
        self,
        retriever: Retriever,
        api_key: str,
        model: str = "mistral-large-latest",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ):
        self.retriever = retriever
        self.model = model
        self.llm = ChatMistralAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.retrieval_config = RetrievalConfig(k=5, search_type="hybrid")
        logger.info(f"Neural engine initialized: {model}")

    def answer(self, query: str, return_sources: bool = True) -> QAResponse:
        """Perform a discrete neural synthesis query."""
        start = time.time()
        results = self.retriever.retrieve(query, self.retrieval_config)
        
        if not results:
            return self._build_null_response(query, time.time() - start)

        context = self.retriever.get_context_from_results(results)
        sources = [
            {
                "source": r.source,
                "chunk_index": r.chunk_index,
                "score": float(r.score),
                "content_preview": r.document.page_content[:200]
            } for r in results
        ]

        try:
            prompt = self.SYSTEM_FABRIC.format(context=context, input=query)
            res = self.llm.invoke([HumanMessage(content=prompt)])
            answer = res.content
            
            # Distance-based confidence heuristic (0.0 to 1.0)
            avg_score = sum(r.score for r in results) / len(results)
            confidence = max(0.01, min(0.99, 1.2 - avg_score))
        except Exception as e:
            logger.error(f"Synthesis fault: {e}")
            answer = f"NEURAL_FAULT: Synthesis disrupted. Raw Context: {context[:500]}..."
            confidence = 0.0

        return QAResponse(
            answer=answer, sources=sources if return_sources else [],
            context=context, query=query, confidence_score=confidence,
            processing_time_seconds=time.time() - start, 
            model_used=self.model, retrieval_config=self.retrieval_config.model_dump()
        )

    def answer_streaming(self, query: str):
        """Stream a neural synthesis response."""
        results = self.retriever.retrieve(query, self.retrieval_config)
        if not results:
            yield "DATA_NULL: Index search returned zero relevant nodes."
            return

        context = self.retriever.get_context_from_results(results)
        try:
            prompt = self.SYSTEM_FABRIC.format(context=context, input=query)
            for chunk in self.llm.stream([HumanMessage(content=prompt)]):
                if chunk.content: yield chunk.content
        except Exception as e:
            logger.error(f"Stream fault: {e}")
            yield f"\nNEURAL_FAULT: {str(e)}"

    def _build_null_response(self, query: str, elapsed: float) -> QAResponse:
        return QAResponse(
            answer="DATA_NULL: Target query is outside current knowledge index boundaries.",
            sources=[], context="", query=query, confidence_score=0.0,
            processing_time_seconds=elapsed, model_used=self.model,
            retrieval_config=self.retrieval_config.model_dump()
        )
