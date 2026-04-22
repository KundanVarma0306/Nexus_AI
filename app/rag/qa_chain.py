"""
AI Research Assistant - QA Chain Module
Question answering using LangChain and Mistral AI with RAG
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


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
    input_tokens: int = 0
    output_tokens: int = 0


class QAChain:
    """
    Precision neural synthesis engine using Mistral AI.
    Implements a lean RAG pipeline with direct context injection.
    """

    SYSTEM_FABRIC = """You are the Nexus Research Synthesis Engine, a high-fidelity cognitive system.
Your goal is to provide deep, analytical research synthesis based on provided document context.

### COGNITIVE OPERATIONS:
1. **Integrated Synthesis**: Analyze the provided context collectively. Identify shared themes and critical insights.
2. **Context-Driven Accuracy**: Base your answer strictly on the provided documents. Use the document names provided in the context to reference sources.
3. **High-Fidelity Response**: Start answering immediately. Zero conversational fluff. Use elegant, academic language.
4. **Transparency**: If the documents do not contain enough information to answer the query, simply state that the current knowledge base does not cover the topic.

### RESEARCH CONTEXT:
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
        self.api_key = api_key
        self.llm = ChatMistralAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.retrieval_config = RetrievalConfig(k=5, search_type="hybrid")
        logger.info(f"Neural engine initialized: {model}")

    def set_model(self, model_name: str):
        """Update the active neural model."""
        if self.model != model_name:
            self.model = model_name
            self.llm = ChatMistralAI(
                api_key=self.api_key,
                model=model_name,
                temperature=self.llm.temperature,
                max_tokens=self.llm.max_tokens,
            )
            logger.info(f"Model swapped to: {model_name}")

    def generate_title(self, context: str) -> str:
        """Generate a short, contextual title for a research node."""
        try:
            prompt = f"Based on this AI research synthesis, generate a very short, professional 3-5 word title that describes the core topic. Do not use quotes, punctuation, or introductory text. \n\nSynthesis: {context[:1000]}"
            res = self.llm.invoke(prompt)
            return res.content.strip().replace('"', '').replace('Title:', '').strip()
        except Exception as e:
            logger.error(f"Title generation failed: {e}")
            return "Untitled Research"

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
            
            # Extract token usage if available
            usage = res.response_metadata.get("token_usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # Robust Fallback: Estimate tokens if API returns 0 (common in some regions/versions)
            if input_tokens == 0:
                input_tokens = len(prompt) // 4
            if output_tokens == 0:
                output_tokens = len(answer) // 4

            # Distance-based confidence heuristic (0.0 to 1.0)
            avg_score = sum(r.score for r in results) / len(results)
            confidence = max(0.01, min(0.99, 1.2 - avg_score))
        except Exception as e:
            logger.error(f"Synthesis fault: {e}")
            answer = f"NEURAL_FAULT: Synthesis disrupted. Raw Context: {context[:500]}..."
            confidence = 0.0
            input_tokens = 0
            output_tokens = 0

        return QAResponse(
            answer=answer, sources=sources if return_sources else [],
            context=context, query=query, confidence_score=confidence,
            processing_time_seconds=time.time() - start, 
            model_used=self.model, retrieval_config=self.retrieval_config.model_dump(),
            input_tokens=input_tokens, output_tokens=output_tokens
        )

    def answer_streaming(self, query: str):
        """Stream a neural synthesis response."""
        results = self.retriever.retrieve(query, self.retrieval_config)
        if not results:
            yield "I couldn't find any information in the current research library that addresses your query."
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
            answer="I am sorry, but your query does not seem to be covered by my current index of documents.",
            sources=[], context="", query=query, confidence_score=0.0,
            processing_time_seconds=elapsed, model_used=self.model,
            retrieval_config=self.retrieval_config.model_dump()
        )
