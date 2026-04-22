"""
AI Research Assistant - Summarizer Module
Document summarization using Mistral AI
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel, Field

from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class SummaryType(str, Enum):
    """Types of summaries available."""
    SHORT = "short"
    BULLET = "bullet"
    DETAILED = "detailed"
    EXECUTIVE = "executive"


class SummaryResponse(BaseModel):
    """Response from summarization."""
    summary: str
    summary_type: SummaryType
    source: str
    key_points: List[str] = Field(default_factory=list)
    word_count: int
    processing_time_seconds: float
    model_used: str
    input_tokens: int = 0
    output_tokens: int = 0


class Summarizer:
    """
    Production-grade document summarizer using Mistral AI.
    Supports multiple summary types: short, bullet points, detailed, and executive.
    """

    SHORT_PROMPT = """You are an expert at summarizing documents concisely.

Create a brief summary of the following document in 2-3 sentences.
Focus on the main topic and key findings.

Document:
{doc_content}

Brief Summary:"""

    BULLET_PROMPT = """You are an expert at extracting key information from documents.

Analyze the following document and extract the main points as a bulleted list.
Each bullet point should capture a key finding, concept, or important detail.

Document:
{doc_content}

Key Points (bullet format):"""

    DETAILED_PROMPT = """You are an expert research analyst.

Create a comprehensive detailed summary of the following document.
Your summary should include:
1. Main topic and purpose
2. Key findings and arguments
3. Important details and evidence
4. Conclusions and implications

Be thorough but concise - cover all important aspects.

Document:
{doc_content}

Detailed Summary:"""

    EXECUTIVE_PROMPT = """You are an executive summary specialist.

Create a professional executive summary of the following document.
This should be suitable for business or academic leadership.

Include:
1. Overview (1-2 sentences)
2. Main objectives or key themes
3. Critical findings
4. Strategic recommendations or conclusions

Document:
{doc_content}

Executive Summary:"""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        api_key: str = "",
        model: str = "mistral-large-latest",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ):
        """
        Initialize the summarizer.

        Args:
            retriever: Optional retriever for document retrieval
            api_key: Mistral AI API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        self.retriever = retriever
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize Mistral chat model
        self.llm = ChatMistralAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Prompts for different summary types
        self.prompts = {
            SummaryType.SHORT: self.SHORT_PROMPT,
            SummaryType.BULLET: self.BULLET_PROMPT,
            SummaryType.DETAILED: self.DETAILED_PROMPT,
            SummaryType.EXECUTIVE: self.EXECUTIVE_PROMPT,
        }

        logger.info(f"Summarizer initialized: model={model}")

    def summarize_document(
        self,
        document_content: str,
        summary_type: SummaryType = SummaryType.SHORT,
        max_input_tokens: int = 10000,
    ) -> SummaryResponse:
        """
        Summarize a document.

        Args:
            document_content: Full document text
            summary_type: Type of summary to generate
            max_input_tokens: Maximum input tokens (truncate if needed)

        Returns:
            SummaryResponse with summary and metadata
        """
        start_time = time.time()

        logger.info(f"Generating {summary_type.value} summary")

        # Truncate content if too long
        truncated_content = self._truncate_content(document_content, max_input_tokens)

        # Get prompt for summary type
        prompt_template = self.prompts.get(
            summary_type,
            self.prompts[SummaryType.SHORT]
        )

        # Generate summary
        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            chain = prompt | self.llm

            response = chain.invoke({"doc_content": truncated_content})

            # Extract content and tokens from response
            summary = ""
            input_tokens = 0
            output_tokens = 0
            
            if hasattr(response, "content"):
                summary = response.content
                usage = response.response_metadata.get("token_usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            elif isinstance(response, dict):
                summary = response.get("content", "")

            # Extract key points if bullet summary
            key_points = []
            if summary_type == SummaryType.BULLET:
                key_points = self._extract_bullet_points(summary)

            elapsed = time.time() - start_time

            return SummaryResponse(
                summary=summary,
                summary_type=summary_type,
                source="document",
                key_points=key_points,
                word_count=len(summary.split()),
                processing_time_seconds=elapsed,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            elapsed = time.time() - start_time

            return SummaryResponse(
                summary=f"Error generating summary: {str(e)}",
                summary_type=summary_type,
                source="document",
                key_points=[],
                word_count=0,
                processing_time_seconds=elapsed,
                model_used=self.model,
            )

    def summarize_from_source(
        self,
        source: str,
        summary_type: SummaryType = SummaryType.SHORT,
        max_docs: int = 10,
    ) -> SummaryResponse:
        """
        Summarize documents from a specific source.

        Args:
            source: Source identifier
            summary_type: Type of summary
            max_docs: Maximum number of documents to retrieve

        Returns:
            SummaryResponse
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")

        logger.info(f"Summarizing from source: {source}")

        # Retrieve documents from source
        retrieval_results = self.retriever.retrieve_by_source(
            query="",  # Empty query to get all
            source=source,
            k=max_docs,
        )

        if not retrieval_results:
            return SummaryResponse(
                summary="No documents found for the specified source.",
                summary_type=summary_type,
                source=source,
                key_points=[],
                word_count=0,
                processing_time_seconds=0.0,
                model_used=self.model,
            )

        # Combine document content
        combined_content = "\n\n".join(
            r.document.page_content for r in retrieval_results
        )

        # Generate summary
        return self.summarize_document(combined_content, summary_type)

    def summarize_all_documents(
        self,
        summary_type: SummaryType = SummaryType.EXECUTIVE,
        max_docs: int = 20,
    ) -> SummaryResponse:
        """
        Summarize all indexed documents.

        Args:
            summary_type: Type of summary
            max_docs: Maximum documents to include

        Returns:
            SummaryResponse
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")

        logger.info("Summarizing all documents")

        # Retrieve documents
        retrieval_results = self.retriever.retrieve(
            query="*",
            config=self.retriever.default_config,
        )

        if not retrieval_results:
            return SummaryResponse(
                summary="No documents available for summarization.",
                summary_type=summary_type,
                source="all",
                key_points=[],
                word_count=0,
                processing_time_seconds=0.0,
                model_used=self.model,
            )

        # Take top documents
        docs = retrieval_results[:max_docs]

        # Combine content
        combined_content = "\n\n".join(
            r.document.page_content for r in docs
        )

        return self.summarize_document(combined_content, summary_type)

    def generate_research_summary(
        self,
        query: str,
        summary_type: SummaryType = SummaryType.DETAILED,
    ) -> SummaryResponse:
        """
        Generate a research-focused summary for a query.

        Args:
            query: Research topic or question
            summary_type: Type of summary

        Returns:
            SummaryResponse
        """
        if not self.retriever:
            raise ValueError("Retriever not initialized")

        logger.info(f"Generating research summary for: {query}")

        # Retrieve relevant documents
        retrieval_results = self.retriever.retrieve(query, k=10)

        if not retrieval_results:
            return SummaryResponse(
                summary="No relevant documents found for the research topic.",
                summary_type=summary_type,
                source=query,
                key_points=[],
                word_count=0,
                processing_time_seconds=0.0,
                model_used=self.model,
            )

        # Get context
        context = self.retriever.get_context_from_results(
            retrieval_results,
            max_chars=8000,
        )

        # Add research-focused prompt
        research_prompt = f"""Based on the following documents related to '{query}', 
provide a comprehensive research summary covering:

1. Overview of the topic
2. Key findings and insights
3. Supporting evidence
4. Conclusions

Documents:
{context}

Research Summary:"""

        # Generate summary
        start_time = time.time()

        try:
            prompt = ChatPromptTemplate.from_template(research_prompt)
            chain = prompt | self.llm

            response = chain.invoke({})

            summary = ""
            input_tokens = 0
            output_tokens = 0

            if hasattr(response, "content"):
                summary = response.content
                usage = response.response_metadata.get("token_usage", {})
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)
            elif isinstance(response, dict):
                summary = response.get("content", "")

            elapsed = time.time() - start_time

            return SummaryResponse(
                summary=summary,
                summary_type=summary_type,
                source=query,
                key_points=self._extract_key_insights(summary),
                word_count=len(summary.split()),
                processing_time_seconds=elapsed,
                model_used=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens
            )

        except Exception as e:
            logger.error(f"Error generating research summary: {str(e)}")
            return SummaryResponse(
                summary=f"Error: {str(e)}",
                summary_type=summary_type,
                source=query,
                key_points=[],
                word_count=0,
                processing_time_seconds=time.time() - start_time,
                model_used=self.model,
            )

    def _truncate_content(
        self,
        content: str,
        max_tokens: int,
    ) -> str:
        """Truncate content to fit within token limit."""
        # Rough estimate: 4 characters per token
        max_chars = max_tokens * 4

        if len(content) <= max_chars:
            return content

        # Truncate and add note
        truncated = content[:max_chars]
        return truncated + "\n\n[Content truncated due to length...]"

    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract bullet points from text."""
        lines = text.split("\n")
        points = []

        for line in lines:
            line = line.strip()
            # Remove bullet characters
            line = line.lstrip("•-*•")
            line = line.strip()

            if line and len(line) > 10:
                points.append(line)

        return points

    def _extract_key_insights(self, text: str) -> List[str]:
        """Extract key insights from summary."""
        # Use bullet extraction for now
        return self._extract_bullet_points(text)

    def get_summary_types(self) -> List[Dict[str, str]]:
        """Get available summary types."""
        return [
            {
                "type": SummaryType.SHORT.value,
                "description": "Brief 2-3 sentence summary",
            },
            {
                "type": SummaryType.BULLET.value,
                "description": "Key points in bullet format",
            },
            {
                "type": SummaryType.DETAILED.value,
                "description": "Comprehensive detailed summary",
            },
            {
                "type": SummaryType.EXECUTIVE.value,
                "description": "Professional executive summary",
            },
        ]
