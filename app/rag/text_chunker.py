"""
AI Research Assistant - Text Chunker Module
Splits documents into smaller chunks for embedding and retrieval
"""

import logging
import re
from typing import List, Optional

from langchain_text_splitters import (
    MarkdownTextSplitter,
    PythonCodeTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)
from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChunkMetadata(BaseModel):
    """Metadata for text chunks."""
    chunk_index: int
    total_chunks: int
    source: str
    source_type: str
    char_count: int
    token_count: int
    title: Optional[str] = None


class TextChunker:
    """
    Production-grade text chunker with multiple splitting strategies.
    Supports recursive splitting, markdown-aware, and code-aware splitting.
    """

    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    MIN_CHUNK_SIZE = 100

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        split_strategy: str = "recursive",
        separators: Optional[List[str]] = None,
    ):
        """
        Initialize the text chunker.

        Args:
            chunk_size: Maximum size of each chunk in characters
            chunk_overlap: Overlap between chunks in characters
            split_strategy: Strategy for splitting ('recursive', 'markdown', 'code')
            separators: Custom separators for splitting
        """
        if chunk_size < self.MIN_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size must be at least {self.MIN_CHUNK_SIZE}"
            )

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be less than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.split_strategy = split_strategy
        self.separators = separators or self._get_default_separators()

        # Initialize the appropriate splitter
        self.splitter = self._create_splitter()

        logger.info(
            f"TextChunker initialized: strategy={split_strategy}, "
            f"size={chunk_size}, overlap={chunk_overlap}"
        )

    def _get_default_separators(self) -> List[str]:
        """Get default separators for recursive splitting."""
        return [
            "\n\n",  # Paragraphs
            "\n",   # Lines
            ". ",   # Sentences
            ", ",   # Clauses
            " ",    # Words
            "",     # Characters
        ]

    def _create_splitter(self) -> TextSplitter:
        """Create the appropriate text splitter based on strategy."""
        if self.split_strategy == "markdown":
            return MarkdownTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
            )
        elif self.split_strategy == "code":
            return PythonCodeTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        else:  # recursive
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                length_function=len,
                is_separator_regex=False,
                keep_separator=True,
            )

    def chunk_documents(
        self,
        documents: List[Document],
        add_metadata: bool = True,
    ) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            documents: List of LangChain documents to chunk
            add_metadata: Whether to add chunk metadata

        Returns:
            List of chunked documents
        """
        if not documents:
            return []

        logger.info(f"Chunking {len(documents)} documents")

        # Split documents
        chunks = self.splitter.split_documents(documents)

        # Add chunk metadata
        if add_metadata:
            chunks = self._add_chunk_metadata(chunks)

        logger.info(f"Created {len(chunks)} chunks")

        return chunks

    def chunk_text(
        self,
        text: str,
        source: str = "text",
        source_type: str = "text",
        add_metadata: bool = True,
    ) -> List[Document]:
        """
        Split raw text into chunks.

        Args:
            text: Raw text to chunk
            source: Source identifier
            source_type: Type of source
            add_metadata: Whether to add chunk metadata

        Returns:
            List of chunked documents
        """
        if not text or not text.strip():
            return []

        # Create a document
        doc = Document(
            page_content=text,
            metadata={
                "source": source,
                "source_type": source_type,
            }
        )

        # Chunk the document
        return self.chunk_documents([doc], add_metadata=add_metadata)

    def _add_chunk_metadata(
        self,
        chunks: List[Document],
    ) -> List[Document]:
        """Add metadata to each chunk."""
        total_chunks = len(chunks)

        for idx, chunk in enumerate(chunks):
            # Add chunk-specific metadata
            chunk.metadata.update({
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "char_count": len(chunk.page_content),
                "token_count": self._estimate_tokens(chunk.page_content),
            })

        return chunks

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count using a simple heuristic.
        ~4 characters per token on average
        """
        return len(text) // 4

    def chunk_by_headings(
        self,
        documents: List[Document],
        include_heading_in_chunk: bool = True,
    ) -> List[Document]:
        """
        Chunk documents by preserving heading structure.
        Best for structured documents like markdown.

        Args:
            documents: List of documents to chunk
            include_heading_in_chunk: Whether to include heading in each chunk

        Returns:
            List of chunked documents
        """
        chunks = []

        for doc in documents:
            content = doc.page_content
            metadata = doc.metadata.copy()

            # Split by headings (markdown style)
            heading_pattern = r"^(#{1,6})\s+(.+)$"
            lines = content.split("\n")

            current_section = []
            current_heading = ""

            for line in lines:
                match = re.match(heading_pattern, line)
                if match:
                    # Save previous section
                    if current_section:
                        chunk_text = "\n".join(current_section)
                        chunk_meta = metadata.copy()
                        if current_heading and include_heading_in_chunk:
                            chunk_text = f"{current_heading}\n\n{chunk_text}"
                            chunk_meta["heading"] = current_heading.strip("# ").strip()

                        chunks.append(Document(
                            page_content=chunk_text,
                            metadata=chunk_meta,
                        ))

                    # Start new section
                    current_heading = line
                    current_section = [line]
                else:
                    current_section.append(line)

            # Don't forget last section
            if current_section:
                chunk_text = "\n".join(current_section)
                chunk_meta = metadata.copy()
                if current_heading and include_heading_in_chunk:
                    chunk_text = f"{current_heading}\n\n{chunk_text}"
                    chunk_meta["heading"] = current_heading.strip("# ").strip()

                chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_meta,
                ))

        # Add chunk metadata
        return self._add_chunk_metadata(chunks)

    def merge_small_chunks(
        self,
        chunks: List[Document],
        min_size: int = 100,
    ) -> List[Document]:
        """
        Merge small chunks with adjacent chunks.

        Args:
            chunks: List of chunks
            min_size: Minimum chunk size threshold

        Returns:
            List of merged chunks
        """
        if not chunks:
            return []

        merged = [chunks[0]]

        for chunk in chunks[1:]:
            if len(merged[-1].page_content) < min_size:
                # Merge with previous chunk
                merged[-1] = Document(
                    page_content=merged[-1].page_content + "\n\n" + chunk.page_content,
                    metadata={
                        **merged[-1].metadata,
                        **chunk.metadata,
                        "merged": True,
                    },
                )
            else:
                merged.append(chunk)

        # Recalculate metadata
        return self._add_chunk_metadata(merged)

    def get_chunk_statistics(self, chunks: List[Document]) -> dict:
        """
        Get statistics about the chunks.

        Args:
            chunks: List of chunks

        Returns:
            Dictionary of statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "avg_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0,
                "total_characters": 0,
            }

        sizes = [len(c.page_content) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) // len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes),
            "total_tokens": sum(self._estimate_tokens(c.page_content) for c in chunks),
        }
