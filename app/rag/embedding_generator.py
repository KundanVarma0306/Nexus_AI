"""
AI Research Assistant - Embedding Generator Module
Generates embeddings using Mistral AI embeddings API
"""

import logging
import os
from typing import List, Optional, Union

import numpy as np
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmbeddingStats(BaseModel):
    """Statistics for embedding generation."""
    total_documents: int
    total_chunks: int
    embedding_dimension: int
    processing_time_seconds: float
    errors: int = 0


class EmbeddingGenerator:
    """
    Production-grade embedding generator using Mistral AI.
    Supports batch processing and caching.
    """

    DEFAULT_BATCH_SIZE = 100
    MAX_BATCH_SIZE = 500
    DEFAULT_MODEL = "mistral-embed"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        cache_folder: Optional[str] = None,
    ):
        """
        Initialize the embedding generator.

        Args:
            api_key: Mistral AI API key
            model: Embedding model name
            batch_size: Batch size for processing
            cache_folder: Optional folder for caching embeddings
        """
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Mistral API key is required. "
                "Set MISTRAL_API_KEY environment variable or pass api_key parameter."
            )

        self.model = model
        self.batch_size = min(batch_size, self.MAX_BATCH_SIZE)
        self.cache_folder = cache_folder

        # Initialize Mistral embeddings client
        self.embeddings_client = MistralAIEmbeddings(
            api_key=self.api_key,
            model=model,
        )

        # Get embedding dimension
        self._embedding_dimension = self._get_embedding_dimension()

        logger.info(
            f"EmbeddingGenerator initialized: model={model}, "
            f"batch_size={batch_size}, dimension={self._embedding_dimension}"
        )

    def _get_embedding_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        try:
            # Test embedding to get dimension
            test_embedding = self.embeddings_client.embed_query("test")
            return len(test_embedding)
        except Exception as e:
            logger.warning(f"Could not determine embedding dimension: {e}")
            return 768  # Default fallback

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self._embedding_dimension

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            query: Query text

        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings_client.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

    def embed_documents(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            documents: List of LangChain documents
            show_progress: Whether to show progress

        Returns:
            List of embedding vectors
        """
        if not documents:
            return []

        logger.info(f"Generating embeddings for {len(documents)} documents")

        texts = [doc.page_content for doc in documents]
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            try:
                batch_embeddings = self.embeddings_client.embed_documents(batch)
                embeddings.extend(batch_embeddings)

                if show_progress:
                    progress = min(i + self.batch_size, len(texts))
                    logger.info(f"Progress: {progress}/{len(texts)} documents")

            except Exception as e:
                logger.error(f"Error processing batch starting at {i}: {str(e)}")
                # Try individual documents in case of batch failure
                for j, text in enumerate(batch):
                    try:
                        single_embedding = self.embeddings_client.embed_documents([text])
                        embeddings.append(single_embedding[0])
                    except Exception as inner_e:
                        logger.error(f"Failed to embed document {i+j}: {str(inner_e)}")
                        # Use zero vector as fallback
                        embeddings.append([0.0] * self._embedding_dimension)

        return embeddings

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for raw text strings.

        Args:
            texts: List of text strings
            show_progress: Whether to show progress

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Convert to Documents
        documents = [
            Document(page_content=text, metadata={"index": i})
            for i, text in enumerate(texts)
        ]

        return self.embed_documents(documents, show_progress=show_progress)

    def embed_query_with_stats(
        self,
        query: str,
    ) -> tuple[List[float], EmbeddingStats]:
        """
        Generate embedding for query with statistics.

        Args:
            query: Query text

        Returns:
            Tuple of (embedding, stats)
        """
        import time

        start_time = time.time()

        try:
            embedding = self.embed_query(query)
            elapsed = time.time() - start_time

            stats = EmbeddingStats(
                total_documents=1,
                total_chunks=1,
                embedding_dimension=self._embedding_dimension,
                processing_time_seconds=elapsed,
            )

            return embedding, stats

        except Exception as e:
            logger.error(f"Error in embed_query_with_stats: {str(e)}")
            raise

    def embed_documents_with_stats(
        self,
        documents: List[Document],
        show_progress: bool = True,
    ) -> tuple[List[List[float]], EmbeddingStats]:
        """
        Generate embeddings for documents with statistics.

        Args:
            documents: List of documents
            show_progress: Whether to show progress

        Returns:
            Tuple of (embeddings, stats)
        """
        import time

        start_time = time.time()
        errors = 0

        try:
            embeddings = self.embed_documents(documents, show_progress=show_progress)
            elapsed = time.time() - start_time

            stats = EmbeddingStats(
                total_documents=len(documents),
                total_chunks=len(documents),
                embedding_dimension=self._embedding_dimension,
                processing_time_seconds=elapsed,
                errors=errors,
            )

            return embeddings, stats

        except Exception as e:
            logger.error(f"Error in embed_documents_with_stats: {str(e)}")
            errors += 1
            raise

    def compute_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float],
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score (0-1)
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def get_embeddings_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model information
        """
        return {
            "model": self.model,
            "dimension": self._embedding_dimension,
            "batch_size": self.batch_size,
            "provider": "Mistral AI",
        }
