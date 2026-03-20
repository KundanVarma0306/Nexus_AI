"""
AI Research Assistant - Document Loader Module
Supports PDF, TXT, MD, HTML files and web URLs
"""

import hashlib
import logging
import re
import tempfile
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
    BSHTMLLoader,
)
from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentMetadata(BaseModel):
    """Metadata for loaded documents."""
    source: str
    source_type: str  # pdf, txt, url, html
    title: str = ""
    author: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    file_size: int = 0
    url: Optional[str] = None
    chunk_count: int = 0


class DocumentLoader:
    """
    Universal document loader supporting multiple file types and URLs.
    Production-grade with error handling and validation.
    """

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm"}
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_URL_SIZE = 10 * 1024 * 1024  # 10MB

    def __init__(self):
        self.loaded_documents: List[Document] = []
        self.metadata: List[DocumentMetadata] = []

    def load_file(self, file_path: Union[str, Path]) -> List[Document]:
        """
        Load a document from a file path.

        Args:
            file_path: Path to the document file

        Returns:
            List of LangChain Document objects
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size} bytes. "
                f"Maximum size: {self.MAX_FILE_SIZE} bytes"
            )

        logger.info(f"Loading file: {path.name} ({file_size} bytes)")

        try:
            if extension == ".pdf":
                documents = self._load_pdf(path)
            elif extension in {".txt", ".md"}:
                documents = self._load_text(path)
            elif extension in {".html", ".htm"}:
                documents = self._load_html(path)
            else:
                raise ValueError(f"Unsupported extension: {extension}")

            # Add source metadata to each document
            for doc in documents:
                doc.metadata["source"] = str(path)
                doc.metadata["source_type"] = extension[1:]
                doc.metadata["file_name"] = path.name
                doc.metadata["file_size"] = file_size
                doc.metadata["loaded_at"] = datetime.now().isoformat()

            self.loaded_documents.extend(documents)
            return documents

        except Exception as e:
            logger.error(f"Error loading file {path}: {str(e)}")
            raise

    def load_url(self, url: str, timeout: int = 30) -> List[Document]:
        """
        Load content from a web URL.

        Args:
            url: The URL to load content from
            timeout: Request timeout in seconds

        Returns:
            List of LangChain Document objects
        """
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid URL: {url}")

        logger.info(f"Loading URL: {url}")

        try:
            # Use WebBaseLoader for general URLs
            loader = WebBaseLoader(
                web_paths=(url,),
                bs_kwargs={
                    "parse_only": BSHTMLParser(
                        exclude_tags=["script", "style", "nav", "footer", "header"]
                    )
                },
                requests_per_second=1,
                requests_timeout=timeout,
            )

            # Attempt with standard loader first
            try:
                documents = loader.load()
            except Exception:
                # Fallback to direct requests if WebBaseLoader fails
                documents = self._load_url_fallback(url, timeout)

            # Add source metadata
            for doc in documents:
                doc.metadata["source"] = url
                doc.metadata["source_type"] = "url"
                doc.metadata["url"] = url
                doc.metadata["loaded_at"] = datetime.now().isoformat()

            self.loaded_documents.extend(documents)
            return documents

        except Exception as e:
            logger.error(f"Error loading URL {url}: {str(e)}")
            raise

    def load_multiple(self, sources: List[Union[str, Path]]) -> List[Document]:
        """
        Load multiple documents from different sources.

        Args:
            sources: List of file paths or URLs

        Returns:
            Combined list of all loaded documents
        """
        all_documents = []

        for source in sources:
            try:
                if self._is_url(str(source)):
                    documents = self.load_url(str(source))
                else:
                    documents = self.load_file(source)
                all_documents.extend(documents)
            except Exception as e:
                logger.warning(f"Failed to load {source}: {str(e)}")
                continue

        return all_documents

    def _load_pdf(self, path: Path) -> List[Document]:
        """Load PDF file using PyPDFLoader."""
        loader = PyPDFLoader(str(path))
        documents = loader.load()

        # Extract title from first page if available
        if documents:
            title_match = re.match(r"^[A-Z][^\n]+", documents[0].page_content)
            if title_match:
                title = title_match.group(0)[:100]
                for doc in documents:
                    doc.metadata["title"] = title

        return documents

    def _load_text(self, path: Path) -> List[Document]:
        """Load text/markdown file."""
        # Try UTF-8 first, then fallback to other encodings
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                loader = TextLoader(str(path), encoding=encoding)
                documents = loader.load()

                # Extract title from first line
                if documents:
                    first_line = documents[0].page_content.split("\n")[0].strip()
                    if first_line and len(first_line) < 100:
                        for doc in documents:
                            doc.metadata["title"] = first_line

                return documents
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode file with any supported encoding")

    def _load_html(self, path: Path) -> List[Document]:
        """Load HTML file."""
        loader = BSHTMLLoader(str(path), open_encoding="utf-8")
        documents = loader.load()

        # Extract title from HTML
        if documents:
            soup = BeautifulSoup(documents[0].page_content, "html.parser")
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text().strip()
                for doc in documents:
                    doc.metadata["title"] = title

        return documents

    def _load_url_fallback(self, url: str, timeout: int) -> List[Document]:
        """Fallback method to load URL using requests and BeautifulSoup."""
        response = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        response.raise_for_status()

        if len(response.content) > self.MAX_URL_SIZE:
            raise ValueError(f"URL content too large: {len(response.content)} bytes")

        # Parse HTML content
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove unwanted elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Get text content
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)

        document = Document(
            page_content=text,
            metadata={
                "source": url,
                "source_type": "url",
                "url": url,
                "title": soup.title.string if soup.title else urlparse(url).netloc,
            }
        )

        return [document]

    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        try:
            result = urlparse(source)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def get_document_hash(self, content: str) -> str:
        """Generate hash for document content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def clear(self) -> None:
        """Clear loaded documents."""
        self.loaded_documents.clear()
        self.metadata.clear()


class BSHTMLParser:
    """Custom BeautifulSoup HTML parser for WebBaseLoader."""

    def __init__(self, exclude_tags: List[str] = None):
        self.exclude_tags = exclude_tags or ["script", "style"]

    def __call__(self, html: str) -> BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(self.exclude_tags):
            tag.decompose()
        return soup
