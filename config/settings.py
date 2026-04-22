import os
from pathlib import Path
from typing import Any, List, Optional

try:
    from pydantic import Field, field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback to satisfy IDE or environment mismatches
    from pydantic.v1 import Field
    from pydantic import field_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @field_validator("allowed_extensions", "cors_origins", mode="before")
    @classmethod
    def parse_list(cls, v: Any) -> Any:
        """Parse a list from a comma-separated string if needed."""
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("[") and v.endswith("]"):
                import json
                try:
                    return json.loads(v)
                except:
                    pass
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    # Application Metadata
    version: str = "2.1.0-beta"
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Global logging level")

    # Mistral AI Configuration
    mistral_api_key: str = Field(
        default="",
        description="Mistral AI API key for LLM and embeddings"
    )

    # ChromaDB Configuration
    chroma_db_path: str = Field(
        default="./database/chroma_db",
        description="Path to ChromaDB persistent storage"
    )

    # Model Configuration
    embedding_model: str = Field(
        default="mistral-embed",
        description="Embedding model name"
    )
    llm_model: str = Field(
        default="mistral-large-latest",
        description="LLM model name for question answering"
    )

    # API Configuration
    api_host: str = Field(
        default="0.0.0.0",
        description="FastAPI server host"
    )
    api_port: int = Field(
        default=8000,
        description="FastAPI server port"
    )

    # Document Processing Configuration
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum allowed file size in MB"
    )
    allowed_extensions: Any = Field(
        default=["pdf", "txt", "md", "html"],
        description="Allowed file extensions for upload"
    )

    # RAG Configuration
    chunk_size: int = Field(
        default=1024,
        description="Text chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=128,
        description="Chunk overlap in tokens"
    )
    top_k_results: int = Field(
        default=5,
        description="Number of top results to retrieve"
    )

    # Caching Configuration
    cache_enabled: bool = Field(
        default=True,
        description="Enable caching for repeated queries"
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache time-to-live in seconds"
    )

    # Security Configuration
    api_key_header: str = Field(
        default="X-API-Key",
        description="Header name for API key authentication"
    )
    cors_origins: Any = Field(
        default=["*"],
        description="CORS allowed origins"
    )

    # Application Paths
    @property
    def database_path(self) -> Path:
        return Path(self.chroma_db_path).resolve()

    @property
    def uploads_path(self) -> Path:
        return Path("./uploads").resolve()

    @property
    def cache_path(self) -> Path:
        return Path("./cache").resolve()

    def ensure_directories(self) -> None:
        """Initialize operational file system structures."""
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.uploads_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

    def validate_api_key(self, provided_key: str) -> bool:
        """Validate API key."""
        env_key = os.getenv("APP_API_KEY")
        if not env_key: return True
        return provided_key == env_key


# Global settings instance
settings = Settings()
settings.ensure_directories()
