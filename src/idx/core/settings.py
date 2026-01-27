"""idx.core.settings - Library configuration via pydantic-settings.

Supports environment variables first; config-file support is deferred.

All settings use the IDX_ prefix for environment variables.

Example usage:
    from idx.core.settings import get_settings

    settings = get_settings()
    print(settings.database_path)

Environment variables:
    IDX_DATABASE_PATH - Path to SQLite database
    IDX_VECTOR_STORE_PATH - LlamaIndex persist directory (rebuildable cache)
    IDX_EMBEDDING_MODEL - Name/path of embedding model
    IDX_TRANSFORMERS_MODEL - Name/path of transformers model for reranking
    IDX_LOG_LEVEL - Logging level (default: INFO)
    IDX_BATCH_SIZE - Default batch size for processing
    IDX_CONCURRENCY - Default concurrency level
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class LangfuseSettings(BaseSettings):
    """Langfuse observability settings (placeholder for future implementation).

    These settings will be used to configure Langfuse integration for
    observability and tracing of LLM calls.
    """

    model_config = SettingsConfigDict(
        env_prefix="IDX_LANGFUSE_",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable Langfuse observability integration",
    )
    public_key: str | None = Field(
        default=None,
        description="Langfuse public key",
    )
    secret_key: str | None = Field(
        default=None,
        description="Langfuse secret key",
    )
    host: str | None = Field(
        default=None,
        description="Langfuse host URL (for self-hosted instances)",
    )


class EmbeddingSettings(BaseSettings):
    """Embedding configuration for vector generation.

    Controls which embedding backend to use and model configuration.
    Supports MLX (Apple Silicon) and HuggingFace backends.
    """

    model_config = SettingsConfigDict(
        env_prefix="IDX_EMBEDDING_",
        extra="ignore",
    )

    backend: Literal["mlx", "huggingface"] = Field(
        default="mlx",
        description="Embedding backend: 'mlx' for Apple Silicon, 'huggingface' for general",
    )
    model_name: str = Field(
        default="mlx-community/all-MiniLM-L6-v2-bf16",
        description="Name or path of the embedding model",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for embedding generation",
    )


class PerformanceSettings(BaseSettings):
    """Default performance settings for batch processing and concurrency."""

    model_config = SettingsConfigDict(
        env_prefix="IDX_",
        extra="ignore",
    )

    batch_size: int = Field(
        default=100,
        ge=1,
        description="Default batch size for document processing",
    )
    concurrency: int = Field(
        default=4,
        ge=1,
        description="Default concurrency level for parallel operations",
    )
    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for embedding generation (deprecated: use embedding.batch_size)",
    )
    chunk_max_bytes: int = Field(
        default=2048,
        ge=256,
        description="Maximum bytes per chunk for embedding",
    )
    chunk_min_bytes: int = Field(
        default=128,
        ge=1,
        description="Minimum bytes per chunk (fragments smaller than this are merged)",
    )


class Settings(BaseSettings):
    """Main configuration settings for the idx library.

    Configuration is loaded from environment variables with the IDX_ prefix.
    Config-file support is deferred for the MVP.

    Attributes:
        database_path: Path to the SQLite database file.
        vector_store_path: Path to the LlamaIndex persist directory (rebuildable cache).
        embedding_model: Name or path of the embedding model (deprecated).
        transformers_model: Name or path of the transformers model for reranking.
        log_level: Logging level for the library.
        embedding: Embedding model configuration (backend, model_name, batch_size).
        langfuse: Langfuse observability settings (placeholder).
        performance: Default performance settings for batch processing.
    """

    model_config = SettingsConfigDict(
        env_prefix="IDX_",
        env_nested_delimiter="__",
        extra="ignore",
        validate_default=True,
    )

    # Core paths
    database_path: Path = Field(
        default=Path("~/.idx/idx.db").expanduser(),
        description="Path to the SQLite database file",
    )
    vector_store_path: Path = Field(
        default=Path("~/.idx/vector_store").expanduser(),
        description="Path to LlamaIndex persist directory (rebuildable cache)",
    )
    cache_path: Path = Field(
        default=Path("~/.idx/cache").expanduser(),
        description="Path to cache directory for temporary files",
    )

    # Model configuration
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Name or path of the embedding model",
    )
    transformers_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Name or path of the transformers model for reranking",
    )

    # Logging
    log_level: LogLevel = Field(
        default="INFO",
        description="Logging level for the library",
    )

    # Nested settings
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="Embedding model configuration",
    )
    langfuse: LangfuseSettings = Field(
        default_factory=LangfuseSettings,
        description="Langfuse observability settings (placeholder)",
    )
    performance: PerformanceSettings = Field(
        default_factory=PerformanceSettings,
        description="Default performance settings",
    )

    def ensure_directories(self) -> None:
        """Ensure that required directories exist.

        Creates the parent directories for database_path and vector_store_path
        if they do not already exist.
        """
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get the singleton Settings instance.

    Returns a cached Settings instance, creating it on first call.
    The settings are loaded from environment variables.

    Returns:
        The singleton Settings instance.

    Example:
        settings = get_settings()
        print(settings.database_path)
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
