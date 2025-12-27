"""Configuration management for PMD."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LMStudioConfig:
    """LM Studio service configuration (default)."""

    base_url: str = "http://localhost:1234"
    embedding_model: str = "nomic-embed-text"
    expansion_model: str = "qwen2:0.5b"
    reranker_model: str = "qwen2:0.5b"
    timeout: float = 120.0


@dataclass
class OpenRouterConfig:
    """OpenRouter API configuration."""

    api_key: str = ""
    base_url: str = "https://openrouter.io/api/v1"
    embedding_model: str = "nomic-ai/nomic-embed-text"
    expansion_model: str = "qwen/qwen-1.5-0.5b"
    reranker_model: str = "qwen/qwen-1.5-0.5b"
    timeout: float = 120.0


@dataclass
class MLXConfig:
    """MLX local model configuration for Apple Silicon."""

    # Model to use for text generation (query expansion, reranking)
    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    # Model to use for embeddings (via mlx-embeddings)
    embedding_model: str = "mlx-community/multilingual-e5-small-mlx"
    # Maximum tokens to generate
    max_tokens: int = 256
    # Sampling temperature
    temperature: float = 0.7
    # Whether to lazy-load models (load on first use)
    lazy_load: bool = True


@dataclass
class SearchConfig:
    """Search pipeline configuration."""

    default_limit: int = 5
    fts_weight: float = 1.0
    vec_weight: float = 1.0
    rrf_k: int = 60
    top_rank_bonus: float = 0.05
    expansion_weight: float = 0.5
    rerank_candidates: int = 30


@dataclass
class ChunkConfig:
    """Document chunking configuration."""

    max_bytes: int = 6 * 1024  # ~2000 tokens
    min_chunk_size: int = 100


def _default_db_path() -> Path:
    """Get default database path."""
    cache_dir = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_dir / "pmd" / "index.db"


@dataclass
class Config:
    """Main application configuration."""

    db_path: Path = field(default_factory=_default_db_path)
    llm_provider: str = "mlx"  # Default LLM provider (local inference on Apple Silicon)
    lm_studio: LMStudioConfig = field(default_factory=LMStudioConfig)
    openrouter: OpenRouterConfig = field(default_factory=OpenRouterConfig)
    mlx: MLXConfig = field(default_factory=MLXConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()

        # LLM Provider selection
        if provider := os.environ.get("LLM_PROVIDER"):
            config.llm_provider = provider

        # LM Studio configuration
        if url := os.environ.get("LM_STUDIO_URL"):
            config.lm_studio.base_url = url

        # OpenRouter configuration
        if api_key := os.environ.get("OPENROUTER_API_KEY"):
            config.openrouter.api_key = api_key

        if url := os.environ.get("OPENROUTER_URL"):
            config.openrouter.base_url = url

        # MLX configuration
        if model := os.environ.get("MLX_MODEL"):
            config.mlx.model = model
        if embedding_model := os.environ.get("MLX_EMBEDDING_MODEL"):
            config.mlx.embedding_model = embedding_model

        if path := os.environ.get("INDEX_PATH"):
            config.db_path = Path(path)

        return config
