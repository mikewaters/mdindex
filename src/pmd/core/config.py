"""Configuration management for PMD."""

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
    embedding_model: str = "mlx-community/nomicai-modernbert-embed-base-4bit"
    # Embedding vector dimension (must match model output)
    # - nomic/modernbert-embed-base: 768
    # - multilingual-e5-small: 384
    embedding_dimension: int = 768
    # Prefix to add for query embeddings (model-specific)
    # - nomic/modernbert: "search_query: "
    # - e5 models: "query: "
    query_prefix: str = "search_query: "
    # Prefix to add for document embeddings (model-specific)
    # - nomic/modernbert: "search_document: "
    # - e5 models: "passage: "
    document_prefix: str = "search_document: "
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


@dataclass
class TracingConfig:
    """Phoenix/OpenTelemetry tracing configuration.

    Tracing is OFF by default. Enable via:
    - CLI: --phoenix-tracing
    - Config: [tracing] enabled = true
    - Env: PHOENIX_TRACING=1
    """

    enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    service_name: str = "pmd"
    service_version: str = "1.0.0"
    sample_rate: float = 1.0
    batch_export: bool = True


@dataclass
class MetadataConfig:
    """Document metadata extraction configuration.

    Controls how metadata (tags, attributes) is extracted from documents.

    Example TOML:
        [metadata]
        default_profile = "obsidian"
        expand_nested_tags = true

        [metadata.tag_namespace_map]
        "project" = "prj"
        "status" = "st"
    """

    # Default profile to use if auto-detection doesn't match
    default_profile: str = "generic"

    # Whether to expand nested tags (e.g., "parent/child" -> ["parent", "parent/child"])
    expand_nested_tags: bool = True

    # Tag namespace mapping for normalization (e.g., {"project" = "prj"})
    # Maps source prefixes to canonical prefixes
    tag_namespace_map: dict[str, str] = field(default_factory=dict)


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
    tracing: TracingConfig = field(default_factory=TracingConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        config = cls()
        _apply_env(config)
        return config

    @classmethod
    def from_file(cls, path: str | Path) -> "Config":
        """Load configuration from a TOML file, then apply environment overrides."""
        config = cls()
        data = _load_toml(path)
        _apply_toml(config, data)
        _apply_env(config)
        return config

    @classmethod
    def from_env_or_file(cls, path: str | Path | None = None) -> "Config":
        """Load config from file or env, honoring PMD_CONFIG when path not provided."""
        env_path = os.environ.get("PMD_CONFIG")
        chosen = path or env_path
        if chosen:
            return cls.from_file(chosen)
        return cls.from_env()

def _load_toml(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration file."""
    with open(path, "rb") as handle:
        return tomllib.load(handle)

def _apply_toml(config: Config, data: dict[str, Any]) -> None:
    """Apply TOML values onto a Config instance."""
    if db_path := data.get("db_path"):
        config.db_path = Path(db_path)
    if llm_provider := data.get("llm_provider"):
        config.llm_provider = llm_provider

    if isinstance(data.get("lm_studio"), dict):
        _update_dataclass(config.lm_studio, data["lm_studio"])
    if isinstance(data.get("openrouter"), dict):
        _update_dataclass(config.openrouter, data["openrouter"])
    if isinstance(data.get("mlx"), dict):
        _update_dataclass(config.mlx, data["mlx"])
    if isinstance(data.get("search"), dict):
        _update_dataclass(config.search, data["search"])
    if isinstance(data.get("chunk"), dict):
        _update_dataclass(config.chunk, data["chunk"])
    if isinstance(data.get("tracing"), dict):
        _update_dataclass(config.tracing, data["tracing"])
    if isinstance(data.get("metadata"), dict):
        _update_dataclass(config.metadata, data["metadata"])

def _update_dataclass(target: Any, values: dict[str, Any]) -> None:
    """Update dataclass fields from a dict (ignores unknown keys)."""
    for key, value in values.items():
        if hasattr(target, key):
            setattr(target, key, value)

def _apply_env(config: Config) -> None:
    """Apply environment variable overrides."""
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

    # Tracing configuration
    if os.environ.get("PHOENIX_TRACING", "").lower() in ("1", "true", "yes"):
        config.tracing.enabled = True
    if endpoint := os.environ.get("PHOENIX_ENDPOINT"):
        config.tracing.phoenix_endpoint = endpoint
    if sample_rate := os.environ.get("PHOENIX_SAMPLE_RATE"):
        try:
            config.tracing.sample_rate = float(sample_rate)
        except ValueError:
            pass
