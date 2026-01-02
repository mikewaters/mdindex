"""Type definitions for PMD."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypedDict


class SearchSource(Enum):
    """Source of search result.

    Attributes:
        FTS: Full-text search (BM25).
        VECTOR: Vector similarity search.
        TAG: Tag-based retrieval.
        HYBRID: Combination of sources (via RRF).
    """

    FTS = "fts"
    VECTOR = "vec"
    TAG = "tag"
    HYBRID = "hybrid"


class OutputFormat(Enum):
    """Output format for results."""

    JSON = "json"
    CSV = "csv"
    XML = "xml"
    MARKDOWN = "md"
    FILES = "files"
    CLI = "cli"


@dataclass(frozen=True)
class VirtualPath:
    """Represents a pmd:// URI path."""

    collection_name: str
    path: str

    def __str__(self) -> str:
        """Convert to string representation."""
        return f"pmd://{self.collection_name}/{self.path}"


@dataclass
class Collection:
    """Represents an indexed collection.

    Collections can source documents from different backends:
    - filesystem: Local directory with glob pattern (default)
    - http: Remote HTTP/HTTPS endpoint
    - entity: Custom entity URI with pluggable resolvers

    Attributes:
        id: Unique collection ID.
        name: Human-readable collection name.
        pwd: Base directory path (for filesystem sources).
        glob_pattern: File pattern to match (for filesystem sources).
        source_type: Type of source ('filesystem', 'http', 'entity').
        source_config: JSON config for non-filesystem sources.
        created_at: When the collection was created.
        updated_at: When the collection was last modified.
    """

    id: int
    name: str
    pwd: str  # Base directory path (filesystem) or base URI (other)
    glob_pattern: str
    created_at: str
    updated_at: str
    source_type: str = "filesystem"
    source_config: Optional[dict] = None

    def get_source_uri(self) -> str:
        """Get the source URI for this collection.

        Returns:
            URI string appropriate for the source type.
        """
        if self.source_type == "filesystem":
            from pathlib import Path
            return Path(self.pwd).resolve().as_uri()
        elif self.source_type in ("http", "https"):
            return self.source_config.get("base_url", self.pwd) if self.source_config else self.pwd
        elif self.source_type == "entity":
            return self.source_config.get("uri", self.pwd) if self.source_config else self.pwd
        else:
            return self.pwd

    def get_source_config_dict(self) -> dict:
        """Get source configuration as a dictionary.

        Returns:
            Configuration dict including type-specific settings.
        """
        base = {
            "source_type": self.source_type,
            "uri": self.get_source_uri(),
        }
        if self.source_type == "filesystem":
            base["glob_pattern"] = self.glob_pattern
        if self.source_config:
            base.update(self.source_config)
        return base


@dataclass
class DocumentResult:
    """Represents a retrieved document."""

    filepath: str
    display_path: str
    title: str
    context: Optional[str]
    hash: str
    collection_id: int
    modified_at: str
    body_length: int
    body: Optional[str] = None


@dataclass
class SearchResult(DocumentResult):
    """Extends DocumentResult with search-specific fields."""

    score: float = 0.0
    source: SearchSource = SearchSource.FTS
    chunk_pos: Optional[int] = None
    snippet: Optional[str] = None


@dataclass
class RankedResult:
    """Result after RRF fusion and reranking.

    Attributes:
        file: Document filepath.
        display_path: Path for display (may be relative).
        title: Document title.
        body: Document content.
        score: Final blended/normalized score.
        fts_score: Original FTS5 BM25 score (if found via FTS).
        vec_score: Original vector similarity score (if found via vector).
        rerank_score: LLM reranker score (if reranking enabled).
        fts_rank: Original rank in FTS results (0-indexed, None if not found).
        vec_rank: Original rank in vector results (0-indexed, None if not found).
        sources_count: Number of sources that found this document (1 or 2).
        relevant: LLM relevance judgment (True/False/None).
        rerank_confidence: LLM reranker confidence (0-1).
        rerank_raw_token: Raw token from reranker ("Yes"/"No").
        blend_weight: Position-aware blend weight used (0.75/0.60/0.40).
    """

    file: str
    display_path: str
    title: str
    body: str
    score: float
    fts_score: Optional[float] = None
    vec_score: Optional[float] = None
    rerank_score: Optional[float] = None
    # Fusion provenance
    fts_rank: Optional[int] = None
    vec_rank: Optional[int] = None
    sources_count: int = 1
    # Reranker details
    relevant: Optional[bool] = None
    rerank_confidence: Optional[float] = None
    rerank_raw_token: Optional[str] = None
    blend_weight: Optional[float] = None


@dataclass
class EmbeddingResult:
    """Result from embedding generation."""

    embedding: list[float]
    model: str


@dataclass
class RerankDocumentResult:
    """Single document reranking result."""

    file: str
    relevant: bool
    confidence: float
    score: float
    raw_token: str
    logprob: Optional[float] = None


@dataclass
class RerankResult:
    """Complete reranking result."""

    results: list[RerankDocumentResult]
    model: str


@dataclass
class PathContext:
    """Context description for a path prefix."""

    id: int
    collection_id: int
    path_prefix: str
    context: str
    created_at: str


@dataclass
class Chunk:
    """Document chunk for embedding."""

    text: str
    pos: int  # Character position in original document


@dataclass
class SnippetResult:
    """Extracted snippet with match position."""

    text: str
    match_start: int
    match_end: int


class DocumentNotFound(TypedDict):
    """Error result when document not found."""

    error: str
    suggestions: list[str]


@dataclass
class IndexStatus:
    """Status information for the index."""

    collections: list[Collection]
    total_documents: int
    embedded_documents: int
    index_size_bytes: int
    cache_entries: int
    ollama_available: bool
    models_available: dict[str, bool]
