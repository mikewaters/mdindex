"""Type definitions for PMD."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypedDict


class SearchSource(Enum):
    """Source of search result."""

    FTS = "fts"
    VECTOR = "vec"
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
    """Represents an indexed collection (directory)."""

    id: int
    name: str
    pwd: str  # Base directory path
    glob_pattern: str
    created_at: str
    updated_at: str


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
    """Result after RRF fusion and reranking."""

    file: str
    display_path: str
    title: str
    body: str
    score: float
    fts_score: Optional[float] = None
    vec_score: Optional[float] = None
    rerank_score: Optional[float] = None


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
