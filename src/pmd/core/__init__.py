"""Core business logic and types for PMD."""

from .config import (
    ChunkConfig,
    Config,
    LMStudioConfig,
    MLXConfig,
    OpenRouterConfig,
    SearchConfig,
)
from .exceptions import (
    CollectionError,  # Deprecated alias
    CollectionExistsError,  # Deprecated alias
    CollectionNotFoundError,  # Deprecated alias
    DatabaseError,
    DocumentError,
    DocumentNotFoundError,
    EmbeddingError,
    FormatError,
    LLMError,
    ModelNotFoundError,
    PMDError,
    SearchError,
    SourceCollectionError,
    SourceCollectionExistsError,
    SourceCollectionNotFoundError,
    VirtualPathError,
)
from .types import (
    Chunk,
    Collection,  # Deprecated alias
    DocumentNotFound,
    DocumentResult,
    EmbeddingResult,
    IndexStatus,
    OutputFormat,
    PathContext,
    RankedResult,
    RerankDocumentResult,
    RerankResult,
    SearchResult,
    SearchSource,
    SnippetResult,
    SourceCollection,
    VirtualPath,
)

__all__ = [
    "Config",
    "LMStudioConfig",
    "OpenRouterConfig",
    "MLXConfig",
    "SearchConfig",
    "ChunkConfig",
    "PMDError",
    "DatabaseError",
    "SourceCollectionError",
    "SourceCollectionNotFoundError",
    "SourceCollectionExistsError",
    "CollectionError",  # Deprecated alias
    "CollectionNotFoundError",  # Deprecated alias
    "CollectionExistsError",  # Deprecated alias
    "DocumentError",
    "DocumentNotFoundError",
    "LLMError",
    "ModelNotFoundError",
    "SearchError",
    "EmbeddingError",
    "FormatError",
    "VirtualPathError",
    "SearchSource",
    "OutputFormat",
    "VirtualPath",
    "SourceCollection",
    "Collection",  # Deprecated alias
    "DocumentResult",
    "SearchResult",
    "RankedResult",
    "EmbeddingResult",
    "RerankDocumentResult",
    "RerankResult",
    "PathContext",
    "Chunk",
    "SnippetResult",
    "DocumentNotFound",
    "IndexStatus",
]
