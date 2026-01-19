"""Data access layer for PMD.

This package provides the persistence layer for PMD including:
- Database: SQLite database connection and transaction management
- Repositories: Typed data access for all entity types
- Facades: Service-oriented wrappers over repositories

Facade Classes:
    IndexFacade: Facade for indexing operations.
    LoadFacade: Facade for document loading operations.
    SearchFacade: Facade for search operations.
    StatusFacade: Facade for status reporting.

Repository Classes:
    SourceCollectionRepository: Source collection CRUD
    DocumentRepository: Document CRUD with content-addressable storage
    ContentRepository: Content-addressable storage operations
    EmbeddingRepository: Vector embedding storage and similarity search
    FTS5SearchRepository: Full-text search using SQLite FTS5
    SourceMetadataRepository: HTTP source metadata
    DocumentMetadataRepository: Extracted document metadata

Example:
    from pmd.store import Database, IndexFacade

    db = Database("pmd.db")
    db.connect()
    facade = IndexFacade(db)
    collection = facade.get_collection_by_name("docs")
"""

import warnings

from .caching import ResourceCacher
from .database import Database
from .facade import DatasetFacade, IndexFacade, LoadFacade, SearchFacade, StatusFacade
from .repositories import (
    ContentRepository,
    DocumentMetadataRepository,
    DocumentRepository,
    EmbeddingRepository,
    FTS5SearchRepository,
    ResourceRepository,
    SearchRepository,
    SourceCollectionRepository,
    SourceMetadata,
    SourceMetadataRepository,
    _serialize_embedding,
)
from .vector_search import VectorSearchRepository

__all__ = [
    # Database
    "Database",
    # Caching
    "ResourceCacher",
    "DocumentCacher",  # Deprecated alias for ResourceCacher
    # Facades
    "DatasetFacade",
    "IndexFacade",
    "LoadFacade",
    "SearchFacade",
    "StatusFacade",
    # Repositories
    "ResourceRepository",
    "SourceCollectionRepository",
    "DocumentRepository",
    "ContentRepository",
    "EmbeddingRepository",
    "_serialize_embedding",
    "FTS5SearchRepository",
    "SearchRepository",
    "SourceMetadataRepository",
    "SourceMetadata",
    "DocumentMetadataRepository",
    # Adapters
    "VectorSearchRepository",
]


# Deprecated aliases - module-level __getattr__ for lazy deprecation warning
def __getattr__(name: str):
    if name == "DocumentCacher":
        warnings.warn(
            "DocumentCacher is deprecated, use ResourceCacher instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return ResourceCacher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
