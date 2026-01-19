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

from .caching import DocumentCacher
from .database import Database
from .facade import IndexFacade, LoadFacade, SearchFacade, StatusFacade
from .repositories import (
    CollectionRepository,
    ContentRepository,
    DocumentMetadataRepository,
    DocumentRepository,
    EmbeddingRepository,
    FTS5SearchRepository,
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
    "DocumentCacher",
    # Facades
    "IndexFacade",
    "LoadFacade",
    "SearchFacade",
    "StatusFacade",
    # Repositories
    "SourceCollectionRepository",
    "CollectionRepository",  # Deprecated alias
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
