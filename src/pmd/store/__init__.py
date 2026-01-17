"""Data access layer for PMD.

This package provides the persistence layer for PMD including:
- Database: SQLite database connection and transaction management
- Repositories: Typed data access for all entity types

Repository Classes:
    SourceCollectionRepository: Source collection CRUD
    DocumentRepository: Document CRUD with content-addressable storage
    ContentRepository: Content-addressable storage operations
    EmbeddingRepository: Vector embedding storage and similarity search
    FTS5SearchRepository: Full-text search using SQLite FTS5
    SourceMetadataRepository: HTTP source metadata
    DocumentMetadataRepository: Extracted document metadata

Example:
    from pmd.store import Database
    from pmd.store.repositories import (
        SourceCollectionRepository,
        DocumentRepository,
    )

    db = Database("pmd.db")
    collections = SourceCollectionRepository(db)
    documents = DocumentRepository(db)
"""

from .database import Database
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
