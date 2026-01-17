"""Repository implementations for PMD data access layer.

This package contains all repository classes that provide database access
for different entity types:

- SourceCollectionRepository: CRUD for source collections
- DocumentRepository: CRUD for documents with content-addressable storage
- ContentRepository: Content-addressable storage operations
- EmbeddingRepository: Vector embedding storage and similarity search
- FTS5SearchRepository: Full-text search using SQLite FTS5
- SourceMetadataRepository: HTTP source metadata (ETags, Last-Modified, etc.)
- DocumentMetadataRepository: Extracted document metadata (tags, attributes)

All repositories follow a consistent pattern:
- Accept a Database instance in __init__
- Provide typed methods for CRUD operations
- Handle transactions internally where needed

Example:
    from pmd.store.database import Database
    from pmd.store.repositories import (
        SourceCollectionRepository,
        DocumentRepository,
        FTS5SearchRepository,
    )

    db = Database(":memory:")
    collections = SourceCollectionRepository(db)
    documents = DocumentRepository(db)
    search = FTS5SearchRepository(db)
"""

from .collections import CollectionRepository, SourceCollectionRepository
from .content import ContentRepository
from .documents import DocumentRepository
from .embeddings import EmbeddingRepository, _serialize_embedding
from .fts import FTS5SearchRepository, SearchRepository
from .metadata import DocumentMetadataRepository
from .source_metadata import SourceMetadata, SourceMetadataRepository

__all__ = [
    # Collections
    "SourceCollectionRepository",
    "CollectionRepository",  # Deprecated alias
    # Documents
    "DocumentRepository",
    # Content
    "ContentRepository",
    # Embeddings
    "EmbeddingRepository",
    "_serialize_embedding",
    # FTS Search
    "FTS5SearchRepository",
    "SearchRepository",
    # Source Metadata
    "SourceMetadataRepository",
    "SourceMetadata",
    # Document Metadata
    "DocumentMetadataRepository",
]
