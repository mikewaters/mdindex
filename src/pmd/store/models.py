"""SQLAlchemy ORM models for PMD storage layer.

This module defines the declarative ORM models mapped to the PMD database tables.
These models use SQLAlchemy 2.0 style with Mapped[] type annotations.

Note: FTS5 and sqlite-vec virtual tables (documents_fts, content_vectors_vec)
are not represented as ORM models since they require special handling.
"""

from enum import Enum
from typing import Optional

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class LoadStatus(str, Enum):
    """Status of resource content loading.

    Attributes:
        PENDING: Resource discovered but not yet loaded.
        LOADED: Content successfully loaded and available.
        ERROR: Loading failed with an error.
        SKIPPED: Resource intentionally skipped (e.g., filtered out).
    """

    PENDING = "pending"
    LOADED = "loaded"
    ERROR = "error"
    SKIPPED = "skipped"


class IndexState(str, Enum):
    """State of resource indexing.

    Attributes:
        PENDING: Resource loaded but not yet indexed.
        INDEXED: Resource fully indexed and searchable.
        ERROR: Indexing failed with an error.
        SKIPPED: Resource intentionally not indexed.
    """

    PENDING = "pending"
    INDEXED = "indexed"
    ERROR = "error"
    SKIPPED = "skipped"


class Base(DeclarativeBase):
    """Base class for all ORM models in PMD."""

    pass


class ContentModel(Base):
    """Content-addressable storage for document content.

    Each unique document body is stored once with its SHA256 hash as the key.
    Multiple documents can reference the same content via their hash foreign key.
    """

    __tablename__ = "content"

    hash: Mapped[str] = mapped_column(String, primary_key=True)
    doc: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)


class SourceCollectionModel(Base):
    """A source collection representing an indexed directory or remote source.

    Source collections define where documents come from and how they are
    discovered (via glob patterns and source-specific configuration).
    """

    __tablename__ = "source_collections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    pwd: Mapped[str] = mapped_column(String, nullable=False)
    glob_pattern: Mapped[str] = mapped_column(
        String, nullable=False, default="**/*.md"
    )
    source_type: Mapped[str] = mapped_column(
        String, nullable=False, default="filesystem"
    )
    source_config: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[str] = mapped_column(String, nullable=False)

    # Relationships
    documents: Mapped[list["DocumentModel"]] = relationship(
        "DocumentModel", back_populates="source_collection", cascade="all, delete-orphan"
    )


class DocumentModel(Base):
    """A document tracked within a source collection.

    Documents map file paths to content hashes and track metadata about
    the document's state (active/inactive, modification time).

    Source metadata columns (source_uri, etag, etc.) are merged from the
    former source_metadata table to allow single-row document representation.
    """

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("source_collection_id", "path", name="uq_documents_collection_path"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_collection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("source_collections.id"), nullable=False
    )
    path: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[str] = mapped_column(String, nullable=False)
    hash: Mapped[str] = mapped_column(
        String, ForeignKey("content.hash"), nullable=False
    )
    active: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    modified_at: Mapped[str] = mapped_column(String, nullable=False)

    # Merged source_metadata columns (nullable for filesystem sources)
    source_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    etag: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_modified: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    last_fetched_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fetch_duration_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    http_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    content_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    extra_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    source_collection: Mapped["SourceCollectionModel"] = relationship(
        "SourceCollectionModel", back_populates="documents"
    )
    content: Mapped["ContentModel"] = relationship("ContentModel")


class ContentVectorsModel(Base):
    """Metadata for vector embeddings stored in the content_vectors_vec virtual table.

    This table tracks which content chunks have been embedded, their position
    within the document, and the model used for embedding. The actual vectors
    are stored in the sqlite-vec virtual table (content_vectors_vec).
    """

    __tablename__ = "content_vectors"

    hash: Mapped[str] = mapped_column(String, primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    pos: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    embedded_at: Mapped[str] = mapped_column(String, nullable=False)


class ResourceModel(Base):
    """A resource tracked for fetch/index lifecycle management.

    Resources represent URIs that can be loaded and indexed independently of
    the documents table. This separates fetch/index state tracking from document
    storage, enabling more granular control over resource processing pipelines.

    The load_status tracks content fetching (pending -> loaded/error/skipped),
    while index_state tracks indexing (pending -> indexed/error/skipped).

    Attributes:
        id: Unique resource identifier.
        source_collection_id: Foreign key to the source collection.
        uri: Unique resource URI within the collection.
        resource_type: Optional type hint (e.g., 'markdown', 'pdf').
        hash: Content hash after loading (for deduplication).
        content_ref: Reference to stored content (e.g., content table hash).
        source_created_at: When the resource was created at the source.
        source_modified_at: When the resource was last modified at the source.
        loaded_at: When the content was successfully loaded.
        load_method: Method used for loading (e.g., 'http', 'filesystem').
        load_status: Current load status (pending/loaded/error/skipped).
        load_error: Error message if load_status is 'error'.
        indexed_at: When the resource was successfully indexed.
        index_state: Current index state (pending/indexed/error/skipped).
        index_method: Method used for indexing (e.g., 'llamaindex', 'manual').
        index_error: Error message if index_state is 'error'.
        resource_metadata: JSON blob for extensible metadata (maps to 'metadata' column).
        created_at: When the resource record was created.
        updated_at: When the resource record was last updated.
    """

    __tablename__ = "resources"
    __table_args__ = (
        UniqueConstraint(
            "source_collection_id", "uri", name="uq_resources_collection_uri"
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_collection_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("source_collections.id"), nullable=False
    )
    uri: Mapped[str] = mapped_column(Text, nullable=False)
    resource_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    hash: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    content_ref: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_created_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_modified_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    loaded_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    load_method: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    load_status: Mapped[str] = mapped_column(
        Text, nullable=False, default=LoadStatus.PENDING.value
    )
    load_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    indexed_at: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    index_state: Mapped[str] = mapped_column(
        Text, nullable=False, default=IndexState.PENDING.value
    )
    index_method: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    index_error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Note: 'metadata' is reserved in SQLAlchemy, so we use 'resource_metadata'
    # as the Python attribute but map it to the 'metadata' column in the database.
    resource_metadata: Mapped[Optional[str]] = mapped_column(
        "metadata", Text, nullable=True
    )
    created_at: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[str] = mapped_column(Text, nullable=False)

    # Relationships
    source_collection: Mapped["SourceCollectionModel"] = relationship(
        "SourceCollectionModel"
    )
