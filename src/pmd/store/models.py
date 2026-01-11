"""SQLAlchemy ORM models for PMD storage layer.

This module defines the declarative ORM models mapped to the PMD database tables.
These models use SQLAlchemy 2.0 style with Mapped[] type annotations.

Note: FTS5 and sqlite-vec virtual tables (documents_fts, content_vectors_vec)
are not represented as ORM models since they require special handling.
"""

from typing import Optional

from sqlalchemy import ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


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
