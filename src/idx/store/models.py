"""idx.store.models - SQLAlchemy ORM models for idx.

Defines the database schema for Dataset and Document entities.

Dataset: Represents a data source (e.g., a directory, Obsidian vault).
Document: Represents a single indexed document within a dataset.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, ForeignKey, Index, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from idx.store.database import Base

__all__ = [
    "Dataset",
    "Document",
]


class Dataset(Base):
    """SQLAlchemy model for a dataset (data source).

    A Dataset represents a source repository that has been indexed,
    such as a local directory or an Obsidian vault.

    Attributes:
        id: Primary key.
        name: Unique name identifier for the dataset (normalized URI format).
        uri: Full URI for the dataset (e.g., "dataset:my-vault").
        source_type: Type of source ("directory", "obsidian", etc.).
        source_path: Filesystem path to the source.
        created_at: When the dataset was created.
        updated_at: When the dataset was last modified.
        documents: Relationship to associated documents.
    """

    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    uri: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationship to documents
    documents: Mapped[list["Document"]] = relationship(
        "Document", back_populates="dataset", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Dataset(id={self.id}, name='{self.name}', source_type='{self.source_type}')>"


class Document(Base):
    """SQLAlchemy model for an indexed document.

    A Document represents a single file or resource that has been
    processed and indexed as part of a dataset.

    Attributes:
        id: Primary key.
        dataset_id: Foreign key to parent dataset.
        path: Relative path within the dataset source.
        active: Whether the document is active (False = soft-deleted).
        content_hash: SHA256 hash of the document content.
        etag: Source-provided etag for change detection (optional).
        last_modified: Source-provided modification time (optional).
        body: Full normalized text content for FTS and chunking.
        metadata_json: Extracted metadata as JSON (frontmatter, tags, etc.).
        created_at: When the document was first indexed.
        updated_at: When the document was last updated.
        dataset: Relationship to parent dataset.
    """

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[int] = mapped_column(
        ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False, index=True
    )
    path: Mapped[str] = mapped_column(String(1024), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    etag: Mapped[str | None] = mapped_column(String(255), nullable=True)
    last_modified: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Relationship to dataset
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="documents")

    # Composite index for common queries
    __table_args__ = (
        Index("ix_documents_dataset_path", "dataset_id", "path", unique=True),
        Index("ix_documents_dataset_active", "dataset_id", "active"),
        Index("ix_documents_content_hash", "content_hash"),
    )

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, path='{self.path}', active={self.active})>"

    def get_metadata(self) -> dict[str, Any]:
        """Parse and return metadata as a dictionary.

        Returns:
            Parsed metadata dictionary, or empty dict if no metadata.
        """
        if not self.metadata_json:
            return {}
        import json

        try:
            return json.loads(self.metadata_json)
        except json.JSONDecodeError:
            return {}

    def set_metadata(self, value: dict[str, Any]) -> None:
        """Set metadata from a dictionary.

        Args:
            value: Metadata dictionary to serialize.
        """
        import json

        self.metadata_json = json.dumps(value)
