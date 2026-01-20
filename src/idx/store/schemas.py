"""idx.store.schemas - Pydantic models for store operations.

These models define the input/output contracts for the DatasetService.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "DatasetCreate",
    "DatasetInfo",
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentInfo",
]


class DatasetCreate(BaseModel):
    """Input model for creating a new dataset."""

    name: str = Field(..., description="Unique name identifier for the dataset")
    source_type: str = Field(..., description="Type of source (directory, obsidian, etc.)")
    source_path: str = Field(..., description="Filesystem path to the source")

    model_config = {"frozen": True}


class DatasetInfo(BaseModel):
    """Output model representing a dataset."""

    id: int
    name: str
    uri: str
    source_type: str
    source_path: str
    created_at: datetime
    updated_at: datetime
    document_count: int = 0

    model_config = {"from_attributes": True}


class DocumentCreate(BaseModel):
    """Input model for creating a new document."""

    path: str = Field(..., description="Relative path within the dataset source")
    content_hash: str = Field(..., description="SHA256 hash of the document content")
    body: str = Field(..., description="Full normalized text content")
    etag: str | None = Field(None, description="Source-provided etag for change detection")
    last_modified: datetime | None = Field(None, description="Source-provided modification time")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Extracted metadata")

    model_config = {"frozen": True}


class DocumentUpdate(BaseModel):
    """Input model for updating an existing document."""

    content_hash: str | None = Field(None, description="New content hash if content changed")
    body: str | None = Field(None, description="New text content if changed")
    etag: str | None = Field(None, description="Updated etag")
    last_modified: datetime | None = Field(None, description="Updated modification time")
    metadata: dict[str, Any] | None = Field(None, description="Updated metadata")
    active: bool | None = Field(None, description="Whether the document is active")

    model_config = {"frozen": True}


class DocumentInfo(BaseModel):
    """Output model representing a document."""

    id: int
    dataset_id: int
    path: str
    active: bool
    content_hash: str
    etag: str | None
    last_modified: datetime | None
    body: str
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_model(cls, doc: Any) -> "DocumentInfo":
        """Create from an ORM Document model."""
        return cls(
            id=doc.id,
            dataset_id=doc.dataset_id,
            path=doc.path,
            active=doc.active,
            content_hash=doc.content_hash,
            etag=doc.etag,
            last_modified=doc.last_modified,
            body=doc.body,
            metadata=doc.get_metadata(),
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
