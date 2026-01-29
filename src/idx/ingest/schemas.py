"""Pydantic schemas for pipeline input/output.

Defines the models used for configuring and reporting pipeline results.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field, model_validator



class IngestDirectoryConfig(BaseModel):
    """Configuration for directory ingestion.

    Attributes:
        source_path: Path to the directory to ingest.
        dataset_name: Name for the dataset (will be normalized).
        patterns: Glob patterns for matching files (default: ["**/*.md"]).
        encoding: File encoding to use (default: utf-8).
        force: If True, reprocess all documents even if unchanged.
    """

    source_path: Path
    dataset_name: str
    patterns: list[str] = Field(default_factory=lambda: ["**/*.md"])
    encoding: str = "utf-8"
    force: bool = False

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_source_path(self) -> "IngestDirectoryConfig":
        """Validate that the source path exists and is a directory."""
        from idx.ingest.directory import DirectorySource
        DirectorySource.validate(self.source_path)
        return self


class IngestObsidianConfig(BaseModel):
    """Configuration for Obsidian vault ingestion.

    Attributes:
        source_path: Path to the Obsidian vault.
        dataset_name: Name for the dataset (will be normalized).
        force: If True, reprocess all documents even if unchanged.
    """

    source_path: Path
    dataset_name: str
    force: bool = False

    model_config = {"arbitrary_types_allowed": True}

    @model_validator(mode="after")
    def validate_source_path(self) -> "IngestObsidianConfig":
        """Validate that the source path exists and is a directory."""
        from idx.ingest.obsidian import ObsidianVaultSource
        ObsidianVaultSource.validate(self.source_path)
        return self
    
    @model_validator(mode='before')
    def post_update(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values.get('source_path', False):
            if not values.get('dataset_name', False):
                values['dataset_name'] = values['source_path'].name

        return values

class DocumentStats(BaseModel):
    """Statistics for a single document ingestion.

    Attributes:
        path: Document path.
        action: What happened (created, updated, skipped).
        content_hash: SHA256 hash of the content.
    """

    path: str
    action: str  # "created", "updated", "skipped"
    content_hash: str


class IngestResult(BaseModel):
    """Result of an ingestion operation.

    Attributes:
        dataset_id: ID of the dataset.
        dataset_name: Normalized name of the dataset.
        documents_read: Total number of documents read.
        documents_created: Number of new documents created.
        documents_updated: Number of documents updated.
        documents_skipped: Number of unchanged documents skipped.
        documents_stale: Number of documents marked as stale (soft-deleted).
        documents_failed: Number of documents that failed to process.
        chunks_created: Number of chunks indexed in FTS.
        vectors_inserted: Number of vectors inserted into vector store.
        started_at: When the ingestion started.
        completed_at: When the ingestion completed.
        errors: List of error messages if any.
    """

    dataset_id: int
    dataset_name: str
    documents_read: int = 0
    documents_created: int = 0
    documents_updated: int = 0
    documents_skipped: int = 0
    documents_stale: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    vectors_inserted: int = 0
    started_at: datetime
    completed_at: datetime | None = None
    errors: list[str] = Field(default_factory=list)

    @property
    def total_processed(self) -> int:
        """Total documents processed successfully."""
        return self.documents_created + self.documents_updated + self.documents_skipped

    @property
    def success(self) -> bool:
        """Whether the ingestion completed without errors."""
        return self.documents_failed == 0 and len(self.errors) == 0


__all__ = [
    "DocumentStats",
    "IngestDirectoryConfig",
    "IngestObsidianConfig",
    "IngestResult",
]
