"""idx.pipelines - Orchestration layer.

Contains Ingestion Pipeline and Retrieval Pipeline.
Client entry point with uncomplicated Pydantic model interfaces.
"""

from idx.pipelines.ingest import IngestPipeline, compute_content_hash
from idx.pipelines.schemas import (
    DocumentStats,
    IngestDirectoryConfig,
    IngestObsidianConfig,
    IngestResult,
)

__all__ = [
    "DocumentStats",
    "IngestDirectoryConfig",
    "IngestObsidianConfig",
    "IngestPipeline",
    "IngestResult",
    "compute_content_hash",
]
