"""Workflow pipeline implementations.

This subpackage contains the concrete pipeline orchestrators:
- IngestionPipeline: Document loading and persistence
- EmbeddingPipeline: Chunk and embed documents

Each pipeline is a class that orchestrates node execution while
delegating actual work to injected protocol implementations.
"""

from pmd.workflows.pipelines.ingestion import IngestionPipeline
from pmd.workflows.pipelines.embedding import EmbeddingPipeline

__all__ = [
    "IngestionPipeline",
    "EmbeddingPipeline",
]
