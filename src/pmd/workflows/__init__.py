"""Workflow pipelines for document ingestion and embedding.

This package contains DAG-style workflow orchestration for:
- Document ingestion (enumerate -> load -> persist -> cleanup)
- Embedding generation (chunk -> embed -> store)

The workflows depend on abstract protocols rather than concrete implementations,
enabling testability with in-memory fakes and swappable components.

Architecture notes:
- Contracts define input/output dataclasses for each pipeline stage
- Pipelines orchestrate node execution with error handling and progress tracking
- Services (IndexingService, etc.) become thin wrappers around workflow execution
- Search remains in pmd.search.pipeline (HybridSearchPipeline already follows this pattern)

Example:
    from pmd.workflows import IngestionPipeline, IngestionRequest
    from pmd.workflows.contracts import WorkflowContext

    pipeline = IngestionPipeline(
        source_collection_repo=repo,
        loader=loading_service,
        document_repo=doc_repo,
        fts_repo=fts_repo,
    )

    request = IngestionRequest(collection_name="my-docs", force=False)
    result = await pipeline.execute(request)
"""

from pmd.workflows.contracts import (
    # Re-exports from existing modules
    IndexResult,
    EmbedResult,
    LoadedDocument,
    # Workflow context
    WorkflowContext,
    ProgressCallback,
    # Ingestion contracts
    IngestionRequest,
    ResolvedCollection,
    EnumeratedRefs,
    LoadInput,
    PersistedDocument,
    # Embedding contracts
    EmbedRequest,
    EmbedTarget,
    Chunk,
    DocumentChunks,
    EmbeddingVector,
    EmbeddedChunks,
    StoreResult,
    # Protocols
    ChunkerProtocol,
)
from pmd.workflows.pipelines.ingestion import IngestionPipeline
from pmd.workflows.pipelines.embedding import EmbeddingPipeline

__all__ = [
    # Re-exports from existing modules
    "IndexResult",
    "EmbedResult",
    "LoadedDocument",
    # Workflow context
    "WorkflowContext",
    "ProgressCallback",
    # Ingestion contracts
    "IngestionRequest",
    "ResolvedCollection",
    "EnumeratedRefs",
    "LoadInput",
    "PersistedDocument",
    # Embedding contracts
    "EmbedRequest",
    "EmbedTarget",
    "Chunk",
    "DocumentChunks",
    "EmbeddingVector",
    "EmbeddedChunks",
    "StoreResult",
    # Protocols
    "ChunkerProtocol",
    # Pipelines
    "IngestionPipeline",
    "EmbeddingPipeline",
]
