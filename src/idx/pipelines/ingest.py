"""Ingest pipeline for indexing documents.

Provides the IngestPipeline class for ingesting documents from various
sources into the idx system, persisting them to the database and
updating derived indexes (FTS, vector).

Uses LlamaIndex's IngestionPipeline for document transformations with
PersistenceTransform handling database persistence and FTS indexing
as the final pipeline step. Uses ambient session via contextvars.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from llama_index.core import Document as LlamaDocument
from llama_index.core.ingestion import IngestionPipeline
from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.pipelines.schemas import IngestDirectoryConfig, IngestObsidianConfig, IngestResult
from idx.source.directory import DirectorySource, SourceDocument
from idx.source.obsidian import ObsidianDocument, ObsidianVaultSource
from idx.store.database import get_session
from idx.store.fts import create_fts_table
from idx.store.repositories import DatasetRepository
from idx.store.service import DatasetService, normalize_dataset_name
from idx.store.session_context import use_session
from idx.transform.llama import PersistenceTransform, TextNormalizerTransform

__all__ = [
    "IngestPipeline",
]

logger = get_logger(__name__)



# TODO: make `config` generic to support multiple ingestion types
def _get_source_instance(config):
    match config:
        case IngestDirectoryConfig():
            return DirectorySource(
                config.source_path,
                patterns=config.patterns,
                encoding=config.encoding,
            )
        case IngestObsidianConfig():
            return ObsidianVaultSource(config.source_path)
        case _:
            raise TypeError(f"Unsupported config type: {type(config)}")


class IngestPipeline:
    """Pipeline for ingesting documents from sources.

    Uses LlamaIndex's IngestionPipeline for document transformations
    with PersistenceTransform at the end handling database persistence
    and FTS indexing within the pipeline.

    The workflow:
    1. Enumerate documents from source
    2. Convert to LlamaIndex Documents
    3. Run through LlamaIndex transformation pipeline:
       - TextNormalizerTransform 
       - PersistenceTransform (creates/updates documents, indexes FTS)
    4. Return results with statistics

    Note: Stale document handling has been moved to idx.store.cleanup.
    Use cleanup_stale_documents() for maintenance operations.

    Example:
        config = IngestDirectoryConfig(
            directory=Path("/path/to/docs"),
            dataset_name="my-docs",
            patterns=["**/*.md"],
        )
        pipeline = IngestPipeline()
        result = pipeline.ingest_directory(config)
        print(f"Processed {result.total_processed} documents")
    """

    def ingest(self, config: IngestDirectoryConfig | IngestObsidianConfig) -> IngestResult:
        """Ingest documents from a local directory.

        Creates or retrieves the dataset, enumerates matching files,
        and runs them through the LlamaIndex transformation pipeline
        with PersistenceTransform handling persistence and FTS indexing.

        Args:
            config: Ingestion configuration, either directory or Obsidian.

        Returns:
            IngestResult with statistics about the operation.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """

        started_at = datetime.now(tz=timezone.utc)
        normalized_name = normalize_dataset_name(config.dataset_name)

        logger.info(
            f"Starting directory ingestion: {config.source_path} -> {normalized_name}"
        )

        source = _get_source_instance(config)

        # Track results
        result = IngestResult(
            dataset_id=0,  # Will be set after dataset creation
            dataset_name=normalized_name,
            started_at=started_at,
        )

        with get_session() as session:
            # Set ambient session for transforms to use
            with use_session(session):
                # Ensure FTS table exists
                engine = session.get_bind()
                if engine is not None:
                    create_fts_table(engine)  # type: ignore

                # Create or get dataset
                dataset_id = DatasetService.create_or_update(
                    session,
                    config.dataset_name,
                    source_type=source.type_name,
                    source_path=str(source.path),
                )
                result.dataset_id = dataset_id

                # Convert source documents to LlamaIndex documents
                source_docs = list(source.enumerate())
                llama_docs = [source.to_llama_doc(doc) for doc in source_docs]  # type: ignore

                # Create persistence transform (uses ambient session)
                persist = PersistenceTransform(
                    dataset_id=dataset_id,
                    force=config.force,
                )

                # Build pipeline with transforms + persistence
                pipeline = IngestionPipeline(
                    transformations=[TextNormalizerTransform(), persist],
                )

                # Run pipeline - persistence happens inside using ambient session
                logger.debug(f"Running {len(llama_docs)} documents through pipeline")
                pipeline.run(documents=llama_docs)

                # Copy stats from persistence transform
                result.documents_created = persist.stats.created
                result.documents_updated = persist.stats.updated
                result.documents_skipped = persist.stats.skipped
                result.documents_failed = persist.stats.failed
                result.errors = persist.stats.errors

        result.completed_at = datetime.now(tz=timezone.utc)

        logger.info(
            f"Ingestion complete: "
            f"created={result.documents_created}, "
            f"updated={result.documents_updated}, "
            f"skipped={result.documents_skipped}, "
            f"failed={result.documents_failed}"
        )

        return result




