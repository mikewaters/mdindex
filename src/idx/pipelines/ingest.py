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
    "compute_content_hash",
    "source_doc_to_llama_doc",
]

logger = get_logger(__name__)


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content.

    Args:
        content: The text content to hash.

    Returns:
        Hexadecimal SHA256 hash string.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def source_doc_to_llama_doc(
    source_doc: SourceDocument,
    *,
    extra_metadata: dict[str, Any] | None = None,
) -> LlamaDocument:
    """Convert a SourceDocument to a LlamaIndex Document.

    Args:
        source_doc: The source document to convert.
        extra_metadata: Optional additional metadata to include.

    Returns:
        LlamaIndex Document with text and metadata.
    """
    metadata: dict[str, Any] = {
        "file_path": str(source_doc.path),
        "relative_path": source_doc.relative_path,
    }

    if source_doc.last_modified is not None:
        metadata["last_modified"] = source_doc.last_modified.isoformat()

    if source_doc.etag is not None:
        metadata["etag"] = source_doc.etag

    if extra_metadata:
        metadata.update(extra_metadata)

    return LlamaDocument(
        text=source_doc.content,
        doc_id=source_doc.relative_path,
        metadata=metadata,
    )


class IngestPipeline:
    """Pipeline for ingesting documents from sources.

    Uses LlamaIndex's IngestionPipeline for document transformations
    with PersistenceTransform at the end handling database persistence
    and FTS indexing within the pipeline.

    The workflow:
    1. Enumerate documents from source
    2. Convert to LlamaIndex Documents
    3. Run through LlamaIndex transformation pipeline:
       - TextNormalizerTransform (and any custom transforms)
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

    def __init__(
        self,
        *,
        transformations: list[Any] | None = None,
        session_factory: Any | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            transformations: Optional list of LlamaIndex TransformComponents
                to run before persistence. If not provided, uses default
                TextNormalizerTransform. Do NOT include PersistenceTransform
                here - it is added automatically with the correct session.
            session_factory: Optional session factory for testing.
                If not provided, uses get_session() from idx.store.database.
        """
        if transformations is None:
            transformations = [TextNormalizerTransform()]

        self._transformations = transformations
        self._session_factory = session_factory

    def _get_session_context(self):
        """Get a session context manager.

        Returns session_factory if provided, otherwise get_session().
        """
        if self._session_factory is not None:
            return self._session_factory()
        return get_session()


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
        # TODO: make `config` generic to support multiple ingestion types
        def get_source_instance(config):
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

        started_at = datetime.now(tz=timezone.utc)
        normalized_name = normalize_dataset_name(config.dataset_name)

        logger.info(
            f"Starting directory ingestion: {config.source_path} -> {normalized_name}"
        )

        source = get_source_instance(config)

        # Track results
        result = IngestResult(
            dataset_id=0,  # Will be set after dataset creation
            dataset_name=normalized_name,
            started_at=started_at,
        )

        with self._get_session_context() as session:
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
                    transformations=[*self._transformations, persist],
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

    def ingest_directory(self, config: IngestDirectoryConfig) -> IngestResult:
        """Ingest documents from a local directory.

        Alias for ingest() with IngestDirectoryConfig.

        Args:
            config: Directory ingestion configuration.

        Returns:
            IngestResult with statistics about the operation.
        """
        return self.ingest(config)

    def _ensure_dataset(
        self,
        session: Session,
        name: str,
        source_type: str,
        source_path: str,
    ) -> int:
        """Ensure dataset exists, creating if necessary.

        Args:
            session: SQLAlchemy session.
            name: Dataset name.
            source_type: Type of source (e.g., "directory", "obsidian").
            source_path: Path to the source.

        Returns:
            Dataset ID.
        """
        normalized_name = normalize_dataset_name(name)
        repo = DatasetRepository(session)

        # Check if dataset exists
        dataset = repo.get_by_name(normalized_name)
        if dataset is not None:
            logger.debug(f"Using existing dataset: {normalized_name}")
            return dataset.id

        # Create new dataset
        dataset = repo.create(
            name=normalized_name,
            uri=f"dataset:{normalized_name}",
            source_type=source_type,
            source_path=source_path,
        )
        session.flush()
        logger.info(f"Created dataset: {normalized_name}")
        return dataset.id

    def ingest_obsidian(self, config: IngestObsidianConfig) -> IngestResult:
        """Ingest documents from an Obsidian vault.

        Creates or retrieves the dataset, enumerates markdown files,
        extracts frontmatter metadata (tags, aliases), transforms them
        via LlamaIndex pipeline, and processes each document for
        persistence and indexing.

        Args:
            config: Obsidian ingestion configuration.

        Returns:
            IngestResult with statistics about the operation.

        Raises:
            ValueError: If the path is not a valid Obsidian vault.
        """
        started_at = datetime.now(tz=timezone.utc)
        normalized_name = normalize_dataset_name(config.dataset_name)

        logger.info(
            f"Starting Obsidian vault ingestion: {config.source_path} -> {normalized_name}"
        )

        # Create source
        source = ObsidianVaultSource(config.source_path)

        # Track results
        result = IngestResult(
            dataset_id=0,  # Will be set after dataset creation
            dataset_name=normalized_name,
            started_at=started_at,
        )

        with self._get_session_context() as session:
            self._ingest_obsidian_with_session(session, source, config, result)

        result.completed_at = datetime.now(tz=timezone.utc)

        logger.info(
            f"Obsidian ingestion complete: "
            f"created={result.documents_created}, "
            f"updated={result.documents_updated}, "
            f"skipped={result.documents_skipped}, "
            f"failed={result.documents_failed}"
        )

        return result

    def _ingest_obsidian_with_session(
        self,
        session: Session,
        source: ObsidianVaultSource,
        config: IngestObsidianConfig,
        result: IngestResult,
    ) -> None:
        """Run Obsidian ingestion with the given session.

        Args:
            session: SQLAlchemy session.
            source: Obsidian vault source.
            config: Ingestion configuration.
            result: Result object to update.
        """
        # Set ambient session for transforms to use
        with use_session(session):
            # Ensure FTS table exists
            engine = session.get_bind()
            if engine is not None:
                create_fts_table(engine)  # type: ignore

            # Create or get dataset
            dataset_id = self._ensure_dataset(
                session,
                config.dataset_name,
                source_type=source.type_name,
                source_path=str(source.path),
            )
            result.dataset_id = dataset_id

            # Collect and convert Obsidian documents
            source_docs = list(source.enumerate())
            llama_docs = [source.to_llama_doc(doc) for doc in source_docs]

            # Create persistence transform (uses ambient session)
            persist = PersistenceTransform(
                dataset_id=dataset_id,
                force=config.force,
            )

            # Build pipeline with transforms + persistence
            pipeline = IngestionPipeline(
                transformations=[*self._transformations, persist],
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

    def _obsidian_doc_to_llama_doc(self, doc: ObsidianDocument) -> LlamaDocument:
        """Convert an ObsidianDocument to a LlamaIndex Document.

        Args:
            doc: The Obsidian document to convert.

        Returns:
            LlamaIndex Document with text and metadata.
        """
        metadata: dict[str, Any] = {
            "file_path": str(doc.path),
            "relative_path": doc.relative_path,
        }

        if doc.last_modified is not None:
            metadata["last_modified"] = doc.last_modified.isoformat()

        if doc.etag is not None:
            metadata["etag"] = doc.etag

        if doc.tags:
            metadata["tags"] = doc.tags

        if doc.aliases:
            metadata["aliases"] = doc.aliases

        if doc.frontmatter:
            metadata["frontmatter"] = doc.frontmatter

        # Use body (content without frontmatter) for text
        return LlamaDocument(
            text=doc.body,
            doc_id=doc.relative_path,
            metadata=metadata,
        )
