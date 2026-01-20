"""Ingest pipeline for indexing documents.

Provides the IngestPipeline class for ingesting documents from various
sources into the idx system, persisting them to the database and
updating derived indexes (FTS, vector).

Uses LlamaIndex's IngestionPipeline for document transformations while
maintaining custom persistence logic for statistics tracking and FTS indexing.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import Document as LlamaDocument
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode
from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.pipelines.schemas import IngestDirectoryConfig, IngestObsidianConfig, IngestResult
from idx.source.directory import DirectorySource, SourceDocument
from idx.source.obsidian import ObsidianDocument, ObsidianVaultSource
from idx.store.database import get_session
from idx.store.fts import FTSManager, create_fts_table
from idx.store.models import Document
from idx.store.repositories import DatasetRepository, DocumentRepository
from idx.store.service import normalize_dataset_name
from idx.transform.llama import TextNormalizerTransform

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
    (normalization, etc.) while maintaining custom persistence logic
    for statistics tracking and FTS indexing.

    The workflow:
    1. Enumerate documents from source
    2. Convert to LlamaIndex Documents
    3. Run through LlamaIndex transformation pipeline
    4. Persist to database with change detection
    5. Update FTS index

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
            transformations: Optional list of LlamaIndex TransformComponents.
                If not provided, uses default TextNormalizerTransform.
            session_factory: Optional session factory for testing.
                If not provided, uses get_session() from idx.store.database.
        """
        if transformations is None:
            transformations = [TextNormalizerTransform()]

        self._llama_pipeline = IngestionPipeline(
            transformations=transformations,
        )
        self._session_factory = session_factory

    def _get_session_context(self):
        """Get a session context manager.

        Returns session_factory if provided, otherwise get_session().
        """
        if self._session_factory is not None:
            return self._session_factory()
        return get_session()

    def ingest_directory(self, config: IngestDirectoryConfig) -> IngestResult:
        """Ingest documents from a local directory.

        Creates or retrieves the dataset, enumerates matching files,
        transforms them via LlamaIndex pipeline, and processes each
        document for persistence and indexing.

        Args:
            config: Directory ingestion configuration.

        Returns:
            IngestResult with statistics about the operation.

        Raises:
            FileNotFoundError: If the directory does not exist.
            NotADirectoryError: If the path is not a directory.
        """
        started_at = datetime.now(tz=timezone.utc)
        normalized_name = normalize_dataset_name(config.dataset_name)

        logger.info(
            f"Starting directory ingestion: {config.directory} -> {normalized_name}"
        )

        # Create source
        source = DirectorySource(
            config.directory,
            patterns=config.patterns,
            encoding=config.encoding,
        )

        # Track results
        result = IngestResult(
            dataset_id=0,  # Will be set after dataset creation
            dataset_name=normalized_name,
            started_at=started_at,
        )

        with self._get_session_context() as session:
            # Ensure FTS table exists
            engine = session.get_bind()
            if engine is not None:
                create_fts_table(engine)  # type: ignore

            # Create or get dataset
            dataset_id = self._ensure_dataset(
                session,
                config.dataset_name,
                source_type="directory",
                source_path=str(source.directory),
            )
            result.dataset_id = dataset_id

            # Initialize managers
            doc_repo = DocumentRepository(session)
            fts = FTSManager(session)

            # Get existing document paths for change detection
            existing_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=False)
            existing_docs = {
                path: doc_repo.get_by_path(dataset_id, path)
                for path in existing_paths
            }

            # Convert source documents to LlamaIndex documents
            source_docs = list(source.enumerate())
            llama_docs = [source_doc_to_llama_doc(doc) for doc in source_docs]

            # Run through LlamaIndex transformation pipeline
            logger.debug(f"Running {len(llama_docs)} documents through transformation pipeline")
            transformed_nodes = self._llama_pipeline.run(documents=llama_docs)

            # Build a map from doc_id (relative_path) to transformed node
            node_map: dict[str, BaseNode] = {
                node.node_id: node for node in transformed_nodes
            }

            # Process each document
            for source_doc in source_docs:
                path = source_doc.relative_path
                node = node_map.get(path)

                if node is None:
                    logger.warning(f"No transformed node found for {path}")
                    result.documents_failed += 1
                    result.errors.append(f"{path}: transformation failed")
                    continue

                try:
                    self._process_node(
                        session=session,
                        doc_repo=doc_repo,
                        fts=fts,
                        dataset_id=dataset_id,
                        node=node,
                        source_doc=source_doc,
                        existing_docs=existing_docs,
                        force=config.force,
                        result=result,
                    )
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    result.documents_failed += 1
                    result.errors.append(f"{path}: {e}")

            # Commit any pending changes
            session.flush()

        result.completed_at = datetime.now(tz=timezone.utc)

        logger.info(
            f"Ingestion complete: "
            f"created={result.documents_created}, "
            f"updated={result.documents_updated}, "
            f"skipped={result.documents_skipped}, "
            f"failed={result.documents_failed}"
        )

        return result

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

    def _process_node(
        self,
        session: Session,
        doc_repo: DocumentRepository,
        fts: FTSManager,
        dataset_id: int,
        node: BaseNode,
        source_doc: SourceDocument,
        existing_docs: dict[str, Document | None],
        force: bool,
        result: IngestResult,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Process a transformed node for persistence.

        Args:
            session: SQLAlchemy session.
            doc_repo: Document repository.
            fts: FTS manager.
            dataset_id: Dataset ID.
            node: Transformed LlamaIndex node.
            source_doc: Original source document (for etag, last_modified).
            existing_docs: Map of path -> existing Document (or None).
            force: If True, always update even if unchanged.
            result: Result object to update.
            metadata: Optional metadata to store with the document.
        """
        path = source_doc.relative_path
        normalized_body = node.get_content()

        # Compute hash on normalized content
        content_hash = compute_content_hash(normalized_body)

        # Merge node metadata with provided metadata
        combined_metadata = dict(node.metadata) if node.metadata else {}
        if metadata:
            combined_metadata.update(metadata)
        metadata_json = json.dumps(combined_metadata) if combined_metadata else None

        # Check if document exists
        existing = existing_docs.get(path)

        if existing is not None:
            # Document exists - check if changed
            if not force and existing.content_hash == content_hash:
                # Unchanged - skip (but ensure active if was inactive)
                if not existing.active:
                    existing.active = True
                    existing.metadata_json = metadata_json
                    session.flush()
                    # Re-index in FTS
                    fts.upsert(existing.id, path, normalized_body)
                    result.documents_updated += 1
                else:
                    result.documents_skipped += 1
                return

            # Changed or force - update
            existing.content_hash = content_hash
            existing.body = normalized_body
            existing.etag = source_doc.etag
            existing.last_modified = source_doc.last_modified
            existing.active = True
            existing.metadata_json = metadata_json
            session.flush()

            # Update FTS
            fts.upsert(existing.id, path, normalized_body)

            logger.debug(f"Updated document: {path}")
            result.documents_updated += 1
        else:
            # New document - create
            doc = doc_repo.create(
                dataset_id=dataset_id,
                path=path,
                content_hash=content_hash,
                body=normalized_body,
                etag=source_doc.etag,
                last_modified=source_doc.last_modified,
                metadata_json=metadata_json,
            )
            session.flush()

            # Index in FTS
            fts.upsert(doc.id, path, normalized_body)

            logger.debug(f"Created document: {path}")
            result.documents_created += 1

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
            f"Starting Obsidian vault ingestion: {config.vault_path} -> {normalized_name}"
        )

        # Create source
        source = ObsidianVaultSource(config.vault_path)

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
        # Ensure FTS table exists
        engine = session.get_bind()
        if engine is not None:
            create_fts_table(engine)  # type: ignore

        # Create or get dataset
        dataset_id = self._ensure_dataset(
            session,
            config.dataset_name,
            source_type="obsidian",
            source_path=str(source.vault_path),
        )
        result.dataset_id = dataset_id

        # Initialize managers
        doc_repo = DocumentRepository(session)
        fts = FTSManager(session)

        # Get existing document paths for change detection
        existing_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=False)
        existing_docs = {
            path: doc_repo.get_by_path(dataset_id, path)
            for path in existing_paths
        }

        # Collect and convert Obsidian documents
        obsidian_docs = list(source.enumerate())
        llama_docs = [
            self._obsidian_doc_to_llama_doc(doc) for doc in obsidian_docs
        ]

        # Run through LlamaIndex transformation pipeline
        logger.debug(f"Running {len(llama_docs)} documents through transformation pipeline")
        transformed_nodes = self._llama_pipeline.run(documents=llama_docs)

        # Build a map from doc_id (relative_path) to transformed node
        node_map: dict[str, BaseNode] = {
            node.node_id: node for node in transformed_nodes
        }

        # Process each document
        for obsidian_doc in obsidian_docs:
            path = obsidian_doc.relative_path
            node = node_map.get(path)

            if node is None:
                logger.warning(f"No transformed node found for {path}")
                result.documents_failed += 1
                result.errors.append(f"{path}: transformation failed")
                continue

            try:
                # Build metadata from frontmatter
                metadata = self._build_obsidian_metadata(obsidian_doc)

                self._process_node(
                    session=session,
                    doc_repo=doc_repo,
                    fts=fts,
                    dataset_id=dataset_id,
                    node=node,
                    source_doc=obsidian_doc,
                    existing_docs=existing_docs,
                    force=config.force,
                    result=result,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                result.documents_failed += 1
                result.errors.append(f"{path}: {e}")

        # Commit any pending changes
        session.flush()

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

    def _build_obsidian_metadata(self, doc: ObsidianDocument) -> dict[str, Any]:
        """Build metadata dictionary from an Obsidian document.

        Args:
            doc: Obsidian document with parsed frontmatter.

        Returns:
            Metadata dictionary with tags, aliases, and frontmatter.
        """
        metadata: dict[str, Any] = {}

        if doc.tags:
            metadata["tags"] = doc.tags

        if doc.aliases:
            metadata["aliases"] = doc.aliases

        if doc.frontmatter:
            # Store raw frontmatter for future use
            metadata["frontmatter"] = doc.frontmatter

        return metadata if metadata else {}
