"""Ingest pipeline for indexing documents.

Provides the IngestPipeline class for ingesting documents from various
sources into the idx system, persisting them to the database and
updating derived indexes (FTS, vector).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.pipelines.schemas import IngestDirectoryConfig, IngestObsidianConfig, IngestResult
from idx.source.directory import DirectorySource, SourceDocument
from idx.source.obsidian import ObsidianDocument, ObsidianVaultSource
from idx.store.database import get_session
from idx.store.cleanup import IndexCleanup
from idx.store.fts import FTSManager, create_fts_table
from idx.store.models import Document
from idx.store.repositories import DatasetRepository, DocumentRepository
from idx.store.service import (
    DatasetExistsError,
    DatasetService,
    normalize_dataset_name,
)
from idx.store.schemas import DatasetCreate, DocumentCreate, DocumentUpdate
from idx.transform.normalize import TextNormalizer

__all__ = [
    "IngestPipeline",
    "compute_content_hash",
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


class IngestPipeline:
    """Pipeline for ingesting documents from sources.

    Handles the full ingestion workflow:
    1. Validate dataset name and create/get dataset
    2. Enumerate documents from source
    3. Normalize text content
    4. Compute content hashes
    5. Persist documents (create or update)
    6. Update FTS index

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

    def __init__(self, session: Session | None = None) -> None:
        """Initialize the pipeline.

        Args:
            session: Optional SQLAlchemy session. If not provided,
                    creates a new session for each operation.
        """
        self._external_session = session
        self._normalizer = TextNormalizer()

    def ingest_directory(self, config: IngestDirectoryConfig) -> IngestResult:
        """Ingest documents from a local directory.

        Creates or retrieves the dataset, enumerates matching files,
        and processes each document for persistence and indexing.

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

        if self._external_session is not None:
            # Use provided session
            self._ingest_with_session(
                self._external_session,
                source,
                config,
                result,
            )
        else:
            # Create new session
            with get_session() as session:
                self._ingest_with_session(session, source, config, result)

        result.completed_at = datetime.now(tz=timezone.utc)

        logger.info(
            f"Ingestion complete: "
            f"created={result.documents_created}, "
            f"updated={result.documents_updated}, "
            f"skipped={result.documents_skipped}, "
            f"stale={result.documents_stale}, "
            f"failed={result.documents_failed}"
        )

        return result

    def _ingest_with_session(
        self,
        session: Session,
        source: DirectorySource,
        config: IngestDirectoryConfig,
        result: IngestResult,
    ) -> None:
        """Run ingestion with the given session.

        Args:
            session: SQLAlchemy session.
            source: Document source.
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
            source_type="directory",
            source_path=str(source.directory),
        )
        result.dataset_id = dataset_id

        # Initialize managers
        doc_repo = DocumentRepository(session)
        fts = FTSManager(session)
        cleanup = IndexCleanup(session)

        # Get existing document paths for change detection
        existing_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=False)
        existing_docs = {
            path: doc_repo.get_by_path(dataset_id, path)
            for path in existing_paths
        }

        # Track which documents we've seen in this enumeration
        seen_paths: set[str] = set()

        # Process documents
        for source_doc in source.enumerate():
            try:
                seen_paths.add(source_doc.relative_path)
                self._process_document(
                    session=session,
                    doc_repo=doc_repo,
                    fts=fts,
                    dataset_id=dataset_id,
                    source_doc=source_doc,
                    existing_docs=existing_docs,
                    force=config.force,
                    result=result,
                )
            except Exception as e:
                logger.error(f"Failed to process {source_doc.relative_path}: {e}")
                result.documents_failed += 1
                result.errors.append(f"{source_doc.relative_path}: {e}")

        # Detect and handle stale documents (in DB but not in source)
        active_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=True)
        stale_paths = active_paths - seen_paths

        if stale_paths:
            logger.info(f"Found {len(stale_paths)} stale documents")
            stale_count = self._handle_stale_documents(
                session=session,
                doc_repo=doc_repo,
                cleanup=cleanup,
                dataset_id=dataset_id,
                stale_paths=stale_paths,
            )
            result.documents_stale = stale_count

        # Commit any pending changes
        session.flush()

    def _handle_stale_documents(
        self,
        session: Session,
        doc_repo: DocumentRepository,
        cleanup: IndexCleanup,
        dataset_id: int,
        stale_paths: set[str],
    ) -> int:
        """Handle stale documents by soft-deleting and cleaning up indexes.

        Args:
            session: SQLAlchemy session.
            doc_repo: Document repository.
            cleanup: Index cleanup manager.
            dataset_id: Dataset ID.
            stale_paths: Set of paths that are stale.

        Returns:
            Number of documents marked as stale.
        """
        # Get the document IDs for FTS cleanup
        stale_doc_ids: list[int] = []
        for path in stale_paths:
            doc = doc_repo.get_by_path(dataset_id, path)
            if doc is not None and doc.active:
                stale_doc_ids.append(doc.id)
                logger.debug(f"Marking stale: {path}")

        # Soft-delete the documents
        deleted_count = doc_repo.soft_delete_by_paths(dataset_id, stale_paths)

        # Clean up FTS entries for stale documents
        if stale_doc_ids:
            cleanup.cleanup_fts_for_documents(stale_doc_ids)

        logger.info(f"Soft-deleted {deleted_count} stale documents")
        return deleted_count

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

    def _process_document(
        self,
        session: Session,
        doc_repo: DocumentRepository,
        fts: FTSManager,
        dataset_id: int,
        source_doc: SourceDocument,
        existing_docs: dict[str, Document | None],
        force: bool,
        result: IngestResult,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Process a single document.

        Args:
            session: SQLAlchemy session.
            doc_repo: Document repository.
            fts: FTS manager.
            dataset_id: Dataset ID.
            source_doc: Source document to process.
            existing_docs: Map of path -> existing Document (or None).
            force: If True, always update even if unchanged.
            result: Result object to update.
            metadata: Optional metadata to store with the document.
        """
        path = source_doc.relative_path

        # Normalize content
        normalized_body = self._normalizer.normalize(source_doc.content)

        # Compute hash
        content_hash = compute_content_hash(normalized_body)

        # Serialize metadata
        metadata_json = json.dumps(metadata) if metadata else None

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
        extracts frontmatter metadata (tags, aliases), and processes
        each document for persistence and indexing.

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

        if self._external_session is not None:
            # Use provided session
            self._ingest_obsidian_with_session(
                self._external_session,
                source,
                config,
                result,
            )
        else:
            # Create new session
            with get_session() as session:
                self._ingest_obsidian_with_session(session, source, config, result)

        result.completed_at = datetime.now(tz=timezone.utc)

        logger.info(
            f"Obsidian ingestion complete: "
            f"created={result.documents_created}, "
            f"updated={result.documents_updated}, "
            f"skipped={result.documents_skipped}, "
            f"stale={result.documents_stale}, "
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
        cleanup = IndexCleanup(session)

        # Get existing document paths for change detection
        existing_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=False)
        existing_docs = {
            path: doc_repo.get_by_path(dataset_id, path)
            for path in existing_paths
        }

        # Track which documents we've seen in this enumeration
        seen_paths: set[str] = set()

        # Process documents
        for obsidian_doc in source.enumerate():
            try:
                seen_paths.add(obsidian_doc.relative_path)

                # Build metadata from frontmatter
                metadata = self._build_obsidian_metadata(obsidian_doc)

                self._process_document(
                    session=session,
                    doc_repo=doc_repo,
                    fts=fts,
                    dataset_id=dataset_id,
                    source_doc=obsidian_doc,
                    existing_docs=existing_docs,
                    force=config.force,
                    result=result,
                    metadata=metadata,
                )
            except Exception as e:
                logger.error(f"Failed to process {obsidian_doc.relative_path}: {e}")
                result.documents_failed += 1
                result.errors.append(f"{obsidian_doc.relative_path}: {e}")

        # Detect and handle stale documents (in DB but not in source)
        active_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=True)
        stale_paths = active_paths - seen_paths

        if stale_paths:
            logger.info(f"Found {len(stale_paths)} stale documents")
            stale_count = self._handle_stale_documents(
                session=session,
                doc_repo=doc_repo,
                cleanup=cleanup,
                dataset_id=dataset_id,
                stale_paths=stale_paths,
            )
            result.documents_stale = stale_count

        # Commit any pending changes
        session.flush()

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
