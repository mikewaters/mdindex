"""idx.store.cleanup - Cleanup hooks for derived indexes.

Provides cleanup functionality for FTS and vector indexes when
documents are soft-deleted or removed.

Example usage:
    from idx.store.cleanup import IndexCleanup

    cleanup = IndexCleanup(session)
    cleanup.remove_fts_for_inactive(dataset_id)
"""

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.ingest.directory import DirectorySource
from idx.store.fts import FTSManager
from idx.store.repositories import DocumentRepository

__all__ = [
    "IndexCleanup",
    "cleanup_fts_for_document",
    "cleanup_fts_for_inactive_documents",
    "cleanup_stale_documents",
]

logger = get_logger(__name__)


class IndexCleanup:
    """Manages cleanup of derived indexes (FTS, vector) for documents.

    Provides methods to clean up FTS entries and vector nodes when
    documents are soft-deleted or removed.
    """

    def __init__(self, session: Session) -> None:
        """Initialize the cleanup manager.

        Args:
            session: SQLAlchemy session for database operations.
        """
        self._session = session
        self._fts = FTSManager(session)

    def cleanup_fts_for_document(self, doc_id: int) -> None:
        """Remove FTS entry for a single document.

        Args:
            doc_id: Document ID to remove from FTS.
        """
        self._fts.delete(doc_id)
        logger.debug(f"Cleaned up FTS for document {doc_id}")

    def cleanup_fts_for_documents(self, doc_ids: list[int]) -> int:
        """Remove FTS entries for multiple documents.

        Args:
            doc_ids: List of document IDs to remove from FTS.

        Returns:
            Number of FTS entries removed.
        """
        if not doc_ids:
            return 0
        count = self._fts.delete_many(doc_ids)
        logger.info(f"Cleaned up FTS for {count} documents")
        return count

    def cleanup_fts_for_inactive(self, dataset_id: int | None = None) -> int:
        """Remove FTS entries for all inactive documents.

        Args:
            dataset_id: Optional dataset ID to limit cleanup scope.

        Returns:
            Number of FTS entries removed.
        """
        # Find inactive documents that still have FTS entries
        if dataset_id is not None:
            result = self._session.execute(
                text("""
                    SELECT d.id
                    FROM documents d
                    WHERE d.active = 0
                    AND d.dataset_id = :dataset_id
                    AND EXISTS (
                        SELECT 1 FROM documents_fts f WHERE f.rowid = d.id
                    )
                """),
                {"dataset_id": dataset_id},
            )
        else:
            result = self._session.execute(
                text("""
                    SELECT d.id
                    FROM documents d
                    WHERE d.active = 0
                    AND EXISTS (
                        SELECT 1 FROM documents_fts f WHERE f.rowid = d.id
                    )
                """)
            )

        doc_ids = [row[0] for row in result]
        if doc_ids:
            return self.cleanup_fts_for_documents(doc_ids)
        return 0

    def cleanup_fts_for_dataset(self, dataset_id: int) -> int:
        """Remove all FTS entries for a dataset.

        Used when deleting an entire dataset.

        Args:
            dataset_id: Dataset ID to clean up.

        Returns:
            Number of FTS entries removed.
        """
        result = self._session.execute(
            text("""
                SELECT d.id
                FROM documents d
                WHERE d.dataset_id = :dataset_id
            """),
            {"dataset_id": dataset_id},
        )
        doc_ids = [row[0] for row in result]
        if doc_ids:
            return self.cleanup_fts_for_documents(doc_ids)
        return 0

    def delete_stale_documents(
        self,
        dataset_id: int,
        stale_paths: set[str],
    ) -> int:
        """Hard-delete stale documents and their FTS entries.

        Removes documents from both the FTS index and the documents table.
        This is a destructive operation that cannot be undone.

        Args:
            dataset_id: Dataset ID containing the stale documents.
            stale_paths: Set of relative paths to delete.

        Returns:
            Number of documents deleted.
        """
        if not stale_paths:
            return 0

        # Get doc IDs for the stale paths
        paths_list = list(stale_paths)
        placeholders = ",".join([f":p{i}" for i in range(len(paths_list))])
        params = {f"p{i}": path for i, path in enumerate(paths_list)}
        params["dataset_id"] = dataset_id

        result = self._session.execute(
            text(f"""
                SELECT id
                FROM documents
                WHERE dataset_id = :dataset_id
                AND path IN ({placeholders})
            """),
            params,
        )
        doc_ids = [row[0] for row in result]

        if not doc_ids:
            return 0

        # Clean up FTS entries first
        self.cleanup_fts_for_documents(doc_ids)

        # Hard-delete the documents
        doc_repo = DocumentRepository(self._session)
        deleted_count = doc_repo.hard_delete_by_paths(dataset_id, stale_paths)

        logger.info(
            f"Deleted {deleted_count} stale documents from dataset {dataset_id}"
        )
        return deleted_count

    # Vector cleanup methods (placeholder for future implementation)

    def cleanup_vectors_for_document(self, doc_id: int, content_hash: str) -> int:
        """Remove vector nodes for a document.

        Placeholder for vector store cleanup. Will be implemented
        when vector store is added.

        Args:
            doc_id: Document ID.
            content_hash: Content hash to identify vector nodes.

        Returns:
            Number of vector nodes removed.
        """
        # TODO: Implement vector store cleanup
        logger.debug(f"Vector cleanup placeholder for doc {doc_id}")
        return 0

    def cleanup_vectors_for_inactive(self, dataset_id: int | None = None) -> int:
        """Remove vector nodes for all inactive documents.

        Placeholder for vector store cleanup.

        Args:
            dataset_id: Optional dataset ID to limit cleanup scope.

        Returns:
            Number of vector nodes removed.
        """
        # TODO: Implement vector store cleanup
        logger.debug("Vector cleanup placeholder for inactive documents")
        return 0


# Convenience functions

def cleanup_fts_for_document(session: Session, doc_id: int) -> None:
    """Remove FTS entry for a single document.

    Args:
        session: SQLAlchemy session.
        doc_id: Document ID to remove from FTS.
    """
    cleanup = IndexCleanup(session)
    cleanup.cleanup_fts_for_document(doc_id)


def cleanup_fts_for_inactive_documents(
    session: Session,
    dataset_id: int | None = None,
) -> int:
    """Remove FTS entries for all inactive documents.

    Args:
        session: SQLAlchemy session.
        dataset_id: Optional dataset ID to limit cleanup scope.

    Returns:
        Number of FTS entries removed.
    """
    cleanup = IndexCleanup(session)
    return cleanup.cleanup_fts_for_inactive(dataset_id)


def cleanup_stale_documents(
    session: Session,
    dataset_id: int,
    source_path: Path,
    patterns: list[str] | None = None,
    dry_run: bool = False,
) -> int:
    """Find and remove documents that no longer exist in the source.

    Compares the current source directory contents against indexed documents
    and removes any documents that are no longer present in the source.

    Args:
        session: SQLAlchemy session for database operations.
        dataset_id: Dataset ID to check for stale documents.
        source_path: Path to the source directory.
        patterns: Glob patterns for file matching. Defaults to ["**/*.md"].
        dry_run: If True, only log stale paths without deleting.

    Returns:
        Number of stale documents found (dry_run) or deleted.
    """
    # Enumerate source paths
    source = DirectorySource(source_path, patterns=patterns)
    source_paths: set[str] = set()
    for doc in source.enumerate():
        source_paths.add(doc.relative_path)

    # Get indexed paths from database
    doc_repo = DocumentRepository(session)
    indexed_paths = doc_repo.list_paths_by_dataset(dataset_id, active_only=False)

    # Find stale paths (in DB but not in source)
    stale_paths = indexed_paths - source_paths

    if not stale_paths:
        logger.debug(f"No stale documents found for dataset {dataset_id}")
        return 0

    if dry_run:
        logger.info(
            f"[DRY RUN] Found {len(stale_paths)} stale documents for dataset "
            f"{dataset_id}: {sorted(stale_paths)}"
        )
        return len(stale_paths)

    # Delete stale documents
    cleanup = IndexCleanup(session)
    return cleanup.delete_stale_documents(dataset_id, stale_paths)
