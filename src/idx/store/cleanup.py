"""idx.store.cleanup - Cleanup hooks for derived indexes.

Provides cleanup functionality for FTS and vector indexes when
documents are soft-deleted or removed.

Example usage:
    from idx.store.cleanup import IndexCleanup

    cleanup = IndexCleanup(session)
    cleanup.remove_fts_for_inactive(dataset_id)
"""

from sqlalchemy import text
from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.store.fts import FTSManager

__all__ = [
    "IndexCleanup",
    "cleanup_fts_for_document",
    "cleanup_fts_for_inactive_documents",
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
