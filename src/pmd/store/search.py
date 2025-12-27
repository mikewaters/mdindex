"""Full-text search implementation for PMD."""

from abc import ABC, abstractmethod

from ..core.types import SearchResult, SearchSource
from .database import Database


class SearchRepository(ABC):
    """Abstract interface for search operations."""

    @abstractmethod
    def search_fts(
        self,
        query: str,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform BM25 full-text search.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            collection_id: Optional collection ID to limit search scope.
            min_score: Minimum score threshold for results.

        Returns:
            List of SearchResult objects sorted by relevance.
        """
        pass


class FTS5SearchRepository(SearchRepository):
    """FTS5-based full-text search implementation."""

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def search_fts(
        self,
        query: str,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform BM25 full-text search using FTS5.

        Args:
            query: Search query string (supports FTS5 syntax).
            limit: Maximum number of results to return.
            collection_id: Optional collection ID to limit search scope.
            min_score: Minimum score threshold for results.

        Returns:
            List of SearchResult objects sorted by relevance (highest first).
        """
        # Escape and prepare query for FTS5
        fts_query = self._prepare_fts_query(query)

        if collection_id is not None:
            sql = """
                SELECT
                    d.id,
                    d.collection_id,
                    d.path,
                    d.path as display_path,
                    d.title,
                    d.hash,
                    d.modified_at,
                    c.doc as body,
                    fts.rank as fts_rank
                FROM documents_fts fts
                JOIN documents d ON fts.rowid = d.id
                JOIN content c ON d.hash = c.hash
                WHERE documents_fts MATCH ? AND d.collection_id = ? AND d.active = 1
                ORDER BY fts.rank ASC
                LIMIT ?
            """
            cursor = self.db.execute(sql, (fts_query, collection_id, limit))
        else:
            sql = """
                SELECT
                    d.id,
                    d.collection_id,
                    d.path,
                    d.path as display_path,
                    d.title,
                    d.hash,
                    d.modified_at,
                    c.doc as body,
                    fts.rank as fts_rank
                FROM documents_fts fts
                JOIN documents d ON fts.rowid = d.id
                JOIN content c ON d.hash = c.hash
                WHERE documents_fts MATCH ? AND d.active = 1
                ORDER BY fts.rank ASC
                LIMIT ?
            """
            cursor = self.db.execute(sql, (fts_query, limit))

        results = []
        max_rank = None

        # First pass: collect results and find max rank for normalization
        rows = cursor.fetchall()
        if rows:
            max_rank = min(abs(row["fts_rank"]) for row in rows)
            if max_rank == 0:
                max_rank = 1

            # Second pass: normalize scores
            for row in rows:
                # FTS5 rank is negative BM25 score
                raw_score = abs(row["fts_rank"]) if row["fts_rank"] else 0.0
                normalized_score = raw_score / max_rank if max_rank else 0.0

                if normalized_score >= min_score:
                    results.append(
                        SearchResult(
                            filepath=row["path"],
                            display_path=row["display_path"],
                            title=row["title"],
                            context=None,
                            hash=row["hash"],
                            collection_id=row["collection_id"],
                            modified_at=row["modified_at"],
                            body_length=len(row["body"]) if row["body"] else 0,
                            body=None,  # Don't include body in search results by default
                            score=normalized_score,
                            source=SearchSource.FTS,
                        )
                    )

        return results

    def index_document(self, doc_id: int, path: str, body: str) -> None:
        """Index a document in FTS5.

        Args:
            doc_id: Document ID from documents table.
            path: Document path.
            body: Document content to index.
        """
        with self.db.transaction() as cursor:
            # Delete existing index entry if present
            cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))

            # Insert into FTS5 index
            cursor.execute(
                "INSERT INTO documents_fts(rowid, path, body) VALUES (?, ?, ?)",
                (doc_id, path, body),
            )

    def reindex_collection(self, collection_id: int) -> int:
        """Reindex all documents in a collection.

        Args:
            collection_id: ID of collection to reindex.

        Returns:
            Number of documents indexed.
        """
        with self.db.transaction() as cursor:
            # Get all active documents in collection
            cursor.execute(
                """
                SELECT d.id, d.path, c.doc FROM documents d
                JOIN content c ON d.hash = c.hash
                WHERE d.collection_id = ? AND d.active = 1
                """,
                (collection_id,),
            )
            documents = cursor.fetchall()

            # Clear existing index for this collection
            cursor.execute(
                """
                DELETE FROM documents_fts WHERE rowid IN (
                    SELECT id FROM documents WHERE collection_id = ?
                )
                """,
                (collection_id,),
            )

            # Reindex all documents
            for doc in documents:
                cursor.execute(
                    "INSERT INTO documents_fts(rowid, path, body) VALUES (?, ?, ?)",
                    (doc["id"], doc["path"], doc["doc"]),
                )

        return len(documents)

    def remove_from_index(self, doc_id: int) -> None:
        """Remove a document from FTS5 index.

        Args:
            doc_id: Document ID to remove.
        """
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))

    def clear_index(self) -> None:
        """Clear all FTS5 index entries."""
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM documents_fts")

    @staticmethod
    def _prepare_fts_query(query: str) -> str:
        """Prepare query string for FTS5.

        Handles basic FTS5 syntax and escaping.

        Args:
            query: Raw query string.

        Returns:
            FTS5-compatible query string.
        """
        # For now, return the query as-is.
        # More sophisticated query preparation (handling operators, etc.)
        # can be added as needed.
        return query
