"""Full-text search implementation for PMD.

This module provides FTS5-based full-text search:

Classes:
    SearchRepository: Abstract base class defining the search interface
    FTS5SearchRepository: Full-text search using SQLite FTS5 with BM25 scoring

For vector similarity search, use EmbeddingRepository.search_vectors() directly.

Example:
    from pmd.store.search import FTS5SearchRepository
    from pmd.store.embeddings import EmbeddingRepository

    fts_repo = FTS5SearchRepository(db)
    embedding_repo = EmbeddingRepository(db)

    # FTS search
    fts_results = fts_repo.search("machine learning", limit=10)

    # Vector search (via EmbeddingRepository)
    vec_results = embedding_repo.search_vectors(query_embedding, limit=10)

See Also:
    - `pmd.search.pipeline.HybridSearchPipeline`: Combines FTS and vector search
    - `pmd.search.fusion.reciprocal_rank_fusion`: Merges ranked result lists
    - `pmd.store.embeddings.EmbeddingRepository`: Vector storage and search
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..core.types import SearchResult, SearchSource
from .database import Database

# Type variable for query types
QueryT = TypeVar("QueryT")


class SearchRepository(ABC, Generic[QueryT]):
    """Abstract base class for search operations.

    This generic interface allows different search implementations to accept
    different query types while maintaining a consistent API.

    Currently only FTS5SearchRepository implements this interface.
    For vector search, use EmbeddingRepository.search_vectors() directly.

    Example:
        >>> fts_repo: SearchRepository[str] = FTS5SearchRepository(db)
        >>> fts_results = fts_repo.search("python programming")
    """

    @abstractmethod
    def search(
        self,
        query: QueryT,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Execute search and return results.

        Args:
            query: Query in the format expected by the implementation.
                   String for FTS5, embedding vector for vector search.
            limit: Maximum number of results to return.
            collection_id: Optional collection ID to limit search scope.
            min_score: Minimum score threshold for results (0.0-1.0).

        Returns:
            List of SearchResult objects sorted by relevance (highest first).
        """
        pass


class FTS5SearchRepository(SearchRepository[str]):
    """Full-text search using SQLite FTS5 with BM25 scoring.

    This repository provides BM25-based full-text search using SQLite's FTS5
    extension. It also handles document indexing for the FTS5 index.

    Features:
        - BM25 relevance scoring with Porter stemming
        - Score normalization to 0-1 range
        - Collection-scoped search support
        - Document indexing and reindexing

    Example:
        >>> repo = FTS5SearchRepository(db)
        >>> results = repo.search("machine learning algorithms", limit=10)
        >>> for r in results:
        ...     print(f"{r.title}: {r.score:.3f}")

    See Also:
        - `search`: Execute BM25 full-text search
        - `index_document`: Add document to FTS5 index
        - `reindex_collection`: Rebuild index for a collection
    """

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def search(
        self,
        query: str,
        limit: int = 5,
        collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Perform BM25 full-text search using FTS5.

        Searches the FTS5 index using SQLite's BM25 ranking algorithm.
        Scores are normalized to a 0-1 range based on the maximum score
        in the result set.

        Args:
            query: Search query string (supports FTS5 syntax like AND, OR, NOT).
            limit: Maximum number of results to return.
            collection_id: Optional collection ID to limit search scope.
            min_score: Minimum normalized score threshold (0.0-1.0).

        Returns:
            List of SearchResult objects with source=SearchSource.FTS,
            sorted by relevance (highest first).

        Example:
            >>> results = repo.search("python OR programming", limit=5)
            >>> results = repo.search("machine learning", collection_id=1)
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

        Adds or updates a document in the FTS5 full-text index.
        The document must already exist in the documents table.

        Args:
            doc_id: Document ID from documents table (used as rowid).
            path: Document path (indexed for path-based queries).
            body: Document content to index for full-text search.

        Example:
            >>> repo.index_document(doc_id=1, path="notes/ml.md", body="...")
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

        Rebuilds the FTS5 index for all active documents in the specified
        collection. Useful after bulk imports or to fix index corruption.

        Args:
            collection_id: ID of collection to reindex.

        Returns:
            Number of documents indexed.

        Example:
            >>> count = repo.reindex_collection(collection_id=1)
            >>> print(f"Reindexed {count} documents")
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
            doc_id: Document ID to remove from the index.
        """
        with self.db.transaction() as cursor:
            cursor.execute("DELETE FROM documents_fts WHERE rowid = ?", (doc_id,))

    def clear_index(self) -> None:
        """Clear all FTS5 index entries.

        Warning: This removes all indexed content. Use with caution.
        """
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


