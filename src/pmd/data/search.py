"""Data access layer for search operations.

This module provides the SearchData class which wraps all repositories
needed by SearchService, simplifying dependency injection.
"""

from pmd.core.types import SearchResult, SourceCollection
from pmd.store.database import Database
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.repositories.fts import FTS5SearchRepository


class SearchData:
    """Data access layer for search operations.

    Wraps all repositories needed by SearchService, taking only Database
    in its constructor. This simplifies dependency injection and provides
    a clean interface for search-related data access.

    Example:
        data = SearchData(db)
        collection = data.get_collection_by_name("my_collection")
        results = data.search("query text", limit=10)
    """

    def __init__(self, db: Database) -> None:
        """Initialize with database connection.

        Creates all required repositories internally.

        Args:
            db: Database instance to use for operations.
        """
        self._db = db
        self._collections = SourceCollectionRepository(db)
        self._fts = FTS5SearchRepository(db)

    @property
    def vec_available(self) -> bool:
        """Check if vector search is available.

        Returns:
            True if sqlite-vec extension is loaded, False otherwise.
        """
        return self._db.vec_available

    def get_collection_by_name(self, name: str) -> SourceCollection | None:
        """Get source collection by name.

        Used to resolve collection names to IDs for scoped searches.

        Args:
            name: Source collection name to look up.

        Returns:
            SourceCollection if found, None otherwise.
        """
        return self._collections.get_by_name(name)

    def search(
        self,
        query: str,
        limit: int = 5,
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Execute FTS5 full-text search.

        Performs BM25-scored full-text search using SQLite's FTS5 extension.

        Args:
            query: Search query string (supports FTS5 syntax like AND, OR, NOT).
            limit: Maximum number of results to return.
            source_collection_id: Optional collection ID to limit search scope.
            min_score: Minimum normalized score threshold (0.0-1.0).

        Returns:
            List of SearchResult objects sorted by relevance (highest first).
        """
        return self._fts.search(
            query,
            limit=limit,
            source_collection_id=source_collection_id,
            min_score=min_score,
        )
