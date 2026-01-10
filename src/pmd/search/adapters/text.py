"""FTS5 text search adapter.

Wraps FTS5SearchRepository to implement the TextSearcher protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmd.core.types import SearchResult
    from pmd.store.search import FTS5SearchRepository


class FTS5TextSearcher:
    """Adapter that wraps FTS5SearchRepository for the TextSearcher protocol.

    This adapter provides a thin wrapper around FTS5SearchRepository,
    implementing the TextSearcher protocol for use in HybridSearchPipeline.

    Example:
        >>> from pmd.store.search import FTS5SearchRepository
        >>> fts_repo = FTS5SearchRepository(db)
        >>> searcher = FTS5TextSearcher(fts_repo)
        >>> results = searcher.search("python programming", limit=10)
    """

    def __init__(self, fts_repo: "FTS5SearchRepository"):
        """Initialize with FTS5 repository.

        Args:
            fts_repo: FTS5SearchRepository instance.
        """
        self._repo = fts_repo

    def search(
        self,
        query: str,
        limit: int,
        collection_id: int | None = None,
    ) -> list["SearchResult"]:
        """Search documents using full-text search.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.
            collection_id: Optional collection to scope search.

        Returns:
            List of SearchResult objects sorted by relevance score.
        """
        return self._repo.search(query, limit=limit, collection_id=collection_id)
