"""idx.search.fts - Full-text search implementation.

Provides FTS search with BM25 normalization and dataset filtering.
Uses ambient session via contextvars.

Example usage:
    from idx.search.fts import FTSSearch
    from idx.store.database import get_session
    from idx.store.session_context import use_session

    with get_session() as session:
        with use_session(session):
            search = FTSSearch()
            results = search.search(SearchCriteria(query="hello", mode="fts"))
"""

from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.search.models import SearchCriteria, SearchResult, SearchResults
from idx.store.fts import FTSManager
from idx.store.session_context import current_session

__all__ = [
    "FTSSearch",
]

logger = get_logger(__name__)


class FTSSearch:
    """Full-text search implementation using FTS5.

    Provides FTS query passthrough with BM25 score normalization
    and optional dataset filtering.

    Uses ambient session via contextvars. The session must be set
    via `use_session()` before calling search methods.

    Example:
        with get_session() as session:
            with use_session(session):
                search = FTSSearch()
                results = search.search(
                    SearchCriteria(query="python tutorial", limit=10)
                )
    """

    def __init__(self, session: Session | None = None) -> None:
        """Initialize the FTS search.

        Args:
            session: Optional SQLAlchemy session. If None, uses ambient
                session from current_session(). Providing explicit session
                is deprecated; prefer using ambient session pattern.
        """
        self._explicit_session = session
        # FTSManager uses ambient session by default
        self._fts = FTSManager(session)

    @property
    def _session(self) -> Session:
        """Get the session to use for database operations.

        Returns explicit session if provided, otherwise ambient session.
        """
        if self._explicit_session is not None:
            return self._explicit_session
        return current_session()

    def search(self, criteria: SearchCriteria) -> SearchResults:
        """Execute an FTS search.

        Args:
            criteria: Search criteria including query, limit, and dataset filter.

        Returns:
            SearchResults with normalized BM25 scores.
        """
        import time

        start = time.perf_counter()

        # Get dataset ID if filtering by name
        dataset_filter = None
        if criteria.dataset_name:
            dataset_filter = self._resolve_dataset_id(criteria.dataset_name)
            if dataset_filter is None:
                # Dataset not found, return empty results
                logger.warning(f"Dataset not found: {criteria.dataset_name}")
                return SearchResults(
                    results=[],
                    query=criteria.query,
                    mode="fts",
                    total_candidates=0,
                    timing_ms=0,
                )

        # Execute search with normalized scores
        raw_results = self._fts.search_with_scores(
            criteria.query,
            limit=criteria.limit,
            dataset_filter=dataset_filter,
        )

        # Convert to SearchResult objects
        results = []
        for doc_id, path, score in raw_results:
            # Get dataset name for the result
            ds_name = self._get_dataset_name_for_doc(doc_id) or ""

            results.append(
                SearchResult(
                    path=path,
                    dataset_name=ds_name,
                    score=score,
                    scores={"fts": score},
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            f"FTS search '{criteria.query}' returned {len(results)} results in {elapsed_ms:.1f}ms"
        )

        return SearchResults(
            results=results,
            query=criteria.query,
            mode="fts",
            total_candidates=len(raw_results),
            timing_ms=elapsed_ms,
        )

    def _resolve_dataset_id(self, dataset_name: str) -> int | None:
        """Resolve dataset name to ID.

        Args:
            dataset_name: Name of the dataset.

        Returns:
            Dataset ID if found, None otherwise.
        """
        from sqlalchemy import text

        from idx.store.dataset import normalize_dataset_name

        normalized = normalize_dataset_name(dataset_name)
        result = self._session.execute(
            text("SELECT id FROM datasets WHERE name = :name"),
            {"name": normalized},
        )
        row = result.fetchone()
        return row[0] if row else None

    def _get_dataset_name_for_doc(self, doc_id: int) -> str | None:
        """Get dataset name for a document.

        Args:
            doc_id: Document ID.

        Returns:
            Dataset name if found, None otherwise.
        """
        from sqlalchemy import text

        result = self._session.execute(
            text("""
                SELECT ds.name
                FROM documents d
                JOIN datasets ds ON ds.id = d.dataset_id
                WHERE d.id = :doc_id
            """),
            {"doc_id": doc_id},
        )
        row = result.fetchone()
        return row[0] if row else None
