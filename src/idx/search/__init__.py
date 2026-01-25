"""idx.search - Search abstractions.

Full-text search, vector search, hybrid retrieval (RRF),
and LLM-as-judge reranking. Called from the orchestration layer.

Example usage:
    from idx.search import search

    # Simple search with automatic session management
    results = search("python async", mode="hybrid")
    for r in results.results:
        print(f"{r.path}: {r.score:.3f}")

    # For more control, use SearchService directly
    from idx.search import SearchService, SearchCriteria
    service = SearchService()
    results = service.search(SearchCriteria(query="...", mode="fts"))
"""

from idx.search.fts import FTSSearch
from idx.search.fts_chunk import FTSChunkRetriever
from idx.search.models import SearchCriteria, SearchResult, SearchResults
from idx.search.rerank import Reranker
from idx.search.service import SearchService


def search(
    query: str,
    mode: str = "hybrid",
    limit: int = 10,
    dataset_name: str | None = None,
) -> SearchResults:
    """Execute a search with automatic session management.

    Convenience function that handles database session setup internally.
    For more control, use SearchService directly.

    Args:
        query: Search query string.
        mode: Search mode - "fts", "vector", or "hybrid".
        limit: Maximum results to return.
        dataset_name: Optional filter to specific dataset.

    Returns:
        SearchResults with matching documents.
    """
    from idx.store.database import get_session
    from idx.store.session_context import use_session

    with get_session() as session:
        with use_session(session):
            service = SearchService()
            return service.search(
                SearchCriteria(
                    query=query,
                    mode=mode,
                    limit=limit,
                    dataset_name=dataset_name,
                )
            )


__all__ = [
    "search",
    "FTSChunkRetriever",
    "FTSSearch",
    "Reranker",
    "SearchCriteria",
    "SearchResult",
    "SearchResults",
    "SearchService",
]
