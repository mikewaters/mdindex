"""Service layer for PMD.

This module provides high-level services that orchestrate business logic,
abstracting the details of database access, LLM operations, and search
from the CLI interface.

Example usage:

    from pmd.app import create_application
    from pmd.core.config import Config

    async with await create_application(Config()) as app:
        # Index a collection
        result = await app.indexing.index_collection("my-docs", force=True)

        # Search
        results = await app.search.hybrid_search("machine learning", limit=10)

        # Get status
        status = app.status.get_index_status()
"""

from .caching import DocumentCacher
from .indexing import CleanupResult, EmbedResult, IndexingService, IndexResult
from .loading import EagerLoadResult, LoadedDocument, LoadingService, LoadResult
from .loading_llamaindex import LlamaIndexLoaderAdapter
from .search import SearchService
from .status import StatusService

__all__ = [
    "DocumentCacher",
    "IndexingService",
    "SearchService",
    "StatusService",
    "IndexResult",
    "EmbedResult",
    "CleanupResult",
    # Loading service types
    "LoadingService",
    "LoadedDocument",
    "LoadResult",
    "EagerLoadResult",
    "LlamaIndexLoaderAdapter",
]
