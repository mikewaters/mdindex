"""Service layer for PMD.

This module provides high-level services that orchestrate business logic,
abstracting the details of database access, LLM operations, and search
from the CLI and MCP interfaces.

Example usage:

    from pmd.services import ServiceContainer
    from pmd.sources import get_default_registry

    async with ServiceContainer(config) as services:
        # Index a collection using source registry
        collection = services.collection_repo.get_by_name("my-docs")
        registry = get_default_registry()
        source = registry.create_source(collection)
        result = await services.indexing.index_collection(
            "my-docs",
            force=True,
            source=source,
        )

        # Search
        results = await services.search.hybrid_search("machine learning", limit=10)

        # Get status
        status = services.status.get_index_status()
"""

from .container import ServiceContainer
from .indexing import CleanupResult, EmbedResult, IndexingService, IndexResult
from .search import SearchService
from .status import StatusService

__all__ = [
    "ServiceContainer",
    "IndexingService",
    "SearchService",
    "StatusService",
    "IndexResult",
    "EmbedResult",
    "CleanupResult",
]
