"""Test fakes for testing without real infrastructure.

This module provides in-memory implementations of:
- Search ports (for testing the search pipeline)
- Repository protocols (for testing services without a database)

Example:
    from tests.fakes import (
        InMemoryDatabase,
        InMemoryCollectionRepository,
        InMemoryDocumentRepository,
        InMemoryFTSRepository,
    )

    # Create service with fakes
    service = IndexingService(
        db=InMemoryDatabase(),
        collection_repo=InMemoryCollectionRepository(),
        document_repo=InMemoryDocumentRepository(),
        fts_repo=InMemoryFTSRepository(),
    )
"""

from .search import (
    InMemoryTextSearcher,
    InMemoryVectorSearcher,
    InMemoryTagSearcher,
    StubQueryExpander,
    StubReranker,
    InMemoryMetadataBooster,
    InMemoryTagInferencer,
    make_search_result as make_search_result_search,
    make_ranked_result,
)

from .repos import (
    InMemoryDatabase,
    InMemoryCursor,
    InMemoryCollectionRepository,
    InMemoryDocumentRepository,
    InMemoryFTSRepository,
    InMemoryEmbeddingRepository,
)

# Use search module's make_search_result
make_search_result = make_search_result_search

__all__ = [
    # Search fakes
    "InMemoryTextSearcher",
    "InMemoryVectorSearcher",
    "InMemoryTagSearcher",
    "StubQueryExpander",
    "StubReranker",
    "InMemoryMetadataBooster",
    "InMemoryTagInferencer",
    "make_search_result",
    "make_ranked_result",
    # Repository fakes
    "InMemoryDatabase",
    "InMemoryCursor",
    "InMemoryCollectionRepository",
    "InMemoryDocumentRepository",
    "InMemoryFTSRepository",
    "InMemoryEmbeddingRepository",
]
