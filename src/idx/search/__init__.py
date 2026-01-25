"""idx.search - Search abstractions.

Full-text search, vector search, hybrid retrieval (RRF),
and LLM-as-judge reranking. Called from the orchestration layer.
"""

from idx.search.fts import FTSSearch
from idx.search.fts_chunk import FTSChunkRetriever
from idx.search.models import SearchCriteria, SearchResult, SearchResults
from idx.search.rerank import Reranker
from idx.search.service import SearchService

__all__ = [
    "FTSChunkRetriever",
    "FTSSearch",
    "Reranker",
    "SearchCriteria",
    "SearchResult",
    "SearchResults",
    "SearchService",
]
