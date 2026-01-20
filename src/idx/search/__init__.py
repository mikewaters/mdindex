"""idx.search - Search abstractions.

Full-text search, vector search, hybrid retrieval (RRF),
and LLM-as-judge reranking. Called from the orchestration layer.
"""

from idx.search.fts import FTSSearch
from idx.search.hybrid import HybridSearch, rrf_fusion
from idx.search.models import SearchCriteria, SearchResult, SearchResults
from idx.search.vector import VectorIndexer, VectorSearch, create_vector_node

__all__ = [
    "FTSSearch",
    "HybridSearch",
    "SearchCriteria",
    "SearchResult",
    "SearchResults",
    "VectorIndexer",
    "VectorSearch",
    "create_vector_node",
    "rrf_fusion",
]
