"""Adapters that implement search ports using concrete infrastructure.

This module provides adapter classes that wrap existing infrastructure
(repositories, LLM providers) and implement the port protocols defined
in pmd.app.protocols.

Adapters:
    - FTS5TextSearcher: Wraps FTS5SearchRepository
    - EmbeddingVectorSearcher: Wraps EmbeddingGenerator + EmbeddingRepository
    - TagRetrieverAdapter: Wraps TagRetriever
    - LLMQueryExpanderAdapter: Wraps QueryExpander
    - LLMRerankerAdapter: Wraps DocumentReranker
    - OntologyMetadataBooster: Encapsulates metadata repo + ontology
    - LexicalTagInferencer: Wraps LexicalTagMatcher + Ontology

Usage:
    from pmd.search.adapters import (
        FTS5TextSearcher,
        EmbeddingVectorSearcher,
        OntologyMetadataBooster,
    )

    # Create adapters from infrastructure
    text_searcher = FTS5TextSearcher(fts_repo)
    vector_searcher = EmbeddingVectorSearcher(embedding_generator)

    # Use with HybridSearchPipeline
    pipeline = HybridSearchPipeline(
        text_searcher=text_searcher,
        vector_searcher=vector_searcher,
    )
"""

from .text import FTS5TextSearcher
from .vector import EmbeddingVectorSearcher
from .tag import TagRetrieverAdapter, LexicalTagInferencer
from .expansion import LLMQueryExpanderAdapter
from .rerank import LLMRerankerAdapter
from .boost import OntologyMetadataBooster

__all__ = [
    "FTS5TextSearcher",
    "EmbeddingVectorSearcher",
    "TagRetrieverAdapter",
    "LexicalTagInferencer",
    "LLMQueryExpanderAdapter",
    "LLMRerankerAdapter",
    "OntologyMetadataBooster",
]
