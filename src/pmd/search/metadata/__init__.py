"""Query-time metadata inference and scoring for search.

This module provides search-specific metadata functionality:
- LexicalTagMatcher: Infer tags from search queries
- TagRetriever: Tag-based document retrieval for RRF fusion
- Scoring: Metadata-based score boosting

For core metadata types, use pmd.metadata.
For extraction profiles, use pmd.sources.metadata.
"""

from .inference import (
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
)
from .retrieval import (
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
)
from .scoring import (
    BoostResult,
    MetadataBoostConfig,
    ScoredResult,
    WeightedBoostResult,
    apply_metadata_boost,
    apply_metadata_boost_v2,
    build_path_to_id_map,
    get_document_tags_batch,
)

__all__ = [
    # Inference
    "LexicalTagMatcher",
    "TagMatch",
    "create_default_matcher",
    # Retrieval
    "TagRetriever",
    "TagSearchConfig",
    "create_tag_retriever",
    # Scoring
    "BoostResult",
    "MetadataBoostConfig",
    "ScoredResult",
    "WeightedBoostResult",
    "apply_metadata_boost",
    "apply_metadata_boost_v2",
    "build_path_to_id_map",
    "get_document_tags_batch",
]
