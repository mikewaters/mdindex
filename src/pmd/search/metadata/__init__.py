"""DEPRECATED: Query-time metadata inference and scoring for search.

This module is deprecated. Import from pmd.metadata or pmd.metadata.query instead.

Migration guide:
    # Old (deprecated):
    from pmd.search.metadata import LexicalTagMatcher, TagRetriever

    # New:
    from pmd.metadata import LexicalTagMatcher, TagRetriever
    # or
    from pmd.metadata.query import LexicalTagMatcher, TagRetriever
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'pmd.search.metadata' is deprecated. "
    "Use 'pmd.metadata' or 'pmd.metadata.query' instead. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from pmd.metadata.query import (
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
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
