"""Search-specific metadata scoring.

This subpackage provides the ScoredResult protocol and metadata boost functions
for search result scoring.

Other metadata functionality has moved to pmd.metadata:
    from pmd.metadata import LexicalTagMatcher, TagRetriever
"""

from pmd.search.metadata.scoring import (
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
    "BoostResult",
    "MetadataBoostConfig",
    "ScoredResult",
    "WeightedBoostResult",
    "apply_metadata_boost",
    "apply_metadata_boost_v2",
    "build_path_to_id_map",
    "get_document_tags_batch",
]
