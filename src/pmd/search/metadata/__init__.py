"""Search-specific metadata scoring.

This subpackage provides the ScoredResult protocol and metadata boost functions
for search result scoring.

Other metadata functionality is available in pmd.ontology:
    from pmd.ontology.inference import LexicalTagMatcher
    from pmd.ontology.retrieval import TagRetriever
"""

# Re-export from new location (pmd.ontology)
from pmd.ontology.booster_scoring import (
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
