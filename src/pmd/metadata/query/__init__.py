"""Query-time metadata inference and scoring for search.

This subpackage provides search-specific metadata functionality:

Inference:
- LexicalTagMatcher: Infer tags from search queries
- TagMatch: Matched tag with confidence score
- create_default_matcher: Factory with default aliases

Retrieval:
- TagRetriever: Tag-based document retrieval for RRF fusion
- TagSearchConfig: Configuration for tag search

Scoring:
- apply_metadata_boost: Simple count-based boosting
- apply_metadata_boost_v2: Weighted boosting with ontology expansion
- MetadataBoostConfig, BoostResult, WeightedBoostResult: Config and result types

Utilities:
- get_document_tags_batch: Efficient batch tag lookup
- build_path_to_id_map: Path to document ID mapping
"""

# Inference
from pmd.metadata.query.inference import (
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
    load_default_aliases,
)

# Retrieval
from pmd.metadata.query.retrieval import (
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
)

# Scoring
from pmd.metadata.query.scoring import (
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
    "load_default_aliases",
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
