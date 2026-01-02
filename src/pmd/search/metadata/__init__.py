"""Metadata extraction for documents.

This module provides app-aware metadata extraction, allowing different
document sources (Obsidian, Drafts, etc.) to have their metadata
parsed and normalized appropriately.
"""

from .implementations import (
    DraftsProfile,
    GenericProfile,
    ObsidianProfile,
)
from .inference import (
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
)
from .scoring import (
    BoostResult,
    MetadataBoostConfig,
    WeightedBoostResult,
    apply_metadata_boost,
    apply_metadata_boost_v2,
)
from .ontology import (
    Ontology,
    OntologyNode,
    load_default_ontology,
    load_ontology,
)
from .parsers import (
    FrontmatterResult,
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)
from .profiles import (
    ExtractedMetadata,
    MetadataProfile,
)
from .registry import (
    MetadataProfileRegistry,
    get_default_profile_registry,
)

__all__ = [
    "BoostResult",
    "DraftsProfile",
    "ExtractedMetadata",
    "FrontmatterResult",
    "GenericProfile",
    "LexicalTagMatcher",
    "MetadataBoostConfig",
    "MetadataProfile",
    "MetadataProfileRegistry",
    "ObsidianProfile",
    "Ontology",
    "OntologyNode",
    "TagMatch",
    "WeightedBoostResult",
    "apply_metadata_boost",
    "apply_metadata_boost_v2",
    "create_default_matcher",
    "extract_inline_tags",
    "extract_tags_from_field",
    "get_default_profile_registry",
    "load_default_ontology",
    "load_ontology",
    "parse_frontmatter",
]
