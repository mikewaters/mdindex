"""Unified metadata domain for PMD.

This package provides the complete metadata handling subsystem. The implementation
has been split into focused modules:

- pmd.extraction: Document metadata extraction profiles and types
- pmd.ontology: Tag ontology, matching, and scoring

This module re-exports commonly used types for backward compatibility.

Public API
----------
    # Core types (most common)
    from pmd.metadata import Ontology, ExtractedMetadata, TagAliases

    # Extraction profiles
    from pmd.metadata import ObsidianProfile, get_default_profile_registry

    # Query-time
    from pmd.metadata import LexicalTagMatcher, create_default_matcher

    # Storage (now in pmd.store.repositories)
    from pmd.store.repositories.metadata import DocumentMetadataRepository

Example
-------
>>> from pmd.metadata import Ontology, TagAliases, ExtractedMetadata
>>> from pmd.metadata import GenericProfile, create_default_matcher
>>>
>>> # Extract metadata from document
>>> profile = GenericProfile()
>>> metadata = profile.extract_metadata(content, path)
>>>
>>> # Infer tags from query
>>> matcher = create_default_matcher()
>>> tags = matcher.get_matching_tags("python web api")
"""

# =============================================================================
# Extraction Types - from pmd.extraction
# =============================================================================
from pmd.extraction.types import (
    ExtractedMetadata,
    MetadataProfile,
    StoredDocumentMetadata,
)

# =============================================================================
# Extraction Profiles - from pmd.extraction.profiles
# =============================================================================
from pmd.extraction.profiles import (
    GenericProfile,
    DraftsProfile,
    ObsidianProfile,
)

# =============================================================================
# Extraction Registry - from pmd.extraction
# =============================================================================
from pmd.extraction.registry import (
    MetadataProfileRegistry,
    get_default_profile_registry,
)

# =============================================================================
# Parsing utilities - from pmd.extraction.profiles.parsing
# =============================================================================
from pmd.extraction.profiles.parsing import (
    FrontmatterResult,
    parse_frontmatter,
    extract_inline_tags,
    extract_tags_from_field,
)

# =============================================================================
# Ontology - from pmd.ontology
# =============================================================================
from pmd.ontology.model import (
    Ontology,
    OntologyNode,
    load_default_ontology,
    load_ontology,
)
from pmd.ontology.aliases import (
    TagAliases,
    load_aliases,
    load_default_aliases,
)

# =============================================================================
# Query-time inference - from pmd.ontology.inference
# =============================================================================
from pmd.ontology.inference import (
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
)

# =============================================================================
# Tag retrieval - from pmd.ontology.retrieval
# =============================================================================
from pmd.ontology.retrieval import (
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
)

# =============================================================================
# Metadata scoring - from pmd.ontology.scoring
# =============================================================================
from pmd.ontology.scoring import (
    MetadataBoostConfig,
    BoostResult,
    WeightedBoostResult,
    apply_metadata_boost,
    apply_metadata_boost_v2,
    build_path_to_id_map,
    get_document_tags_batch,
)

__all__ = [
    # === Extraction Types ===
    "ExtractedMetadata",
    "MetadataProfile",
    "StoredDocumentMetadata",
    # === Ontology ===
    "Ontology",
    "OntologyNode",
    "load_default_ontology",
    "load_ontology",
    # Aliases
    "TagAliases",
    "load_aliases",
    "load_default_aliases",
    # === Extraction Profiles ===
    "GenericProfile",
    "DraftsProfile",
    "ObsidianProfile",
    # Registry
    "MetadataProfileRegistry",
    "get_default_profile_registry",
    # Parsing
    "FrontmatterResult",
    "parse_frontmatter",
    "extract_inline_tags",
    "extract_tags_from_field",
    # === Query ===
    # Inference
    "LexicalTagMatcher",
    "TagMatch",
    "create_default_matcher",
    # Retrieval
    "TagRetriever",
    "TagSearchConfig",
    "create_tag_retriever",
    # Scoring
    "MetadataBoostConfig",
    "BoostResult",
    "WeightedBoostResult",
    "apply_metadata_boost",
    "apply_metadata_boost_v2",
    "build_path_to_id_map",
    "get_document_tags_batch",
]
