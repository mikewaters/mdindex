"""Unified metadata domain for PMD.

This package provides the complete metadata handling subsystem with clear
submodules for different concerns:

Subpackages
-----------
model
    Core types: ExtractedMetadata, StoredDocumentMetadata, Ontology, TagAliases
extraction
    Source-aware extraction: profiles (Obsidian, Drafts, Generic), parsing
query
    Query-time operations: tag inference, retrieval, scoring
store
    Persistence: DocumentMetadataRepository

Public API
----------
This module re-exports commonly used types from subpackages for convenience.
For specialized functionality, import directly from subpackages:

    # Core types (most common)
    from pmd.metadata import Ontology, ExtractedMetadata, TagAliases

    # Extraction profiles
    from pmd.metadata.extraction import ObsidianProfile, get_default_profile_registry

    # Query-time
    from pmd.metadata.query import LexicalTagMatcher, create_default_matcher

    # Storage (now in pmd.store.repositories)
    from pmd.store.repositories.metadata import DocumentMetadataRepository

Example
-------
>>> from pmd.metadata import Ontology, TagAliases, ExtractedMetadata
>>> from pmd.metadata.extraction import GenericProfile
>>> from pmd.metadata.query import create_default_matcher
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
# Model - Core types and ontology
# =============================================================================
from pmd.metadata.model import (
    # Types
    ExtractedMetadata,
    MetadataProfile,
    StoredDocumentMetadata,
    # Ontology
    Ontology,
    OntologyNode,
    load_default_ontology,
    load_ontology,
    # Aliases
    TagAliases,
    load_aliases,
    load_default_aliases,
)

# =============================================================================
# Extraction - Source-aware metadata extraction
# =============================================================================
from pmd.metadata.extraction import (
    # Profiles
    GenericProfile,
    DraftsProfile,
    ObsidianProfile,
    # Registry
    MetadataProfileRegistry,
    get_default_profile_registry,
    # Parsing utilities
    FrontmatterResult,
    parse_frontmatter,
    extract_inline_tags,
    extract_tags_from_field,
)

# =============================================================================
# Query - Query-time inference and scoring
# =============================================================================
from pmd.metadata.query import (
    # Inference
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
    # Retrieval
    TagRetriever,
    TagSearchConfig,
    create_tag_retriever,
    # Scoring
    MetadataBoostConfig,
    BoostResult,
    WeightedBoostResult,
    apply_metadata_boost,
    apply_metadata_boost_v2,
    build_path_to_id_map,
    get_document_tags_batch,
)

__all__ = [
    # === Model Types ===
    "ExtractedMetadata",
    "MetadataProfile",
    "StoredDocumentMetadata",
    # Ontology
    "Ontology",
    "OntologyNode",
    "load_default_ontology",
    "load_ontology",
    # Aliases
    "TagAliases",
    "load_aliases",
    "load_default_aliases",
    # === Extraction ===
    # Profiles
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
