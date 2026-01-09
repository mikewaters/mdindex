"""Core metadata types, parsing, and ontology.

This module provides the foundational types and utilities for metadata
handling across both source extraction and search inference:

- Types: ExtractedMetadata, MetadataProfile protocol
- Parsing: Frontmatter extraction, inline tag parsing
- Ontology: Tag hierarchy for semantic matching
- Aliases: Tag alias mappings for normalization
"""


from .ontology import (
    Ontology,
    OntologyNode,
    load_default_ontology,
    load_ontology,
)
from .aliases import (
    TagAliases,
    load_aliases,
    load_default_aliases,
)

__all__ = [
    # Ontology
    "Ontology",
    "OntologyNode",
    "load_default_ontology",
    "load_ontology",
    # Aliases
    "TagAliases",
    "load_aliases",
    "load_default_aliases",
]
