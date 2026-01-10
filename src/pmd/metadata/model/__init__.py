"""Core metadata model types and ontology.

This subpackage provides the foundational types for metadata handling:

- Types: ExtractedMetadata, StoredDocumentMetadata, MetadataProfile
- Ontology: Tag hierarchy for semantic matching
- Aliases: Tag alias mappings for normalization
"""

from .types import (
    ExtractedMetadata,
    MetadataProfile,
    StoredDocumentMetadata,
)
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
    # Types
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
]
