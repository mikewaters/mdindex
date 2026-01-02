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
    "DraftsProfile",
    "ExtractedMetadata",
    "FrontmatterResult",
    "GenericProfile",
    "MetadataProfile",
    "MetadataProfileRegistry",
    "ObsidianProfile",
    "extract_inline_tags",
    "extract_tags_from_field",
    "get_default_profile_registry",
    "parse_frontmatter",
]
