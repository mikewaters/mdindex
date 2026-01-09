"""Source-side metadata extraction utilities.

This module provides app-aware metadata extraction profiles for
different document sources (Obsidian, Drafts, generic markdown).

Core types are re-exported from pmd.metadata for convenience.
"""

# Re-export core types from pmd.metadata


# Source-specific implementations
from .base import GenericProfile
from .drafts import DraftsProfile
from .obsidian import ObsidianProfile
from .registry import MetadataProfileRegistry, get_default_profile_registry
from .types import (
    ExtractedMetadata
)
from .parsing import (
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
    FrontmatterResult
)

__all__ = [

    "DraftsProfile",
    "GenericProfile",
    "ObsidianProfile",
    "MetadataProfileRegistry",
    "get_default_profile_registry",
    "ExtractedMetadata",
    "extract_inline_tags",
    "extract_tags_from_field",
    "parse_frontmatter",
    "FrontmatterResult"
]
