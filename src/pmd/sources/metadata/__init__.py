"""Source-side metadata extraction utilities."""

from .implementations import DraftsProfile, GenericProfile, ObsidianProfile
from .parsers import (
    FrontmatterResult,
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)
from .profiles import ExtractedMetadata, MetadataProfile
from .registry import MetadataProfileRegistry, get_default_profile_registry

__all__ = [
    "DraftsProfile",
    "GenericProfile",
    "ObsidianProfile",
    "FrontmatterResult",
    "extract_inline_tags",
    "extract_tags_from_field",
    "parse_frontmatter",
    "ExtractedMetadata",
    "MetadataProfile",
    "MetadataProfileRegistry",
    "get_default_profile_registry",
]
