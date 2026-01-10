"""DEPRECATED: Source-side metadata extraction utilities.

This module is deprecated. Import from pmd.metadata or pmd.metadata.extraction instead.

Migration guide:
    # Old (deprecated):
    from pmd.sources.metadata import GenericProfile, ObsidianProfile

    # New:
    from pmd.metadata import GenericProfile, ObsidianProfile
    # or
    from pmd.metadata.extraction import GenericProfile, ObsidianProfile
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'pmd.sources.metadata' is deprecated. "
    "Use 'pmd.metadata' or 'pmd.metadata.extraction' instead. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from pmd.metadata.extraction import (
    GenericProfile,
    DraftsProfile,
    ObsidianProfile,
    MetadataProfileRegistry,
    get_default_profile_registry,
    FrontmatterResult,
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)
from pmd.metadata.model import ExtractedMetadata

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
    "FrontmatterResult",
]
