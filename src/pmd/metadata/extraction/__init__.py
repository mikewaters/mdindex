"""Source-aware metadata extraction.

This subpackage provides app-aware metadata extraction profiles for
different document sources (Obsidian, Drafts, generic markdown).

Profiles:
- GenericProfile: Fallback for standard markdown
- ObsidianProfile: Obsidian vault documents with nested tags
- DraftsProfile: Drafts app exports

Registry:
- MetadataProfileRegistry: Manages profiles with auto-detection
- get_default_profile_registry: Returns pre-configured registry

Parsing utilities:
- parse_frontmatter: Extract YAML frontmatter
- extract_inline_tags: Find #tags in content
- extract_tags_from_field: Parse tags from various YAML formats
"""

# Profiles
from pmd.metadata.extraction.generic import GenericProfile
from pmd.metadata.extraction.drafts import DraftsProfile
from pmd.metadata.extraction.obsidian import ObsidianProfile

# Registry
from pmd.metadata.extraction.registry import (
    MetadataProfileRegistry,
    get_default_profile_registry,
    DetectorFunc,
    ProfileRegistration,
)

# Parsing utilities
from pmd.metadata.extraction.parsing import (
    FrontmatterResult,
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)

# Detection functions (for custom registry setup)
from pmd.metadata.extraction.obsidian import detect_obsidian_content
from pmd.metadata.extraction.drafts import detect_drafts_content

__all__ = [
    # Profiles
    "GenericProfile",
    "DraftsProfile",
    "ObsidianProfile",
    # Registry
    "MetadataProfileRegistry",
    "get_default_profile_registry",
    "DetectorFunc",
    "ProfileRegistration",
    # Parsing
    "FrontmatterResult",
    "extract_inline_tags",
    "extract_tags_from_field",
    "parse_frontmatter",
    # Detection
    "detect_obsidian_content",
    "detect_drafts_content",
]
