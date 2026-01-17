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

# Re-export from new location (pmd.extraction)
from pmd.extraction.profiles.generic import GenericProfile
from pmd.extraction.profiles.drafts import DraftsProfile
from pmd.extraction.profiles.obsidian import ObsidianProfile

# Registry
from pmd.extraction.registry import (
    MetadataProfileRegistry,
    get_default_profile_registry,
    DetectorFunc,
    ProfileRegistration,
)

# Parsing utilities
from pmd.extraction.profiles.parsing import (
    FrontmatterResult,
    extract_inline_tags,
    extract_tags_from_field,
    parse_frontmatter,
)

# Detection functions (for custom registry setup)
from pmd.extraction.profiles.obsidian import detect_obsidian_content
from pmd.extraction.profiles.drafts import detect_drafts_content

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
