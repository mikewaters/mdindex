"""Metadata extraction profile implementations.

Provides built-in profiles for extracting metadata from documents:
- GenericProfile: Default profile for most markdown documents
- ObsidianProfile: Specialized for Obsidian vault documents
- DraftsProfile: Specialized for Drafts app documents
"""

from pmd.extraction.profiles.generic import GenericProfile
from pmd.extraction.profiles.obsidian import ObsidianProfile, detect_obsidian_content
from pmd.extraction.profiles.drafts import DraftsProfile, detect_drafts_content

__all__ = [
    "GenericProfile",
    "ObsidianProfile",
    "DraftsProfile",
    "detect_obsidian_content",
    "detect_drafts_content",
]
