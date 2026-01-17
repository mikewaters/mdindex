"""Document metadata extraction module.

Provides:
- ExtractedMetadata: Core metadata type for extracted document metadata
- MetadataProfile: Protocol for metadata extraction profiles
- MetadataProfileRegistry: Registry for managing profiles with auto-detection
- Profile implementations: GenericProfile, ObsidianProfile, DraftsProfile
"""

from pmd.extraction.types import ExtractedMetadata, MetadataProfile
from pmd.extraction.registry import MetadataProfileRegistry, ProfileRegistration

__all__ = [
    "ExtractedMetadata",
    "MetadataProfile",
    "MetadataProfileRegistry",
    "ProfileRegistration",
]
