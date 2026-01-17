"""Metadata profile registry with auto-detection.

Provides a registry for managing metadata profiles and selecting
the appropriate profile based on document path and content.

At ingest time, if the user has not provided an explicit profile for a collection,
we can detect it using some heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

from pmd.extraction.profiles.drafts import detect_drafts_content
from pmd.extraction.profiles.obsidian import detect_obsidian_content

if TYPE_CHECKING:
    from pmd.extraction.types import MetadataProfile


# Type for detector functions
# Takes (content, path) and returns True if the profile should be used
DetectorFunc = Callable[[str, str], bool]


@dataclass
class ProfileRegistration:
    """Registration entry for a metadata profile.

    Attributes:
        profile: The metadata profile instance.
        path_patterns: Regex patterns to match against document paths.
        detectors: Additional detector functions for content-based matching.
        priority: Higher priority profiles are checked first (default: 0).
    """

    profile: "MetadataProfile"
    path_patterns: list[re.Pattern[str]] = field(default_factory=list)
    detectors: list[DetectorFunc] = field(default_factory=list)
    priority: int = 0


class MetadataProfileRegistry:
    """Registry for metadata profiles with auto-detection.

    Manages profile registration and selection. Profiles can be:
    - Selected explicitly by name
    - Auto-detected based on path patterns
    - Auto-detected based on content inspection
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._profiles: dict[str, ProfileRegistration] = {}

    def register(
        self,
        profile: "MetadataProfile",
        *,
        path_patterns: list[str] | None = None,
        detectors: list[DetectorFunc] | None = None,
        priority: int = 0,
        override: bool = False,
    ) -> None:
        """Register a metadata profile."""
        name = profile.name

        if name in self._profiles and not override:
            raise ValueError(
                f"Profile '{name}' is already registered. "
                f"Use override=True to replace."
            )

        compiled_patterns = []
        if path_patterns:
            compiled_patterns = [re.compile(p, re.IGNORECASE) for p in path_patterns]

        self._profiles[name] = ProfileRegistration(
            profile=profile,
            path_patterns=compiled_patterns,
            detectors=detectors or [],
            priority=priority,
        )

    def unregister(self, name: str) -> bool:
        """Remove a registered profile."""
        if name in self._profiles:
            del self._profiles[name]
            return True
        return False

    def get(self, name: str) -> "MetadataProfile | None":
        """Get a profile by name."""
        registration = self._profiles.get(name)
        return registration.profile if registration else None

    def detect(self, content: str, path: str) -> "MetadataProfile | None":
        """Auto-detect the appropriate profile for a document."""
        # Sort by priority (highest first)
        sorted_regs = sorted(
            self._profiles.values(),
            key=lambda r: r.priority,
            reverse=True,
        )

        for reg in sorted_regs:
            # Check path patterns
            for pattern in reg.path_patterns:
                if pattern.search(path):
                    return reg.profile

            # Check content detectors
            for detector in reg.detectors:
                try:
                    if detector(content, path):
                        return reg.profile
                except Exception:
                    # Detector failed, skip it
                    continue

        return None

    def detect_or_default(
        self,
        content: str,
        path: str,
        default_name: str = "generic",
    ) -> "MetadataProfile":
        """Auto-detect profile or return a default."""
        profile = self.detect(content, path)
        if profile:
            return profile

        default = self.get(default_name)
        if not default:
            raise ValueError(
                f"Default profile '{default_name}' is not registered. "
                f"Register a profile with name='{default_name}' first."
            )
        return default

    def list_profiles(self) -> list[str]:
        """Get list of all registered profile names."""
        return sorted(self._profiles.keys())

    def is_registered(self, name: str) -> bool:
        """Check if a profile is registered."""
        return name in self._profiles


# =============================================================================
# Default Registry
# =============================================================================

_default_registry: MetadataProfileRegistry | None = None


def get_default_profile_registry() -> MetadataProfileRegistry:
    """Get the default global profile registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = MetadataProfileRegistry()
        _register_builtin_profiles(_default_registry)
    return _default_registry


def _register_builtin_profiles(registry: MetadataProfileRegistry) -> None:
    """Register built-in metadata profiles."""
    from pmd.extraction.profiles.generic import GenericProfile
    from pmd.extraction.profiles.drafts import DraftsProfile
    from pmd.extraction.profiles.obsidian import ObsidianProfile

    # Generic profile as lowest-priority fallback
    registry.register(
        GenericProfile(),
        priority=-100,
    )

    # Obsidian profile with path detection
    registry.register(
        ObsidianProfile(),
        path_patterns=[
            r"\.obsidian",  # .obsidian config directory
            r"vault",  # common vault name
        ],
        detectors=[detect_obsidian_content],
        priority=10,
    )

    # Drafts profile with path detection
    registry.register(
        DraftsProfile(),
        path_patterns=[
            r"drafts",  # drafts directory
            r"\.drafts",  # .drafts config
        ],
        detectors=[detect_drafts_content],
        priority=10,
    )
