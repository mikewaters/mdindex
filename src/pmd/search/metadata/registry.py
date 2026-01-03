"""Backward-compatible shim for metadata registry helpers."""

from pmd.sources.metadata.registry import (
    MetadataProfileRegistry,
    ProfileRegistration,
    get_default_profile_registry,
)

__all__ = [
    "MetadataProfileRegistry",
    "ProfileRegistration",
    "get_default_profile_registry",
]
