"""Backward-compatible shim for metadata profile implementations."""

from pmd.sources.metadata.implementations import (
    DraftsProfile,
    GenericProfile,
    ObsidianProfile,
)

__all__ = ["DraftsProfile", "GenericProfile", "ObsidianProfile"]
