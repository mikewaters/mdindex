"""idx.source - Source management.

Abstractions for reading, parsing, and extracting from dataset sources.
"""

from idx.source.directory import DirectorySource, SourceDocument
from idx.source.obsidian import ObsidianDocument, ObsidianVaultSource

__all__ = [
    "DirectorySource",
    "ObsidianDocument",
    "ObsidianVaultSource",
    "SourceDocument",
]
