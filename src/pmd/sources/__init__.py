"""Document source abstractions for PMD.

The sources module now exposes a small, fixed set of sources and an adapter
for LlamaIndex loaders/readers. Callers are expected to construct the source
instance they need and inject it into indexing code; there is no registry or
URI-based dispatch.

Available sources:
- FileSystemSource: scan local files via glob.
- LlamaIndexSource: wrap a LlamaIndex reader/loader and adapt its documents.

Custom sources can still implement the DocumentSource protocol directly and
be passed to the indexer.
"""

from .base import (
    BaseDocumentSource,
    DocumentReference,
    DocumentSource,
    FetchResult,
    SourceCapabilities,
    SourceConfig,
    SourceError,
    SourceFetchError,
    SourceListError,
)
from .filesystem import FileSystemConfig, FileSystemSource
from .llamaindex import LlamaIndexSource
from .metadata import (
    DraftsProfile,
    ExtractedMetadata,
    GenericProfile,
    MetadataProfile,
    MetadataProfileRegistry,
    ObsidianProfile,
    get_default_profile_registry,
)

__all__ = [
    # Base types
    "BaseDocumentSource",
    "DocumentReference",
    "DocumentSource",
    "FetchResult",
    "SourceCapabilities",
    "SourceConfig",
    # Base exceptions
    "SourceError",
    "SourceFetchError",
    "SourceListError",
    # Filesystem source
    "FileSystemConfig",
    "FileSystemSource",
    # LlamaIndex adapter
    "LlamaIndexSource",
    # Metadata helpers
    "DraftsProfile",
    "ExtractedMetadata",
    "GenericProfile",
    "MetadataProfile",
    "MetadataProfileRegistry",
    "ObsidianProfile",
    "get_default_profile_registry",
]
