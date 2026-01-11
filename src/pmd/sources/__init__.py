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

from .content import (
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
from .content import FileSystemConfig, FileSystemSource
from .content import LlamaIndexSource
from .content.llamaindex import (
    create_custom_loader,
    create_directory_loader,
    create_web_loader,
)
from .content import (
    SourceFactory,
    SourceRegistry,
    get_default_registry,
    reset_default_registry,
)
from pmd.metadata import (
    GenericProfile,
    MetadataProfileRegistry,
    get_default_profile_registry,
    ObsidianProfile,
    DraftsProfile
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
    # Source registry
    "SourceFactory",
    "SourceRegistry",
    "get_default_registry",
    "reset_default_registry",
    # Filesystem source
    "FileSystemConfig",
    "FileSystemSource",
    # LlamaIndex adapter
    "LlamaIndexSource",
    "create_custom_loader",
    "create_directory_loader",
    "create_web_loader",
    # Metadata helpers
    "DraftsProfile",
    "GenericProfile",
    "MetadataProfileRegistry",
    "ObsidianProfile",
    "get_default_profile_registry",
]
