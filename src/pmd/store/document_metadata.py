"""DEPRECATED: Document metadata storage for PMD.

This module is deprecated. Import from pmd.metadata or pmd.metadata.store instead.

Migration guide:
    # Old (deprecated):
    from pmd.store.document_metadata import DocumentMetadataRepository, StoredDocumentMetadata

    # New:
    from pmd.metadata import DocumentMetadataRepository, StoredDocumentMetadata
    # or
    from pmd.metadata.store import DocumentMetadataRepository
    from pmd.metadata.model import StoredDocumentMetadata
"""

import warnings

# Issue deprecation warning on import
warnings.warn(
    "Importing from 'pmd.store.document_metadata' is deprecated. "
    "Use 'pmd.metadata' or 'pmd.metadata.store' instead. "
    "This import path will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from new locations for backward compatibility
from pmd.metadata.store import DocumentMetadataRepository
from pmd.metadata.model import StoredDocumentMetadata

__all__ = [
    "DocumentMetadataRepository",
    "StoredDocumentMetadata",
]
