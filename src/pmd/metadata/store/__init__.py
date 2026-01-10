"""Document metadata persistence layer.

This subpackage provides storage for extracted document metadata
including tags, attributes, and the profile used for extraction.

Repository:
- DocumentMetadataRepository: CRUD operations for document metadata

The repository uses two tables:
- document_metadata: Main metadata storage (profile, tags JSON, attributes)
- document_tags: Junction table for fast tag-based lookups
"""

from pmd.metadata.store.repository import DocumentMetadataRepository

__all__ = [
    "DocumentMetadataRepository",
]
