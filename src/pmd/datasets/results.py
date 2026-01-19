"""Result dataclasses for Dataset orchestration.

These dataclasses capture the results of Dataset operations:
- SyncResult: Results from sync_resources() operation
- MaterializeResult: Results from materialize_documents() operation
- DatasetIndexResult: Results from index() operation
"""

from dataclasses import dataclass, field


@dataclass
class SyncResult:
    """Result of a sync_resources() operation.

    Tracks how many resources were added, updated, unchanged, or failed
    during synchronization with the source.

    Attributes:
        added: Number of new resources discovered and added.
        updated: Number of existing resources that were updated (content changed).
        unchanged: Number of resources that were already up-to-date.
        failed: Number of resources that failed to sync.
        errors: List of (uri, error_message) tuples for failed resources.
    """

    added: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of resources processed."""
        return self.added + self.updated + self.unchanged + self.failed


@dataclass
class MaterializeResult:
    """Result of a materialize_documents() operation.

    Tracks how many documents were created, updated, or skipped
    when materializing resources into searchable documents.

    Attributes:
        created: Number of new documents created.
        updated: Number of existing documents that were updated.
        skipped: Number of resources skipped (already materialized, no change).
    """

    created: int = 0
    updated: int = 0
    skipped: int = 0

    @property
    def total(self) -> int:
        """Total number of resources processed."""
        return self.created + self.updated + self.skipped


@dataclass
class DatasetIndexResult:
    """Result of an index() operation.

    Tracks how many documents were indexed or failed during
    the indexing phase (FTS + optional embeddings).

    Attributes:
        indexed: Number of documents successfully indexed.
        failed: Number of documents that failed to index.
        errors: List of (uri, error_message) tuples for failed documents.
    """

    indexed: int = 0
    failed: int = 0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        """Total number of documents processed."""
        return self.indexed + self.failed
