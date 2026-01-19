"""Dataset orchestration layer for PMD.

This package provides high-level orchestration for managing datasets:
- Syncing resources from sources
- Materializing documents from resources
- Indexing documents for search

Classes:
    SyncResult: Result of sync_resources() operation.
    MaterializeResult: Result of materialize_documents() operation.
    DatasetIndexResult: Result of index() operation.
    Dataset: Main orchestration class (to be implemented).

Example:
    from pmd.datasets import Dataset, SyncResult

    dataset = Dataset(collection, facade, cacher)
    result = await dataset.sync_resources()
    print(f"Added {result.added} resources")
"""

from .dataset import Dataset
from .results import DatasetIndexResult, MaterializeResult, SyncResult

__all__ = [
    "Dataset",
    "DatasetIndexResult",
    "MaterializeResult",
    "SyncResult",
]
