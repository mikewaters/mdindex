"""Data access layer modules for PMD.

This package contains data access classes that wrap repositories and provide
unified interfaces for specific use cases. These classes take only Database
in their constructors and create repositories internally.

Classes:
    IndexingData: Data access for indexing operations.
    LoadingData: Data access for document loading operations.
    SearchData: Data access for search operations.
    StatusData: Data access for status reporting operations (mostly read-only).
"""

from pmd.data.indexing import IndexingData
from pmd.data.loading import LoadingData
from pmd.data.search import SearchData
from pmd.data.status import StatusData

__all__ = ["IndexingData", "LoadingData", "SearchData", "StatusData"]
