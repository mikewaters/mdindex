"""In-memory repository fakes for testing.

These implementations allow testing services without a real database.
Each fake implements the corresponding protocol from pmd.app.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from pmd.core.types import SourceCollection, DocumentResult, SearchResult


@dataclass
class InMemoryDatabase:
    """In-memory database fake for testing.

    Provides minimal database interface for services that need direct SQL.
    Most services should use repository fakes instead.
    """

    path: Path = field(default_factory=lambda: Path(":memory:"))
    _vec_available: bool = True
    _rows: list[dict] = field(default_factory=list)

    @property
    def vec_available(self) -> bool:
        """Check if vector storage is available."""
        return self._vec_available

    def connect(self) -> None:
        """Initialize database (no-op for fake)."""
        pass

    def close(self) -> None:
        """Close database (no-op for fake)."""
        pass

    def execute(self, sql: str, params: tuple = ()) -> "InMemoryCursor":
        """Execute SQL and return cursor.

        Note: This fake returns empty results for all queries.
        Override _rows or subclass for specific test needs.
        """
        return InMemoryCursor(self._rows)

    def transaction(self) -> Iterator["InMemoryCursor"]:
        """Context manager for transactions (no-op for fake)."""
        yield InMemoryCursor(self._rows)


@dataclass
class InMemoryCursor:
    """Fake cursor for InMemoryDatabase."""

    _rows: list[dict] = field(default_factory=list)
    _index: int = 0

    def fetchone(self) -> dict | None:
        """Fetch next row."""
        if self._index < len(self._rows):
            row = self._rows[self._index]
            self._index += 1
            return row
        return {"count": 0}  # Default for COUNT queries

    def fetchall(self) -> list[dict]:
        """Fetch all remaining rows."""
        rows = self._rows[self._index:]
        self._index = len(self._rows)
        return rows


@dataclass
class InMemorySourceCollectionRepository:
    """In-memory source collection repository for testing."""

    _source_collections: dict[str, SourceCollection] = field(default_factory=dict)
    _id_counter: int = 0

    def list_all(self) -> list[SourceCollection]:
        """Get all source collections."""
        return list(self._source_collections.values())

    def get_by_name(self, name: str) -> SourceCollection | None:
        """Get source collection by name."""
        return self._source_collections.get(name)

    def get_by_id(self, source_collection_id: int) -> SourceCollection | None:
        """Get source collection by ID."""
        for c in self._source_collections.values():
            if c.id == source_collection_id:
                return c
        return None

    def create(
        self,
        name: str,
        pwd: str,
        glob_pattern: str = "**/*.md",
        source_type: str = "filesystem",
        source_config: dict[str, Any] | None = None,
    ) -> SourceCollection:
        """Create a new source collection."""
        self._id_counter += 1
        now = datetime.utcnow().isoformat()
        source_collection = SourceCollection(
            id=self._id_counter,
            name=name,
            pwd=pwd,
            glob_pattern=glob_pattern,
            source_type=source_type,
            source_config=source_config or {},
            created_at=now,
            updated_at=now,
        )
        self._source_collections[name] = source_collection
        return source_collection

    def remove(self, source_collection_id: int) -> tuple[int, int]:
        """Remove a source collection."""
        for name, c in list(self._source_collections.items()):
            if c.id == source_collection_id:
                del self._source_collections[name]
                return (0, 0)  # (docs_deleted, orphans_cleaned)
        return (0, 0)

    def rename(self, source_collection_id: int, new_name: str) -> None:
        """Rename a source collection."""
        for name, c in list(self._source_collections.items()):
            if c.id == source_collection_id:
                del self._source_collections[name]
                self._source_collections[new_name] = SourceCollection(
                    id=c.id,
                    name=new_name,
                    pwd=c.pwd,
                    glob_pattern=c.glob_pattern,
                    source_type=c.source_type,
                    source_config=c.source_config,
                    created_at=c.created_at,
                    updated_at=datetime.utcnow().isoformat(),
                )
                return


@dataclass
class InMemoryDocumentRepository:
    """In-memory document repository for testing."""

    _documents: dict[tuple[int, str], DocumentResult] = field(default_factory=dict)
    _content: dict[str, str] = field(default_factory=dict)
    _id_counter: int = 0

    def add_or_update(
        self,
        source_collection_id: int,
        path: str,
        title: str,
        content: str,
    ) -> tuple[DocumentResult, bool]:
        """Add or update a document."""
        key = (source_collection_id, path)
        is_new = key not in self._documents

        self._id_counter += 1
        # Simple hash of content
        hash_value = str(hash(content))[:16]
        now = datetime.utcnow().isoformat()

        doc = DocumentResult(
            id=self._id_counter,
            source_collection_id=source_collection_id,
            path=path,
            title=title,
            hash=hash_value,
            created_at=now,
            updated_at=now,
        )
        self._documents[key] = doc
        self._content[hash_value] = content
        return (doc, is_new)

    def get(self, source_collection_id: int, path: str) -> DocumentResult | None:
        """Get document by path."""
        return self._documents.get((source_collection_id, path))

    def get_by_hash(self, hash_value: str) -> str | None:
        """Get content by hash."""
        return self._content.get(hash_value)

    def list_by_collection(
        self,
        source_collection_id: int,
        active_only: bool = True,
    ) -> list[DocumentResult]:
        """List documents in a source collection."""
        return [
            doc for (cid, _), doc in self._documents.items()
            if cid == source_collection_id
        ]

    def delete(self, source_collection_id: int, path: str) -> bool:
        """Delete a document."""
        key = (source_collection_id, path)
        if key in self._documents:
            del self._documents[key]
            return True
        return False


@dataclass
class InMemoryFTSRepository:
    """In-memory FTS repository for testing."""

    _index: dict[int, tuple[str, str]] = field(default_factory=dict)  # doc_id -> (path, body)
    _results: list[SearchResult] = field(default_factory=list)

    def search(
        self,
        query: str,
        limit: int = 5,
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Execute FTS search."""
        # Return pre-configured results or empty
        results = [r for r in self._results if r.score >= min_score]
        return results[:limit]

    def index_document(self, doc_id: int, path: str, body: str) -> None:
        """Index a document."""
        self._index[doc_id] = (path, body)

    def remove_from_index(self, doc_id: int) -> None:
        """Remove document from index."""
        self._index.pop(doc_id, None)

    def reindex_collection(self, source_collection_id: int) -> int:
        """Reindex source collection (no-op for fake)."""
        return 0

    # Test helpers
    def add_result(self, result: SearchResult) -> None:
        """Add a pre-configured search result."""
        self._results.append(result)

    def set_results(self, results: list[SearchResult]) -> None:
        """Set all search results."""
        self._results = results


@dataclass
class InMemoryEmbeddingRepository:
    """In-memory embedding repository for testing."""

    _embeddings: dict[str, list[tuple[int, int, list[float], str]]] = field(
        default_factory=dict
    )  # hash -> [(seq, pos, embedding, model)]
    _search_results: list[SearchResult] = field(default_factory=list)

    def store_embedding(
        self,
        hash_value: str,
        seq: int,
        pos: int,
        embedding: list[float],
        model: str,
    ) -> None:
        """Store embedding vector."""
        if hash_value not in self._embeddings:
            self._embeddings[hash_value] = []
        self._embeddings[hash_value].append((seq, pos, embedding, model))

    def has_embeddings(self, hash_value: str, model: str | None = None) -> bool:
        """Check if content has embeddings."""
        if hash_value not in self._embeddings:
            return False
        if model is None:
            return True
        return any(m == model for _, _, _, m in self._embeddings[hash_value])

    def delete_embeddings(self, hash_value: str) -> int:
        """Delete embeddings for content."""
        if hash_value in self._embeddings:
            count = len(self._embeddings[hash_value])
            del self._embeddings[hash_value]
            return count
        return 0

    def search_vectors(
        self,
        query_embedding: list[float],
        limit: int = 5,
        source_collection_id: int | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search by vector similarity."""
        results = [r for r in self._search_results if r.score >= min_score]
        return results[:limit]

    # Test helpers
    def add_search_result(self, result: SearchResult) -> None:
        """Add a pre-configured search result."""
        self._search_results.append(result)

    def set_search_results(self, results: list[SearchResult]) -> None:
        """Set all search results."""
        self._search_results = results


@dataclass
class InMemoryLoadingService:
    """In-memory loading service fake for testing.

    Implements LoadingServiceProtocol for testing IndexingService
    without needing actual file I/O.
    """

    _documents: list = field(default_factory=list)
    _errors: list = field(default_factory=list)
    _enumerated_paths: set = field(default_factory=set)

    async def load_collection_eager(
        self,
        collection_name: str,
        source: Any = None,
        force: bool = False,
    ):
        """Return configured documents for testing."""
        from pmd.services.loading import EagerLoadResult

        return EagerLoadResult(
            documents=list(self._documents),
            enumerated_paths=set(self._enumerated_paths),
            errors=list(self._errors),
        )

    # Test helpers
    def add_document(self, doc) -> None:
        """Add a document to be returned by load_collection_eager."""
        self._documents.append(doc)

    def add_error(self, path: str, error: str) -> None:
        """Add an error to be returned by load_collection_eager."""
        self._errors.append((path, error))

    def add_enumerated_path(self, path: str) -> None:
        """Add an enumerated path."""
        self._enumerated_paths.add(path)

    def clear(self) -> None:
        """Clear all configured data."""
        self._documents.clear()
        self._errors.clear()
        self._enumerated_paths.clear()
