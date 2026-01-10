"""In-memory repository fakes for testing.

These implementations allow testing services without a real database.
Each fake implements the corresponding protocol from pmd.app.types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from pmd.core.types import Collection, DocumentResult, SearchResult


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
class InMemoryCollectionRepository:
    """In-memory collection repository for testing."""

    _collections: dict[str, Collection] = field(default_factory=dict)
    _id_counter: int = 0

    def list_all(self) -> list[Collection]:
        """Get all collections."""
        return list(self._collections.values())

    def get_by_name(self, name: str) -> Collection | None:
        """Get collection by name."""
        return self._collections.get(name)

    def get_by_id(self, collection_id: int) -> Collection | None:
        """Get collection by ID."""
        for c in self._collections.values():
            if c.id == collection_id:
                return c
        return None

    def create(
        self,
        name: str,
        pwd: str,
        glob_pattern: str = "**/*.md",
        source_type: str = "filesystem",
        source_config: dict[str, Any] | None = None,
    ) -> Collection:
        """Create a new collection."""
        self._id_counter += 1
        now = datetime.utcnow().isoformat()
        collection = Collection(
            id=self._id_counter,
            name=name,
            pwd=pwd,
            glob_pattern=glob_pattern,
            source_type=source_type,
            source_config=source_config or {},
            created_at=now,
            updated_at=now,
        )
        self._collections[name] = collection
        return collection

    def remove(self, collection_id: int) -> tuple[int, int]:
        """Remove a collection."""
        for name, c in list(self._collections.items()):
            if c.id == collection_id:
                del self._collections[name]
                return (0, 0)  # (docs_deleted, orphans_cleaned)
        return (0, 0)

    def rename(self, collection_id: int, new_name: str) -> None:
        """Rename a collection."""
        for name, c in list(self._collections.items()):
            if c.id == collection_id:
                del self._collections[name]
                self._collections[new_name] = Collection(
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
        collection_id: int,
        path: str,
        title: str,
        content: str,
    ) -> tuple[DocumentResult, bool]:
        """Add or update a document."""
        key = (collection_id, path)
        is_new = key not in self._documents

        self._id_counter += 1
        # Simple hash of content
        hash_value = str(hash(content))[:16]
        now = datetime.utcnow().isoformat()

        doc = DocumentResult(
            id=self._id_counter,
            collection_id=collection_id,
            path=path,
            title=title,
            hash=hash_value,
            created_at=now,
            updated_at=now,
        )
        self._documents[key] = doc
        self._content[hash_value] = content
        return (doc, is_new)

    def get(self, collection_id: int, path: str) -> DocumentResult | None:
        """Get document by path."""
        return self._documents.get((collection_id, path))

    def get_by_hash(self, hash_value: str) -> str | None:
        """Get content by hash."""
        return self._content.get(hash_value)

    def list_by_collection(
        self,
        collection_id: int,
        active_only: bool = True,
    ) -> list[DocumentResult]:
        """List documents in a collection."""
        return [
            doc for (cid, _), doc in self._documents.items()
            if cid == collection_id
        ]

    def delete(self, collection_id: int, path: str) -> bool:
        """Delete a document."""
        key = (collection_id, path)
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
        collection_id: int | None = None,
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

    def reindex_collection(self, collection_id: int) -> int:
        """Reindex collection (no-op for fake)."""
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
        collection_id: int | None = None,
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
