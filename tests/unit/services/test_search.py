"""Tests for SearchService."""

import pytest
from pathlib import Path

from pmd.core.config import Config
from pmd.core.types import SearchResult, SearchSource
from pmd.services import ServiceContainer
from pmd.services.search import SearchService


class TestSearchServiceInit:
    """Tests for SearchService initialization."""

    def test_init_stores_container(self, connected_container: ServiceContainer):
        """SearchService should store container reference."""
        service = SearchService(connected_container)

        assert service._container is connected_container


class TestSearchServiceFtsSearch:
    """Tests for SearchService.fts_search method."""

    def test_fts_search_empty_database(self, connected_container: ServiceContainer):
        """fts_search should return empty list for empty database."""
        results = connected_container.search.fts_search("test query")

        assert results == []

    def test_fts_search_returns_list(self, connected_container: ServiceContainer):
        """fts_search should return a list."""
        results = connected_container.search.fts_search("test", limit=5)

        assert isinstance(results, list)

    def test_fts_search_with_documents(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """fts_search should find indexed documents."""
        # Create collection and document
        collection = connected_container.collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        connected_container.document_repo.add_or_update(
            collection.id,
            "test.md",
            "Test Document",
            "# Machine Learning\n\nThis document is about machine learning algorithms.",
        )

        # Get doc_id for FTS indexing
        cursor = connected_container.db.execute(
            "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
            (collection.id, "test.md"),
        )
        doc_id = cursor.fetchone()["id"]

        # Index in FTS
        connected_container.fts_repo.index_document(
            doc_id,
            "test.md",
            "# Machine Learning\n\nThis document is about machine learning algorithms.",
        )

        results = connected_container.search.fts_search("machine learning")

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == SearchSource.FTS for r in results)

    def test_fts_search_respects_limit(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """fts_search should respect the limit parameter."""
        # Create collection and multiple documents
        collection = connected_container.collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )

        for i in range(5):
            connected_container.document_repo.add_or_update(
                collection.id,
                f"doc{i}.md",
                f"Document {i}",
                f"# Document {i}\n\nThis is test content number {i}.",
            )
            cursor = connected_container.db.execute(
                "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
                (collection.id, f"doc{i}.md"),
            )
            doc_id = cursor.fetchone()["id"]
            connected_container.fts_repo.index_document(
                doc_id, f"doc{i}.md", f"# Document {i}\n\nThis is test content."
            )

        results = connected_container.search.fts_search("test", limit=2)

        assert len(results) <= 2

    def test_fts_search_filters_by_collection(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """fts_search should filter by collection name."""
        # Create two collections
        coll1 = connected_container.collection_repo.create(
            "collection1", str(tmp_path / "coll1"), "**/*.md"
        )
        (tmp_path / "coll1").mkdir()

        coll2 = connected_container.collection_repo.create(
            "collection2", str(tmp_path / "coll2"), "**/*.md"
        )
        (tmp_path / "coll2").mkdir()

        # Add document only to collection1
        connected_container.document_repo.add_or_update(
            coll1.id, "doc.md", "Unique Doc", "# Unique content here"
        )
        cursor = connected_container.db.execute(
            "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
            (coll1.id, "doc.md"),
        )
        doc_id = cursor.fetchone()["id"]
        connected_container.fts_repo.index_document(
            doc_id, "doc.md", "# Unique content here"
        )

        # Search in collection2 should find nothing
        results = connected_container.search.fts_search(
            "unique", collection_name="collection2"
        )

        assert len(results) == 0


class TestSearchServiceResolveCollectionId:
    """Tests for SearchService._resolve_collection_id method."""

    def test_resolve_collection_id_none(self, connected_container: ServiceContainer):
        """_resolve_collection_id should return None for None input."""
        result = connected_container.search._resolve_collection_id(None)

        assert result is None

    def test_resolve_collection_id_found(
        self, connected_container: ServiceContainer, tmp_path: Path
    ):
        """_resolve_collection_id should return ID for existing collection."""
        collection = connected_container.collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )

        result = connected_container.search._resolve_collection_id("test")

        assert result == collection.id

    def test_resolve_collection_id_not_found(
        self, connected_container: ServiceContainer
    ):
        """_resolve_collection_id should return None for unknown collection."""
        result = connected_container.search._resolve_collection_id("nonexistent")

        assert result is None


class TestSearchServiceVectorSearch:
    """Tests for SearchService.vector_search method."""

    @pytest.mark.asyncio
    async def test_vector_search_raises_without_vec(self, config: Config):
        """vector_search should raise if vec not available."""
        async with ServiceContainer(config) as services:
            # Skip if vec is actually available
            if services.vec_available:
                pytest.skip("sqlite-vec is available")

            with pytest.raises(RuntimeError, match="Vector search not available"):
                await services.search.vector_search("test query")


class TestSearchServiceHybridSearch:
    """Tests for SearchService.hybrid_search method."""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_list(self, config: Config, tmp_path: Path):
        """hybrid_search should return a list."""
        async with ServiceContainer(config) as services:
            # Create a collection with some documents
            collection = services.collection_repo.create(
                "test", str(tmp_path), "**/*.md"
            )
            services.document_repo.add_or_update(
                collection.id, "doc.md", "Test Doc", "# Test content"
            )
            cursor = services.db.execute(
                "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
                (collection.id, "doc.md"),
            )
            doc_id = cursor.fetchone()["id"]
            services.fts_repo.index_document(doc_id, "doc.md", "# Test content")

            results = await services.search.hybrid_search("test", limit=5)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_limit(self, config: Config, tmp_path: Path):
        """hybrid_search should respect limit parameter."""
        async with ServiceContainer(config) as services:
            # Create collection and multiple documents
            collection = services.collection_repo.create(
                "test", str(tmp_path), "**/*.md"
            )

            for i in range(5):
                services.document_repo.add_or_update(
                    collection.id,
                    f"doc{i}.md",
                    f"Document {i}",
                    f"# Document\n\nSearchable content here.",
                )
                cursor = services.db.execute(
                    "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
                    (collection.id, f"doc{i}.md"),
                )
                doc_id = cursor.fetchone()["id"]
                services.fts_repo.index_document(
                    doc_id, f"doc{i}.md", "# Document\n\nSearchable content here."
                )

            results = await services.search.hybrid_search("searchable", limit=2)

            assert len(results) <= 2
