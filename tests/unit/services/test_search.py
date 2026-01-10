"""Tests for SearchService."""

import pytest
from pathlib import Path

from pmd.core.config import Config
from pmd.core.types import SearchResult, SearchSource
from pmd.services import ServiceContainer
from pmd.services.search import SearchService


class TestSearchServiceInit:
    """Tests for SearchService initialization."""

    def test_init_with_explicit_deps(self, connected_container: ServiceContainer):
        """SearchService should work with explicit dependencies."""
        service = SearchService(
            db=connected_container.db,
            fts_repo=connected_container.fts_repo,
            collection_repo=connected_container.collection_repo,
            embedding_repo=connected_container.embedding_repo,
        )

        assert service._db is connected_container.db
        assert service._fts_repo is connected_container.fts_repo
        assert service._collection_repo is connected_container.collection_repo

    def test_from_container_factory(self, connected_container: ServiceContainer):
        """SearchService.from_container should create service with container deps."""
        service = SearchService.from_container(connected_container)

        assert service._db is connected_container.db
        assert service._fts_repo is connected_container.fts_repo
        assert service._collection_repo is connected_container.collection_repo


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

    @pytest.mark.asyncio
    async def test_vector_search_returns_list(
        self, config: Config, mock_embedding_generator
    ):
        """vector_search should return a list of SearchResult."""
        from unittest.mock import MagicMock

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services._embedding_generator = mock_embedding_generator
            services.embedding_repo.search_vectors = MagicMock(return_value=[])

            results = await services.search.vector_search("test query")

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_vector_search_calls_embed_query(
        self, config: Config, mock_embedding_generator
    ):
        """vector_search should call embed_query on the query."""
        from unittest.mock import MagicMock

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services._embedding_generator = mock_embedding_generator
            services.embedding_repo.search_vectors = MagicMock(return_value=[])

            await services.search.vector_search("my search query")

            mock_embedding_generator.embed_query.assert_called_once_with("my search query")

    @pytest.mark.asyncio
    async def test_vector_search_calls_search_vectors(
        self, config: Config, mock_embedding_generator
    ):
        """vector_search should call embedding_repo.search_vectors."""
        from unittest.mock import MagicMock

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services._embedding_generator = mock_embedding_generator
            mock_search = MagicMock(return_value=[])
            services.embedding_repo.search_vectors = mock_search

            await services.search.vector_search("query", limit=10, min_score=0.5)

            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args.kwargs["limit"] == 10
            assert call_args.kwargs["min_score"] == 0.5

    @pytest.mark.asyncio
    async def test_vector_search_returns_empty_on_embed_failure(
        self, config: Config
    ):
        """vector_search should return empty list if embedding fails."""
        from unittest.mock import AsyncMock, MagicMock

        async with ServiceContainer(config) as services:
            services._vec_available = True

            # Mock embedding generator that returns None (failure)
            mock_gen = AsyncMock()
            mock_gen.embed_query = AsyncMock(return_value=None)
            services._embedding_generator = mock_gen
            services.embedding_repo.search_vectors = MagicMock(return_value=[])

            results = await services.search.vector_search("query")

            assert results == []
            # search_vectors should NOT be called if embedding failed
            services.embedding_repo.search_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_vector_search_passes_collection_id(
        self, config: Config, tmp_path: Path, mock_embedding_generator
    ):
        """vector_search should pass collection_id when filtering."""
        from unittest.mock import MagicMock

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services._embedding_generator = mock_embedding_generator

            # Create a collection
            collection = services.collection_repo.create(
                "test", str(tmp_path), "**/*.md"
            )

            mock_search = MagicMock(return_value=[])
            services.embedding_repo.search_vectors = mock_search

            await services.search.vector_search("query", collection_name="test")

            mock_search.assert_called_once()
            call_args = mock_search.call_args
            assert call_args.kwargs["collection_id"] == collection.id

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(
        self, config: Config, mock_embedding_generator
    ):
        """vector_search should return results from search_vectors."""
        from unittest.mock import MagicMock

        # Create mock search result
        mock_result = SearchResult(
            filepath="doc.md",
            display_path="doc.md",
            title="Document",
            context=None,
            hash="abc123",
            collection_id=1,
            modified_at="2024-01-01",
            body_length=100,
            body=None,
            score=0.85,
            source=SearchSource.VECTOR,
            chunk_pos=0,
        )

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services._embedding_generator = mock_embedding_generator
            services.embedding_repo.search_vectors = MagicMock(return_value=[mock_result])

            results = await services.search.vector_search("query")

            assert len(results) == 1
            assert results[0].filepath == "doc.md"
            assert results[0].score == 0.85
            assert results[0].source == SearchSource.VECTOR


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
