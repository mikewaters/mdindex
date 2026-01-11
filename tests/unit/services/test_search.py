"""Tests for SearchService."""

import pytest
from pathlib import Path

from pmd.app import create_application
from pmd.core.config import Config
from pmd.core.types import SearchResult, SearchSource
from pmd.services.search import SearchService
from pmd.store.collections import SourceCollectionRepository
from pmd.store.documents import DocumentRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.search import FTS5SearchRepository
from pmd.store.database import Database


class TestSearchServiceFtsSearch:
    """Tests for SearchService.fts_search method."""

    def test_fts_search_empty_database(self, db: Database):
        """fts_search should return empty list for empty database."""
        source_collection_repo = SourceCollectionRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        results = service.fts_search("test query")

        assert results == []

    def test_fts_search_returns_list(self, db: Database):
        """fts_search should return a list."""
        source_collection_repo = SourceCollectionRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        results = service.fts_search("test", limit=5)

        assert isinstance(results, list)

    def test_fts_search_with_documents(self, db: Database, tmp_path: Path):
        """fts_search should find indexed documents."""
        source_collection_repo = SourceCollectionRepository(db)
        document_repo = DocumentRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        # Create collection and document
        collection = source_collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )
        document_repo.add_or_update(
            collection.id,
            "test.md",
            "Test Document",
            "# Machine Learning\n\nThis document is about machine learning algorithms.",
        )

        # Get doc_id for FTS indexing
        cursor = db.execute(
            "SELECT id FROM documents WHERE source_collection_id = ? AND path = ?",
            (collection.id, "test.md"),
        )
        doc_id = cursor.fetchone()["id"]

        # Index in FTS
        fts_repo.index_document(
            doc_id,
            "test.md",
            "# Machine Learning\n\nThis document is about machine learning algorithms.",
        )

        results = service.fts_search("machine learning")

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(r.source == SearchSource.FTS for r in results)

    def test_fts_search_respects_limit(self, db: Database, tmp_path: Path):
        """fts_search should respect the limit parameter."""
        source_collection_repo = SourceCollectionRepository(db)
        document_repo = DocumentRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        # Create collection and multiple documents
        collection = source_collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )

        for i in range(5):
            document_repo.add_or_update(
                collection.id,
                f"doc{i}.md",
                f"Document {i}",
                f"# Document {i}\n\nThis is test content number {i}.",
            )
            cursor = db.execute(
                "SELECT id FROM documents WHERE source_collection_id = ? AND path = ?",
                (collection.id, f"doc{i}.md"),
            )
            doc_id = cursor.fetchone()["id"]
            fts_repo.index_document(
                doc_id, f"doc{i}.md", f"# Document {i}\n\nThis is test content."
            )

        results = service.fts_search("test", limit=2)

        assert len(results) <= 2

    def test_fts_search_filters_by_collection(self, db: Database, tmp_path: Path):
        """fts_search should filter by collection name."""
        source_collection_repo = SourceCollectionRepository(db)
        document_repo = DocumentRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        # Create two collections
        (tmp_path / "coll1").mkdir()
        (tmp_path / "coll2").mkdir()

        coll1 = source_collection_repo.create(
            "collection1", str(tmp_path / "coll1"), "**/*.md"
        )
        coll2 = source_collection_repo.create(
            "collection2", str(tmp_path / "coll2"), "**/*.md"
        )

        # Add document only to collection1
        document_repo.add_or_update(
            coll1.id, "doc.md", "Unique Doc", "# Unique content here"
        )
        cursor = db.execute(
            "SELECT id FROM documents WHERE source_collection_id = ? AND path = ?",
            (coll1.id, "doc.md"),
        )
        doc_id = cursor.fetchone()["id"]
        fts_repo.index_document(
            doc_id, "doc.md", "# Unique content here"
        )

        # Search in collection2 should find nothing
        results = service.fts_search(
            "unique", collection_name="collection2"
        )

        assert len(results) == 0


class TestSearchServiceResolveCollectionId:
    """Tests for SearchService._resolve_collection_id method."""

    def test_resolve_collection_id_none(self, db: Database):
        """_resolve_collection_id should return None for None input."""
        source_collection_repo = SourceCollectionRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        result = service._resolve_collection_id(None)

        assert result is None

    def test_resolve_collection_id_found(self, db: Database, tmp_path: Path):
        """_resolve_collection_id should return ID for existing collection."""
        source_collection_repo = SourceCollectionRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        collection = source_collection_repo.create(
            "test", str(tmp_path), "**/*.md"
        )

        result = service._resolve_collection_id("test")

        assert result == collection.id

    def test_resolve_collection_id_not_found(self, db: Database):
        """_resolve_collection_id should return None for unknown collection."""
        source_collection_repo = SourceCollectionRepository(db)
        fts_repo = FTS5SearchRepository(db)
        embedding_repo = EmbeddingRepository(db)

        service = SearchService(
            db=db,
            fts_repo=fts_repo,
            source_collection_repo=source_collection_repo,
            embedding_repo=embedding_repo,
        )

        result = service._resolve_collection_id("nonexistent")

        assert result is None


class TestSearchServiceVectorSearch:
    """Tests for SearchService.vector_search method."""

    @pytest.mark.asyncio
    async def test_vector_search_raises_without_vec(self, config: Config):
        """vector_search should raise if vec not available."""
        async with await create_application(config) as app:
            # Skip if vec is actually available
            if app.vec_available:
                pytest.skip("sqlite-vec is available")

            with pytest.raises(RuntimeError, match="Vector search not available"):
                await app.search.vector_search("test query")


class TestSearchServiceHybridSearch:
    """Tests for SearchService.hybrid_search method."""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_list(self, config: Config, tmp_path: Path):
        """hybrid_search should return a list."""
        async with await create_application(config) as app:
            # Create a collection with some documents
            collection = app.source_collection_repo.create(
                "test", str(tmp_path), "**/*.md"
            )
            app.document_repo.add_or_update(
                collection.id, "doc.md", "Test Doc", "# Test content"
            )
            cursor = app.db.execute(
                "SELECT id FROM documents WHERE source_collection_id = ? AND path = ?",
                (collection.id, "doc.md"),
            )
            doc_id = cursor.fetchone()["id"]
            from pmd.store.search import FTS5SearchRepository
            fts_repo = FTS5SearchRepository(app.db)
            fts_repo.index_document(doc_id, "doc.md", "# Test content")

            results = await app.search.hybrid_search("test", limit=5)

            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_search_respects_limit(self, config: Config, tmp_path: Path):
        """hybrid_search should respect limit parameter."""
        async with await create_application(config) as app:
            # Create collection and multiple documents
            collection = app.source_collection_repo.create(
                "test", str(tmp_path), "**/*.md"
            )

            from pmd.store.search import FTS5SearchRepository
            fts_repo = FTS5SearchRepository(app.db)

            for i in range(5):
                app.document_repo.add_or_update(
                    collection.id,
                    f"doc{i}.md",
                    f"Document {i}",
                    f"# Document\n\nSearchable content here.",
                )
                cursor = app.db.execute(
                    "SELECT id FROM documents WHERE source_collection_id = ? AND path = ?",
                    (collection.id, f"doc{i}.md"),
                )
                doc_id = cursor.fetchone()["id"]
                fts_repo.index_document(
                    doc_id, f"doc{i}.md", "# Document\n\nSearchable content here."
                )

            results = await app.search.hybrid_search("searchable", limit=2)

            assert len(results) <= 2
