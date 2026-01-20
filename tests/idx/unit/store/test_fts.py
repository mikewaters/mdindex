"""Tests for idx.store.fts module."""

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

from idx.store.database import Base, create_engine_for_path
from idx.store.fts import FTSManager, FTSResult, create_fts_table, drop_fts_table
from idx.store.models import Dataset, Document


class TestFTSTable:
    """Tests for FTS table creation and management."""

    def test_create_fts_table(self, tmp_path: Path) -> None:
        """FTS5 table is created successfully."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)

        create_fts_table(engine)

        # Verify table exists
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'")
            )
            assert result.scalar() == "documents_fts"

    def test_create_fts_table_idempotent(self, tmp_path: Path) -> None:
        """Creating FTS table multiple times is safe."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)

        # Create twice
        create_fts_table(engine)
        create_fts_table(engine)

        # Should still exist
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'")
            )
            assert result.scalar() == "documents_fts"

    def test_drop_fts_table(self, tmp_path: Path) -> None:
        """FTS5 table can be dropped."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        create_fts_table(engine)

        drop_fts_table(engine)

        # Verify table is gone
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'")
            )
            assert result.scalar() is None


class TestFTSManager:
    """Tests for FTSManager class."""

    @pytest.fixture
    def db_session(self, tmp_path: Path):
        """Create a test database session with FTS table."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)
        create_fts_table(engine)

        factory = sessionmaker(bind=engine, expire_on_commit=False)
        session = factory()
        yield session
        session.close()

    @pytest.fixture
    def fts_manager(self, db_session) -> FTSManager:
        """Create an FTS manager."""
        return FTSManager(db_session)

    @pytest.fixture
    def sample_dataset(self, db_session) -> Dataset:
        """Create a sample dataset."""
        dataset = Dataset(
            name="test-dataset",
            uri="dataset:test-dataset",
            source_type="directory",
            source_path="/test/path",
        )
        db_session.add(dataset)
        db_session.flush()
        return dataset

    def test_upsert_document(self, fts_manager: FTSManager, db_session) -> None:
        """Document can be indexed."""
        fts_manager.upsert(doc_id=1, path="test.md", body="Hello world")
        db_session.flush()

        # Verify indexed
        assert fts_manager.count() == 1

    def test_upsert_updates_existing(self, fts_manager: FTSManager, db_session) -> None:
        """Upsert replaces existing document."""
        fts_manager.upsert(doc_id=1, path="test.md", body="Original content")
        db_session.flush()

        fts_manager.upsert(doc_id=1, path="test.md", body="Updated content")
        db_session.flush()

        # Should still be only one document
        assert fts_manager.count() == 1

    def test_delete_document(self, fts_manager: FTSManager, db_session) -> None:
        """Document can be deleted from index."""
        fts_manager.upsert(doc_id=1, path="test.md", body="Delete me")
        db_session.flush()
        assert fts_manager.count() == 1

        fts_manager.delete(doc_id=1)
        db_session.flush()

        assert fts_manager.count() == 0

    def test_delete_many(self, fts_manager: FTSManager, db_session) -> None:
        """Multiple documents can be deleted."""
        fts_manager.upsert(doc_id=1, path="a.md", body="Content A")
        fts_manager.upsert(doc_id=2, path="b.md", body="Content B")
        fts_manager.upsert(doc_id=3, path="c.md", body="Content C")
        db_session.flush()
        assert fts_manager.count() == 3

        deleted = fts_manager.delete_many([1, 2])
        db_session.flush()

        assert deleted == 2
        assert fts_manager.count() == 1

    def test_delete_many_empty_list(self, fts_manager: FTSManager) -> None:
        """Deleting empty list returns 0."""
        result = fts_manager.delete_many([])
        assert result == 0

    def test_search_basic(
        self, fts_manager: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """Basic search returns matching documents."""
        # Create actual documents to join with
        doc1 = Document(
            dataset_id=sample_dataset.id,
            path="hello.md",
            content_hash="h1",
            body="Hello world this is a test",
        )
        doc2 = Document(
            dataset_id=sample_dataset.id,
            path="goodbye.md",
            content_hash="h2",
            body="Goodbye world",
        )
        db_session.add_all([doc1, doc2])
        db_session.flush()

        # Index them
        fts_manager.upsert(doc1.id, doc1.path, doc1.body)
        fts_manager.upsert(doc2.id, doc2.path, doc2.body)
        db_session.flush()

        # Search for "hello"
        results = fts_manager.search("hello")

        assert len(results) == 1
        assert results[0].path == "hello.md"

    def test_search_returns_fts_result(
        self, fts_manager: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """Search results are FTSResult objects."""
        doc = Document(
            dataset_id=sample_dataset.id,
            path="test.md",
            content_hash="h",
            body="Testing search functionality",
        )
        db_session.add(doc)
        db_session.flush()

        fts_manager.upsert(doc.id, doc.path, doc.body)
        db_session.flush()

        results = fts_manager.search("search")

        assert len(results) == 1
        result = results[0]
        assert isinstance(result, FTSResult)
        assert result.doc_id == doc.id
        assert result.path == "test.md"
        assert result.rank is not None
        # Note: snippet may be None with contentless FTS

    def test_search_with_dataset_filter(
        self, fts_manager: FTSManager, db_session
    ) -> None:
        """Search can be filtered by dataset."""
        # Create two datasets
        ds1 = Dataset(
            name="ds1", uri="dataset:ds1", source_type="dir", source_path="/p1"
        )
        ds2 = Dataset(
            name="ds2", uri="dataset:ds2", source_type="dir", source_path="/p2"
        )
        db_session.add_all([ds1, ds2])
        db_session.flush()

        # Create documents in each
        doc1 = Document(
            dataset_id=ds1.id, path="doc1.md", content_hash="h1", body="Apple pie recipe"
        )
        doc2 = Document(
            dataset_id=ds2.id, path="doc2.md", content_hash="h2", body="Apple cider recipe"
        )
        db_session.add_all([doc1, doc2])
        db_session.flush()

        fts_manager.upsert(doc1.id, doc1.path, doc1.body)
        fts_manager.upsert(doc2.id, doc2.path, doc2.body)
        db_session.flush()

        # Search in ds1 only
        results = fts_manager.search("apple", dataset_filter=ds1.id)

        assert len(results) == 1
        assert results[0].path == "doc1.md"

    def test_search_excludes_inactive(
        self, fts_manager: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """Search excludes inactive documents."""
        doc1 = Document(
            dataset_id=sample_dataset.id,
            path="active.md",
            content_hash="h1",
            body="Active document content",
            active=True,
        )
        doc2 = Document(
            dataset_id=sample_dataset.id,
            path="inactive.md",
            content_hash="h2",
            body="Inactive document content",
            active=False,
        )
        db_session.add_all([doc1, doc2])
        db_session.flush()

        fts_manager.upsert(doc1.id, doc1.path, doc1.body)
        fts_manager.upsert(doc2.id, doc2.path, doc2.body)
        db_session.flush()

        # Search should only find active document
        results = fts_manager.search("document")

        assert len(results) == 1
        assert results[0].path == "active.md"

    def test_search_with_scores(
        self, fts_manager: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """search_with_scores returns normalized scores."""
        doc1 = Document(
            dataset_id=sample_dataset.id,
            path="doc1.md",
            content_hash="h1",
            body="Python programming tutorial",
        )
        doc2 = Document(
            dataset_id=sample_dataset.id,
            path="doc2.md",
            content_hash="h2",
            body="Python is great for programming and Python is easy",
        )
        db_session.add_all([doc1, doc2])
        db_session.flush()

        fts_manager.upsert(doc1.id, doc1.path, doc1.body)
        fts_manager.upsert(doc2.id, doc2.path, doc2.body)
        db_session.flush()

        results = fts_manager.search_with_scores("python")

        assert len(results) == 2
        # Scores should be between 0 and 1
        for doc_id, path, score in results:
            assert 0 <= score <= 1

    def test_count(self, fts_manager: FTSManager, db_session) -> None:
        """Count returns number of indexed documents."""
        assert fts_manager.count() == 0

        fts_manager.upsert(1, "a.md", "A")
        fts_manager.upsert(2, "b.md", "B")
        db_session.flush()

        assert fts_manager.count() == 2

    def test_ensure_table_exists(self, tmp_path: Path) -> None:
        """ensure_table_exists creates table if needed."""
        db_path = tmp_path / "fresh.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)

        factory = sessionmaker(bind=engine, expire_on_commit=False)
        session = factory()

        manager = FTSManager(session)
        manager.ensure_table_exists()

        # Verify table exists
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name='documents_fts'")
            )
            assert result.scalar() == "documents_fts"

        session.close()
