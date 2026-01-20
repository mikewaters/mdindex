"""Tests for idx.store.cleanup module."""

from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from idx.store.cleanup import IndexCleanup
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import FTSManager, create_fts_table
from idx.store.models import Dataset, Document


class TestIndexCleanup:
    """Tests for IndexCleanup class."""

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
    def cleanup(self, db_session) -> IndexCleanup:
        """Create an IndexCleanup manager."""
        return IndexCleanup(db_session)

    @pytest.fixture
    def fts(self, db_session) -> FTSManager:
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

    def test_cleanup_fts_for_document(
        self, cleanup: IndexCleanup, fts: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """FTS entry is removed for a single document."""
        doc = Document(
            dataset_id=sample_dataset.id,
            path="test.md",
            content_hash="hash",
            body="Test content",
        )
        db_session.add(doc)
        db_session.flush()

        # Index document
        fts.upsert(doc.id, doc.path, doc.body)
        db_session.flush()
        assert fts.count() == 1

        # Clean up
        cleanup.cleanup_fts_for_document(doc.id)
        db_session.flush()

        assert fts.count() == 0

    def test_cleanup_fts_for_documents(
        self, cleanup: IndexCleanup, fts: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """FTS entries are removed for multiple documents."""
        doc1 = Document(
            dataset_id=sample_dataset.id,
            path="doc1.md",
            content_hash="h1",
            body="Content 1",
        )
        doc2 = Document(
            dataset_id=sample_dataset.id,
            path="doc2.md",
            content_hash="h2",
            body="Content 2",
        )
        doc3 = Document(
            dataset_id=sample_dataset.id,
            path="doc3.md",
            content_hash="h3",
            body="Content 3",
        )
        db_session.add_all([doc1, doc2, doc3])
        db_session.flush()

        # Index all
        for doc in [doc1, doc2, doc3]:
            fts.upsert(doc.id, doc.path, doc.body)
        db_session.flush()
        assert fts.count() == 3

        # Clean up two
        count = cleanup.cleanup_fts_for_documents([doc1.id, doc2.id])
        db_session.flush()

        assert count == 2
        assert fts.count() == 1

    def test_cleanup_fts_for_inactive(
        self, cleanup: IndexCleanup, fts: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """FTS entries are removed for inactive documents."""
        active_doc = Document(
            dataset_id=sample_dataset.id,
            path="active.md",
            content_hash="h1",
            body="Active content",
            active=True,
        )
        inactive_doc = Document(
            dataset_id=sample_dataset.id,
            path="inactive.md",
            content_hash="h2",
            body="Inactive content",
            active=False,
        )
        db_session.add_all([active_doc, inactive_doc])
        db_session.flush()

        # Index both
        fts.upsert(active_doc.id, active_doc.path, active_doc.body)
        fts.upsert(inactive_doc.id, inactive_doc.path, inactive_doc.body)
        db_session.flush()
        assert fts.count() == 2

        # Clean up inactive only
        count = cleanup.cleanup_fts_for_inactive()
        db_session.flush()

        assert count == 1
        assert fts.count() == 1

    def test_cleanup_fts_for_inactive_with_dataset_filter(
        self, cleanup: IndexCleanup, fts: FTSManager, db_session
    ) -> None:
        """FTS cleanup can be limited to a specific dataset."""
        ds1 = Dataset(
            name="ds1", uri="dataset:ds1", source_type="dir", source_path="/p1"
        )
        ds2 = Dataset(
            name="ds2", uri="dataset:ds2", source_type="dir", source_path="/p2"
        )
        db_session.add_all([ds1, ds2])
        db_session.flush()

        # Create inactive documents in both datasets
        doc1 = Document(
            dataset_id=ds1.id,
            path="inactive1.md",
            content_hash="h1",
            body="Content 1",
            active=False,
        )
        doc2 = Document(
            dataset_id=ds2.id,
            path="inactive2.md",
            content_hash="h2",
            body="Content 2",
            active=False,
        )
        db_session.add_all([doc1, doc2])
        db_session.flush()

        fts.upsert(doc1.id, doc1.path, doc1.body)
        fts.upsert(doc2.id, doc2.path, doc2.body)
        db_session.flush()
        assert fts.count() == 2

        # Clean up only ds1
        count = cleanup.cleanup_fts_for_inactive(dataset_id=ds1.id)
        db_session.flush()

        assert count == 1
        assert fts.count() == 1

    def test_cleanup_fts_for_dataset(
        self, cleanup: IndexCleanup, fts: FTSManager, db_session, sample_dataset: Dataset
    ) -> None:
        """All FTS entries for a dataset are removed."""
        doc1 = Document(
            dataset_id=sample_dataset.id,
            path="doc1.md",
            content_hash="h1",
            body="Content 1",
        )
        doc2 = Document(
            dataset_id=sample_dataset.id,
            path="doc2.md",
            content_hash="h2",
            body="Content 2",
        )
        db_session.add_all([doc1, doc2])
        db_session.flush()

        fts.upsert(doc1.id, doc1.path, doc1.body)
        fts.upsert(doc2.id, doc2.path, doc2.body)
        db_session.flush()
        assert fts.count() == 2

        # Clean up entire dataset
        count = cleanup.cleanup_fts_for_dataset(sample_dataset.id)
        db_session.flush()

        assert count == 2
        assert fts.count() == 0

    def test_cleanup_empty_list(self, cleanup: IndexCleanup) -> None:
        """Cleaning up empty list returns 0."""
        count = cleanup.cleanup_fts_for_documents([])
        assert count == 0

    def test_vector_cleanup_placeholder(
        self, cleanup: IndexCleanup, sample_dataset: Dataset
    ) -> None:
        """Vector cleanup methods exist (placeholders)."""
        # These are placeholders - just verify they don't crash
        count = cleanup.cleanup_vectors_for_document(doc_id=1, content_hash="hash")
        assert count == 0

        count = cleanup.cleanup_vectors_for_inactive()
        assert count == 0
