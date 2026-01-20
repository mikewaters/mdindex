"""Tests for idx.store.models module."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import text

from idx.store.database import Base, create_engine_for_path, get_session_factory
from idx.store.models import Dataset, Document


class TestDataset:
    """Tests for Dataset model."""

    @pytest.fixture
    def db_session(self, tmp_path: Path):
        """Create a test database session."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)

        # Create session
        factory = get_session_factory.__wrapped__(engine)
        session = factory()
        yield session
        session.close()

    def test_create_dataset(self, db_session) -> None:
        """Dataset can be created with required fields."""
        dataset = Dataset(
            name="test-vault",
            uri="dataset:test-vault",
            source_type="obsidian",
            source_path="/path/to/vault",
        )
        db_session.add(dataset)
        db_session.commit()

        assert dataset.id is not None
        assert dataset.name == "test-vault"
        assert dataset.uri == "dataset:test-vault"
        assert dataset.source_type == "obsidian"
        assert dataset.source_path == "/path/to/vault"
        assert dataset.created_at is not None
        assert dataset.updated_at is not None

    def test_dataset_name_is_unique(self, db_session) -> None:
        """Dataset names must be unique."""
        dataset1 = Dataset(
            name="unique-name",
            uri="dataset:unique-name",
            source_type="directory",
            source_path="/path1",
        )
        db_session.add(dataset1)
        db_session.commit()

        dataset2 = Dataset(
            name="unique-name",
            uri="dataset:unique-name-2",
            source_type="directory",
            source_path="/path2",
        )
        db_session.add(dataset2)

        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_dataset_uri_is_unique(self, db_session) -> None:
        """Dataset URIs must be unique."""
        dataset1 = Dataset(
            name="name1",
            uri="dataset:shared-uri",
            source_type="directory",
            source_path="/path1",
        )
        db_session.add(dataset1)
        db_session.commit()

        dataset2 = Dataset(
            name="name2",
            uri="dataset:shared-uri",
            source_type="directory",
            source_path="/path2",
        )
        db_session.add(dataset2)

        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_dataset_repr(self, db_session) -> None:
        """Dataset has useful repr."""
        dataset = Dataset(
            name="repr-test",
            uri="dataset:repr-test",
            source_type="obsidian",
            source_path="/path",
        )
        db_session.add(dataset)
        db_session.commit()

        repr_str = repr(dataset)
        assert "repr-test" in repr_str
        assert "obsidian" in repr_str


class TestDocument:
    """Tests for Document model."""

    @pytest.fixture
    def db_session(self, tmp_path: Path):
        """Create a test database session."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)

        factory = get_session_factory.__wrapped__(engine)
        session = factory()
        yield session
        session.close()

    @pytest.fixture
    def dataset(self, db_session) -> Dataset:
        """Create a test dataset."""
        ds = Dataset(
            name="test-dataset",
            uri="dataset:test-dataset",
            source_type="directory",
            source_path="/test/path",
        )
        db_session.add(ds)
        db_session.commit()
        return ds

    def test_create_document(self, db_session, dataset: Dataset) -> None:
        """Document can be created with required fields."""
        doc = Document(
            dataset_id=dataset.id,
            path="notes/test.md",
            content_hash="abc123def456",
            body="# Test Document\n\nContent here.",
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.id is not None
        assert doc.dataset_id == dataset.id
        assert doc.path == "notes/test.md"
        assert doc.content_hash == "abc123def456"
        assert doc.body == "# Test Document\n\nContent here."
        assert doc.active is True
        assert doc.created_at is not None
        assert doc.updated_at is not None

    def test_document_default_active_true(self, db_session, dataset: Dataset) -> None:
        """Document active defaults to True."""
        doc = Document(
            dataset_id=dataset.id,
            path="test.md",
            content_hash="hash",
            body="content",
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.active is True

    def test_document_can_be_soft_deleted(self, db_session, dataset: Dataset) -> None:
        """Document can be soft-deleted by setting active=False."""
        doc = Document(
            dataset_id=dataset.id,
            path="deletable.md",
            content_hash="hash",
            body="content",
        )
        db_session.add(doc)
        db_session.commit()

        doc.active = False
        db_session.commit()

        assert doc.active is False

    def test_document_optional_etag(self, db_session, dataset: Dataset) -> None:
        """Document etag is optional."""
        doc = Document(
            dataset_id=dataset.id,
            path="test.md",
            content_hash="hash",
            body="content",
            etag="W/\"abc123\"",
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.etag == "W/\"abc123\""

    def test_document_optional_last_modified(self, db_session, dataset: Dataset) -> None:
        """Document last_modified is optional."""
        modified = datetime(2024, 1, 15, 10, 30, 0)
        doc = Document(
            dataset_id=dataset.id,
            path="test.md",
            content_hash="hash",
            body="content",
            last_modified=modified,
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.last_modified == modified

    def test_document_get_metadata(self, db_session, dataset: Dataset) -> None:
        """Document get_metadata parses JSON."""
        doc = Document(
            dataset_id=dataset.id,
            path="test.md",
            content_hash="hash",
            body="content",
            metadata_json='{"tags": ["foo", "bar"], "author": "test"}',
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.get_metadata() == {"tags": ["foo", "bar"], "author": "test"}

    def test_document_set_metadata(self, db_session, dataset: Dataset) -> None:
        """Document set_metadata serializes to JSON."""
        doc = Document(
            dataset_id=dataset.id,
            path="test.md",
            content_hash="hash",
            body="content",
        )
        doc.set_metadata({"key": "value", "number": 42})
        db_session.add(doc)
        db_session.commit()

        assert json.loads(doc.metadata_json) == {"key": "value", "number": 42}

    def test_document_get_metadata_empty_when_null(self, db_session, dataset: Dataset) -> None:
        """Document get_metadata returns empty dict when null."""
        doc = Document(
            dataset_id=dataset.id,
            path="test.md",
            content_hash="hash",
            body="content",
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.get_metadata() == {}

    def test_document_unique_path_per_dataset(self, db_session, dataset: Dataset) -> None:
        """Document paths must be unique within a dataset."""
        doc1 = Document(
            dataset_id=dataset.id,
            path="same/path.md",
            content_hash="hash1",
            body="content1",
        )
        db_session.add(doc1)
        db_session.commit()

        doc2 = Document(
            dataset_id=dataset.id,
            path="same/path.md",
            content_hash="hash2",
            body="content2",
        )
        db_session.add(doc2)

        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_document_repr(self, db_session, dataset: Dataset) -> None:
        """Document has useful repr."""
        doc = Document(
            dataset_id=dataset.id,
            path="repr/test.md",
            content_hash="hash",
            body="content",
        )
        db_session.add(doc)
        db_session.commit()

        repr_str = repr(doc)
        assert "repr/test.md" in repr_str
        assert "active=True" in repr_str


class TestDatasetDocumentRelationship:
    """Tests for Dataset-Document relationship."""

    @pytest.fixture
    def db_session(self, tmp_path: Path):
        """Create a test database session."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)

        factory = get_session_factory.__wrapped__(engine)
        session = factory()
        yield session
        session.close()

    def test_dataset_has_documents_relationship(self, db_session) -> None:
        """Dataset.documents returns associated documents."""
        dataset = Dataset(
            name="with-docs",
            uri="dataset:with-docs",
            source_type="directory",
            source_path="/path",
        )
        db_session.add(dataset)
        db_session.commit()

        doc1 = Document(
            dataset_id=dataset.id,
            path="doc1.md",
            content_hash="hash1",
            body="content1",
        )
        doc2 = Document(
            dataset_id=dataset.id,
            path="doc2.md",
            content_hash="hash2",
            body="content2",
        )
        db_session.add_all([doc1, doc2])
        db_session.commit()

        assert len(dataset.documents) == 2
        paths = {d.path for d in dataset.documents}
        assert paths == {"doc1.md", "doc2.md"}

    def test_document_has_dataset_relationship(self, db_session) -> None:
        """Document.dataset returns parent dataset."""
        dataset = Dataset(
            name="parent-ds",
            uri="dataset:parent-ds",
            source_type="directory",
            source_path="/path",
        )
        db_session.add(dataset)
        db_session.commit()

        doc = Document(
            dataset_id=dataset.id,
            path="child.md",
            content_hash="hash",
            body="content",
        )
        db_session.add(doc)
        db_session.commit()

        assert doc.dataset.name == "parent-ds"

    def test_cascade_delete_documents(self, db_session) -> None:
        """Deleting dataset cascades to documents."""
        dataset = Dataset(
            name="cascade-test",
            uri="dataset:cascade-test",
            source_type="directory",
            source_path="/path",
        )
        db_session.add(dataset)
        db_session.commit()
        dataset_id = dataset.id

        doc = Document(
            dataset_id=dataset.id,
            path="cascade.md",
            content_hash="hash",
            body="content",
        )
        db_session.add(doc)
        db_session.commit()
        doc_id = doc.id

        # Delete dataset
        db_session.delete(dataset)
        db_session.commit()

        # Document should be deleted too
        remaining = db_session.query(Document).filter_by(id=doc_id).first()
        assert remaining is None


# Helper for accessing wrapped function
def get_session_factory_for_engine(engine):
    """Create a session factory for a specific engine."""
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, expire_on_commit=False)


# Replace the fixture helper
get_session_factory.__wrapped__ = get_session_factory_for_engine
