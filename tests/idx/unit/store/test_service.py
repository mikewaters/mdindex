"""Tests for idx.store.service module."""

from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from idx.store.database import Base, create_engine_for_path
from idx.store.schemas import DatasetCreate, DocumentCreate, DocumentUpdate
from idx.store.dataset import (
    DatasetExistsError,
    DatasetNotFoundError,
    DatasetService,
    DocumentNotFoundError,
    normalize_dataset_name,
)


class TestNormalizeDatasetName:
    """Tests for normalize_dataset_name function."""

    @pytest.mark.parametrize(
        "input_name,expected",
        [
            ("my-vault", "my-vault"),
            ("My Vault", "my-vault"),
            ("MY_VAULT", "my-vault"),
            ("My Obsidian Vault", "my-obsidian-vault"),
            ("vault@2024", "vault2024"),
            ("  spaces  ", "spaces"),
            ("multiple---hyphens", "multiple-hyphens"),
            ("MixedCase123", "mixedcase123"),
        ],
    )
    def test_normalizes_correctly(self, input_name: str, expected: str) -> None:
        """Names are normalized to URI-acceptable format."""
        assert normalize_dataset_name(input_name) == expected


class TestDatasetService:
    """Tests for DatasetService."""

    @pytest.fixture
    def db_session(self, tmp_path: Path):
        """Create a test database session."""
        db_path = tmp_path / "test.db"
        engine = create_engine_for_path(db_path)
        Base.metadata.create_all(engine)

        factory = sessionmaker(bind=engine, expire_on_commit=False)
        session = factory()
        yield session
        session.close()

    @pytest.fixture
    def service(self, db_session) -> DatasetService:
        """Create a DatasetService with test session."""
        return DatasetService(session=db_session)

    # Dataset tests

    def test_create_dataset(self, service: DatasetService) -> None:
        """Dataset can be created."""
        data = DatasetCreate(
            name="test-vault",
            source_type="obsidian",
            source_path="/path/to/vault",
        )

        result = service.create_dataset(data)

        assert result.id is not None
        assert result.name == "test-vault"
        assert result.uri == "dataset:test-vault"
        assert result.source_type == "obsidian"
        assert result.source_path == "/path/to/vault"
        assert result.document_count == 0

    def test_create_dataset_normalizes_name(self, service: DatasetService) -> None:
        """Dataset name is normalized on creation."""
        data = DatasetCreate(
            name="My Fancy Vault",
            source_type="obsidian",
            source_path="/path",
        )

        result = service.create_dataset(data)

        assert result.name == "my-fancy-vault"
        assert result.uri == "dataset:my-fancy-vault"

    def test_create_dataset_duplicate_raises(self, service: DatasetService) -> None:
        """Creating duplicate dataset raises error."""
        data = DatasetCreate(
            name="unique-name",
            source_type="directory",
            source_path="/path",
        )
        service.create_dataset(data)

        with pytest.raises(DatasetExistsError) as exc_info:
            service.create_dataset(data)

        assert "unique-name" in str(exc_info.value)

    def test_get_dataset(self, service: DatasetService) -> None:
        """Dataset can be retrieved by ID."""
        data = DatasetCreate(
            name="get-test",
            source_type="directory",
            source_path="/path",
        )
        created = service.create_dataset(data)

        result = service.get_dataset(created.id)

        assert result.id == created.id
        assert result.name == "get-test"

    def test_get_dataset_not_found(self, service: DatasetService) -> None:
        """Getting non-existent dataset raises error."""
        with pytest.raises(DatasetNotFoundError):
            service.get_dataset(99999)

    def test_get_dataset_by_name(self, service: DatasetService) -> None:
        """Dataset can be retrieved by name."""
        data = DatasetCreate(
            name="by-name-test",
            source_type="directory",
            source_path="/path",
        )
        created = service.create_dataset(data)

        result = service.get_dataset_by_name("by-name-test")

        assert result.id == created.id

    def test_get_dataset_by_name_normalizes(self, service: DatasetService) -> None:
        """Dataset name is normalized when retrieving."""
        data = DatasetCreate(
            name="normalize-lookup",
            source_type="directory",
            source_path="/path",
        )
        service.create_dataset(data)

        result = service.get_dataset_by_name("Normalize Lookup")

        assert result.name == "normalize-lookup"

    def test_list_datasets(self, service: DatasetService) -> None:
        """All datasets can be listed."""
        service.create_dataset(
            DatasetCreate(name="ds1", source_type="dir", source_path="/p1")
        )
        service.create_dataset(
            DatasetCreate(name="ds2", source_type="dir", source_path="/p2")
        )

        results = service.list_datasets()

        assert len(results) == 2
        names = {ds.name for ds in results}
        assert names == {"ds1", "ds2"}

    def test_delete_dataset(self, service: DatasetService, db_session) -> None:
        """Dataset can be deleted."""
        data = DatasetCreate(
            name="delete-me",
            source_type="directory",
            source_path="/path",
        )
        created = service.create_dataset(data)
        db_session.flush()

        service.delete_dataset(created.id)
        db_session.flush()
        db_session.expire_all()

        # After flush, the dataset should no longer be in the list
        datasets = service.list_datasets()
        assert all(ds.name != "delete-me" for ds in datasets)

    # Document tests

    def test_create_document(self, service: DatasetService) -> None:
        """Document can be created in a dataset."""
        ds = service.create_dataset(
            DatasetCreate(name="doc-test", source_type="dir", source_path="/p")
        )
        doc_data = DocumentCreate(
            path="notes/test.md",
            content_hash="abc123",
            body="# Test Content",
            metadata={"tags": ["test"]},
        )

        result = service.create_document(ds.id, doc_data)

        assert result.id is not None
        assert result.dataset_id == ds.id
        assert result.path == "notes/test.md"
        assert result.content_hash == "abc123"
        assert result.body == "# Test Content"
        assert result.active is True
        assert result.metadata == {"tags": ["test"]}

    def test_create_document_invalid_dataset(self, service: DatasetService) -> None:
        """Creating document in non-existent dataset raises error."""
        doc_data = DocumentCreate(
            path="test.md",
            content_hash="hash",
            body="content",
        )

        with pytest.raises(DatasetNotFoundError):
            service.create_document(99999, doc_data)

    def test_get_document(self, service: DatasetService) -> None:
        """Document can be retrieved by ID."""
        ds = service.create_dataset(
            DatasetCreate(name="get-doc", source_type="dir", source_path="/p")
        )
        created = service.create_document(
            ds.id,
            DocumentCreate(path="test.md", content_hash="hash", body="content"),
        )

        result = service.get_document(created.id)

        assert result.id == created.id
        assert result.path == "test.md"

    def test_get_document_by_path(self, service: DatasetService) -> None:
        """Document can be retrieved by dataset and path."""
        ds = service.create_dataset(
            DatasetCreate(name="path-lookup", source_type="dir", source_path="/p")
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="notes/specific.md", content_hash="hash", body="content"),
        )

        result = service.get_document_by_path(ds.id, "notes/specific.md")

        assert result.path == "notes/specific.md"

    def test_update_document(self, service: DatasetService) -> None:
        """Document can be updated."""
        ds = service.create_dataset(
            DatasetCreate(name="update-doc", source_type="dir", source_path="/p")
        )
        created = service.create_document(
            ds.id,
            DocumentCreate(path="test.md", content_hash="old", body="old content"),
        )

        result = service.update_document(
            created.id,
            DocumentUpdate(
                content_hash="new",
                body="new content",
                metadata={"updated": True},
            ),
        )

        assert result.content_hash == "new"
        assert result.body == "new content"
        assert result.metadata == {"updated": True}

    def test_list_documents(self, service: DatasetService) -> None:
        """Documents in a dataset can be listed."""
        ds = service.create_dataset(
            DatasetCreate(name="list-docs", source_type="dir", source_path="/p")
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="doc1.md", content_hash="h1", body="c1"),
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="doc2.md", content_hash="h2", body="c2"),
        )

        results = service.list_documents(ds.id)

        assert len(results) == 2
        paths = {doc.path for doc in results}
        assert paths == {"doc1.md", "doc2.md"}

    def test_list_documents_active_only(self, service: DatasetService) -> None:
        """Can filter to only active documents."""
        ds = service.create_dataset(
            DatasetCreate(name="active-filter", source_type="dir", source_path="/p")
        )
        doc1 = service.create_document(
            ds.id,
            DocumentCreate(path="active.md", content_hash="h1", body="c1"),
        )
        doc2 = service.create_document(
            ds.id,
            DocumentCreate(path="inactive.md", content_hash="h2", body="c2"),
        )
        service.soft_delete_document(doc2.id)

        results = service.list_documents(ds.id, active_only=True)

        assert len(results) == 1
        assert results[0].path == "active.md"

    def test_list_document_paths(self, service: DatasetService) -> None:
        """Document paths can be listed."""
        ds = service.create_dataset(
            DatasetCreate(name="paths", source_type="dir", source_path="/p")
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="a.md", content_hash="h1", body="c1"),
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="b.md", content_hash="h2", body="c2"),
        )

        paths = service.list_document_paths(ds.id)

        assert paths == {"a.md", "b.md"}

    def test_soft_delete_document(self, service: DatasetService) -> None:
        """Document can be soft-deleted."""
        ds = service.create_dataset(
            DatasetCreate(name="soft-del", source_type="dir", source_path="/p")
        )
        doc = service.create_document(
            ds.id,
            DocumentCreate(path="delete-me.md", content_hash="h", body="c"),
        )

        result = service.soft_delete_document(doc.id)

        assert result.active is False

    def test_soft_delete_documents_by_paths(self, service: DatasetService) -> None:
        """Multiple documents can be soft-deleted by path."""
        ds = service.create_dataset(
            DatasetCreate(name="bulk-del", source_type="dir", source_path="/p")
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="keep.md", content_hash="h1", body="c1"),
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="delete1.md", content_hash="h2", body="c2"),
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="delete2.md", content_hash="h3", body="c3"),
        )

        count = service.soft_delete_documents_by_paths(
            ds.id, {"delete1.md", "delete2.md"}
        )

        assert count == 2
        active_docs = service.list_documents(ds.id, active_only=True)
        assert len(active_docs) == 1
        assert active_docs[0].path == "keep.md"

    def test_delete_document(self, service: DatasetService, db_session) -> None:
        """Document can be hard-deleted."""
        ds = service.create_dataset(
            DatasetCreate(name="hard-del", source_type="dir", source_path="/p")
        )
        doc = service.create_document(
            ds.id,
            DocumentCreate(path="gone.md", content_hash="h", body="c"),
        )
        db_session.flush()

        service.delete_document(doc.id)
        db_session.flush()
        db_session.expire_all()

        # After flush, the document should no longer be in the list
        docs = service.list_documents(ds.id)
        assert all(d.path != "gone.md" for d in docs)

    def test_dataset_document_count(self, service: DatasetService) -> None:
        """Dataset info includes active document count."""
        ds = service.create_dataset(
            DatasetCreate(name="count-test", source_type="dir", source_path="/p")
        )
        doc1 = service.create_document(
            ds.id,
            DocumentCreate(path="a.md", content_hash="h1", body="c1"),
        )
        service.create_document(
            ds.id,
            DocumentCreate(path="b.md", content_hash="h2", body="c2"),
        )
        service.soft_delete_document(doc1.id)

        result = service.get_dataset(ds.id)

        assert result.document_count == 1  # Only active documents counted
