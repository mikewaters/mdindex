"""Tests for DatasetFacade operations."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.facade import DatasetFacade
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.models import LoadStatus, IndexState


@pytest.fixture
def dataset_facade(db: Database) -> DatasetFacade:
    """Provide a DatasetFacade instance."""
    return DatasetFacade(db)


class TestDatasetFacadeCollections:
    """Tests for collection operations via DatasetFacade."""

    def test_get_collection_by_name(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """get_collection_by_name should find existing collection."""
        found = dataset_facade.get_collection_by_name(sample_collection.name)

        assert found is not None
        assert found.id == sample_collection.id
        assert found.name == sample_collection.name

    def test_get_collection_by_name_not_exists(self, dataset_facade: DatasetFacade):
        """get_collection_by_name should return None if not found."""
        found = dataset_facade.get_collection_by_name("nonexistent")

        assert found is None

    def test_get_collection_by_id(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """get_collection_by_id should find existing collection."""
        found = dataset_facade.get_collection_by_id(sample_collection.id)

        assert found is not None
        assert found.id == sample_collection.id

    def test_list_all_collections(
        self,
        dataset_facade: DatasetFacade,
        collection_repo: SourceCollectionRepository,
        tmp_path: Path,
    ):
        """list_all_collections should return all collections."""
        collection_repo.create("coll1", str(tmp_path / "a"), "**/*.md")
        collection_repo.create("coll2", str(tmp_path / "b"), "**/*.md")

        collections = dataset_facade.list_all_collections()

        names = {c.name for c in collections}
        assert "coll1" in names
        assert "coll2" in names


class TestDatasetFacadeResources:
    """Tests for resource operations via DatasetFacade."""

    def test_upsert_resource(self, dataset_facade: DatasetFacade, sample_collection):
        """upsert_resource should create new resource."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id,
            "file:///test.md",
            resource_type="markdown",
        )

        assert resource is not None
        assert resource.uri == "file:///test.md"
        assert resource.resource_type == "markdown"

    def test_get_resource_by_uri(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """get_resource_by_uri should find created resource."""
        dataset_facade.upsert_resource(sample_collection.id, "file:///test.md")

        found = dataset_facade.get_resource_by_uri(
            sample_collection.id, "file:///test.md"
        )

        assert found is not None
        assert found.uri == "file:///test.md"

    def test_get_resource_by_id(self, dataset_facade: DatasetFacade, sample_collection):
        """get_resource_by_id should find created resource."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        found = dataset_facade.get_resource_by_id(resource.id)

        assert found is not None
        assert found.id == resource.id

    def test_list_resources_by_collection(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """list_resources_by_collection should return all resources."""
        dataset_facade.upsert_resource(sample_collection.id, "file:///a.md")
        dataset_facade.upsert_resource(sample_collection.id, "file:///b.md")

        resources = dataset_facade.list_resources_by_collection(sample_collection.id)

        assert len(resources) == 2
        uris = {r.uri for r in resources}
        assert uris == {"file:///a.md", "file:///b.md"}

    def test_list_resources_needing_index(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """list_resources_needing_index should find loaded resources needing indexing."""
        # Create loaded resource with pending index
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )
        dataset_facade.mark_loaded(resource.id, "hash123", None)

        needing = dataset_facade.list_resources_needing_index(sample_collection.id)

        assert len(needing) == 1
        assert needing[0].uri == "file:///test.md"

    def test_delete_orphaned_resources(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """delete_orphaned_resources should remove resources not in valid set."""
        dataset_facade.upsert_resource(sample_collection.id, "file:///keep.md")
        dataset_facade.upsert_resource(sample_collection.id, "file:///delete.md")

        deleted = dataset_facade.delete_orphaned_resources(
            sample_collection.id, {"file:///keep.md"}
        )

        assert deleted == 1
        resources = dataset_facade.list_resources_by_collection(sample_collection.id)
        assert len(resources) == 1
        assert resources[0].uri == "file:///keep.md"


class TestDatasetFacadeStateTransitions:
    """Tests for resource state transitions via DatasetFacade."""

    def test_mark_loading(self, dataset_facade: DatasetFacade, sample_collection):
        """mark_loading should set load_status='loading'."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        dataset_facade.mark_loading(resource.id)

        updated = dataset_facade.get_resource_by_id(resource.id)
        assert updated.load_status == "loading"

    def test_mark_loaded(self, dataset_facade: DatasetFacade, sample_collection):
        """mark_loaded should set load_status='loaded' and update hash."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        dataset_facade.mark_loaded(resource.id, "sha256:abc", "/cache/abc")

        updated = dataset_facade.get_resource_by_id(resource.id)
        assert updated.load_status == "loaded"
        assert updated.hash == "sha256:abc"
        assert updated.content_ref == "/cache/abc"

    def test_mark_load_failed(self, dataset_facade: DatasetFacade, sample_collection):
        """mark_load_failed should set load_status='error'."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        dataset_facade.mark_load_failed(resource.id, "Connection failed")

        updated = dataset_facade.get_resource_by_id(resource.id)
        assert updated.load_status == "error"
        assert updated.load_error == "Connection failed"

    def test_mark_indexing(self, dataset_facade: DatasetFacade, sample_collection):
        """mark_indexing should set index_state='indexing'."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        dataset_facade.mark_indexing(resource.id)

        updated = dataset_facade.get_resource_by_id(resource.id)
        assert updated.index_state == "indexing"

    def test_mark_indexed(self, dataset_facade: DatasetFacade, sample_collection):
        """mark_indexed should set index_state='indexed'."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        dataset_facade.mark_indexed(resource.id, method="fts")

        updated = dataset_facade.get_resource_by_id(resource.id)
        assert updated.index_state == "indexed"
        assert updated.index_method == "fts"

    def test_mark_index_failed(self, dataset_facade: DatasetFacade, sample_collection):
        """mark_index_failed should set index_state='error'."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        dataset_facade.mark_index_failed(resource.id, "FTS error")

        updated = dataset_facade.get_resource_by_id(resource.id)
        assert updated.index_state == "error"
        assert updated.index_error == "FTS error"

    def test_mark_stale(self, dataset_facade: DatasetFacade, sample_collection):
        """mark_stale should set appropriate status."""
        resource = dataset_facade.upsert_resource(
            sample_collection.id, "file:///test.md"
        )

        dataset_facade.mark_stale(resource.id, "load")

        updated = dataset_facade.get_resource_by_id(resource.id)
        assert updated.load_status == "stale"


class TestDatasetFacadeDocuments:
    """Tests for document operations via DatasetFacade."""

    def test_add_or_update_document(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """add_or_update_document should create new document."""
        doc_result, is_new = dataset_facade.add_or_update_document(
            sample_collection.id,
            "test.md",
            "Test Title",
            "# Test Content",
        )

        assert is_new is True
        assert doc_result.filepath == "test.md"
        assert doc_result.title == "Test Title"

    def test_get_document(self, dataset_facade: DatasetFacade, sample_collection):
        """get_document should find created document."""
        dataset_facade.add_or_update_document(
            sample_collection.id, "test.md", "Title", "Content"
        )

        found = dataset_facade.get_document(sample_collection.id, "test.md")

        assert found is not None
        assert found.filepath == "test.md"

    def test_list_documents_by_collection(
        self, dataset_facade: DatasetFacade, sample_collection
    ):
        """list_documents_by_collection should return all documents."""
        dataset_facade.add_or_update_document(
            sample_collection.id, "a.md", "A", "Content A"
        )
        dataset_facade.add_or_update_document(
            sample_collection.id, "b.md", "B", "Content B"
        )

        docs = dataset_facade.list_documents_by_collection(sample_collection.id)

        assert len(docs) == 2
        paths = {d.filepath for d in docs}
        assert paths == {"a.md", "b.md"}


# Note: Content operations tests removed as ContentRepository
# focuses on orphan cleanup, not direct content access.
# Content is managed through DocumentRepository.
