"""Tests for resource CRUD and state transition operations."""

import pytest
from pathlib import Path

from pmd.store.database import Database
from pmd.store.repositories.resource import ResourceRepository
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.models import IndexState, LoadStatus, ResourceModel


@pytest.fixture
def resource_repo(db: Database) -> ResourceRepository:
    """Provide a ResourceRepository instance."""
    return ResourceRepository(db)


class TestResourceUpsert:
    """Tests for resource upsert (create/update)."""

    def test_upsert_creates_new_resource(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """Upsert should create a new resource when it doesn't exist."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
            resource_type="markdown",
        )

        assert isinstance(resource, ResourceModel)
        assert resource.id is not None
        assert resource.source_collection_id == sample_collection.id
        assert resource.uri == "file:///docs/test.md"
        assert resource.resource_type == "markdown"
        assert resource.load_status == LoadStatus.PENDING.value
        assert resource.index_state == IndexState.PENDING.value

    def test_upsert_updates_existing_resource(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """Upsert should update an existing resource with same collection and uri."""
        # First create
        resource1 = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
            resource_type="markdown",
        )

        # Then update
        resource2 = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
            resource_type="text",
            hash="abc123",
        )

        assert resource2.id == resource1.id
        assert resource2.resource_type == "text"
        assert resource2.hash == "abc123"

    def test_upsert_sets_timestamps(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """Upsert should set created_at and updated_at on new resource."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
        )

        assert resource.created_at is not None
        assert resource.updated_at is not None
        assert resource.created_at == resource.updated_at

    def test_upsert_updates_timestamp_on_update(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """Upsert should update updated_at when updating existing resource."""
        resource1 = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
        )
        original_updated = resource1.updated_at

        resource2 = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
            resource_type="updated",
        )

        assert resource2.updated_at >= original_updated

    def test_upsert_with_metadata_dict(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """Upsert should serialize metadata dict to JSON."""
        metadata = {"size": 1024, "author": "test"}
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
            resource_metadata=metadata,
        )

        assert resource.resource_metadata is not None
        import json
        stored = json.loads(resource.resource_metadata)
        assert stored == metadata

    def test_upsert_with_enum_values(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """Upsert should accept LoadStatus and IndexState enum values."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.INDEXED,
        )

        assert resource.load_status == LoadStatus.LOADED.value
        assert resource.index_state == IndexState.INDEXED.value


class TestResourceGet:
    """Tests for resource retrieval."""

    def test_get_by_uri_exists(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """get_by_uri should return resource if exists."""
        created = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
        )

        found = resource_repo.get_by_uri(sample_collection.id, "file:///docs/test.md")

        assert found is not None
        assert found.id == created.id
        assert found.uri == "file:///docs/test.md"

    def test_get_by_uri_not_exists(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """get_by_uri should return None if not exists."""
        found = resource_repo.get_by_uri(sample_collection.id, "nonexistent")

        assert found is None

    def test_get_by_id_exists(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """get_by_id should return resource if exists."""
        created = resource_repo.upsert(
            sample_collection.id,
            "file:///docs/test.md",
        )

        found = resource_repo.get_by_id(created.id)

        assert found is not None
        assert found.id == created.id
        assert found.uri == "file:///docs/test.md"

    def test_get_by_id_not_exists(self, resource_repo: ResourceRepository):
        """get_by_id should return None if not exists."""
        found = resource_repo.get_by_id(99999)

        assert found is None


class TestResourceListByCollection:
    """Tests for listing resources by collection."""

    def test_list_empty_collection(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_by_collection should return empty list for empty collection."""
        resources = resource_repo.list_by_collection(sample_collection.id)

        assert resources == []

    def test_list_resources_in_collection(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_by_collection should return all resources."""
        resource_repo.upsert(sample_collection.id, "file:///a.md")
        resource_repo.upsert(sample_collection.id, "file:///b.md")
        resource_repo.upsert(sample_collection.id, "file:///c.md")

        resources = resource_repo.list_by_collection(sample_collection.id)

        assert len(resources) == 3
        uris = {r.uri for r in resources}
        assert uris == {"file:///a.md", "file:///b.md", "file:///c.md"}

    def test_list_ordered_by_uri(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_by_collection should return resources ordered by uri."""
        resource_repo.upsert(sample_collection.id, "file:///z.md")
        resource_repo.upsert(sample_collection.id, "file:///a.md")
        resource_repo.upsert(sample_collection.id, "file:///m.md")

        resources = resource_repo.list_by_collection(sample_collection.id)
        uris = [r.uri for r in resources]

        assert uris == ["file:///a.md", "file:///m.md", "file:///z.md"]

    def test_list_filter_by_status(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_by_collection should filter by load_status."""
        resource_repo.upsert(
            sample_collection.id, "file:///a.md", load_status=LoadStatus.PENDING
        )
        resource_repo.upsert(
            sample_collection.id, "file:///b.md", load_status=LoadStatus.LOADED
        )
        resource_repo.upsert(
            sample_collection.id, "file:///c.md", load_status=LoadStatus.LOADED
        )

        resources = resource_repo.list_by_collection(
            sample_collection.id, status=LoadStatus.LOADED
        )

        assert len(resources) == 2
        assert all(r.load_status == LoadStatus.LOADED.value for r in resources)

    def test_list_filter_by_state(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_by_collection should filter by index_state."""
        resource_repo.upsert(
            sample_collection.id, "file:///a.md", index_state=IndexState.PENDING
        )
        resource_repo.upsert(
            sample_collection.id, "file:///b.md", index_state=IndexState.INDEXED
        )

        resources = resource_repo.list_by_collection(
            sample_collection.id, state=IndexState.INDEXED
        )

        assert len(resources) == 1
        assert resources[0].index_state == IndexState.INDEXED.value

    def test_list_filter_by_status_and_state(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_by_collection should filter by both status and state."""
        resource_repo.upsert(
            sample_collection.id,
            "file:///a.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.PENDING,
        )
        resource_repo.upsert(
            sample_collection.id,
            "file:///b.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.INDEXED,
        )
        resource_repo.upsert(
            sample_collection.id,
            "file:///c.md",
            load_status=LoadStatus.PENDING,
            index_state=IndexState.PENDING,
        )

        resources = resource_repo.list_by_collection(
            sample_collection.id,
            status=LoadStatus.LOADED,
            state=IndexState.PENDING,
        )

        assert len(resources) == 1
        assert resources[0].uri == "file:///a.md"


class TestResourceDelete:
    """Tests for resource deletion."""

    def test_delete_existing_resource(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """delete should remove existing resource."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        result = resource_repo.delete(resource.id)

        assert result is True
        assert resource_repo.get_by_id(resource.id) is None

    def test_delete_nonexistent_resource(self, resource_repo: ResourceRepository):
        """delete should return False for nonexistent resource."""
        result = resource_repo.delete(99999)

        assert result is False


class TestResourceCount:
    """Tests for resource counting."""

    def test_count_empty_collection(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """count_by_collection should return 0 for empty collection."""
        count = resource_repo.count_by_collection(sample_collection.id)

        assert count == 0

    def test_count_resources(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """count_by_collection should return correct count."""
        resource_repo.upsert(sample_collection.id, "file:///a.md")
        resource_repo.upsert(sample_collection.id, "file:///b.md")

        count = resource_repo.count_by_collection(sample_collection.id)

        assert count == 2

    def test_count_with_status_filter(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """count_by_collection should respect status filter."""
        resource_repo.upsert(
            sample_collection.id, "file:///a.md", load_status=LoadStatus.LOADED
        )
        resource_repo.upsert(
            sample_collection.id, "file:///b.md", load_status=LoadStatus.PENDING
        )
        resource_repo.upsert(
            sample_collection.id, "file:///c.md", load_status=LoadStatus.LOADED
        )

        count = resource_repo.count_by_collection(
            sample_collection.id, status=LoadStatus.LOADED
        )

        assert count == 2


class TestResourceMarkLoading:
    """Tests for mark_loading state transition."""

    def test_mark_loading_sets_status(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_loading should set load_status='loading'."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_loading(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.load_status == "loading"

    def test_mark_loading_clears_error(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_loading should clear previous load_error."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            load_status=LoadStatus.ERROR,
            load_error="Previous error",
        )

        resource_repo.mark_loading(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.load_error is None

    def test_mark_loading_updates_timestamp(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_loading should update updated_at."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")
        original = resource.updated_at

        resource_repo.mark_loading(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.updated_at >= original


class TestResourceMarkLoaded:
    """Tests for mark_loaded state transition."""

    def test_mark_loaded_sets_status_and_hash(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_loaded should set load_status='loaded' and update hash/content_ref."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_loaded(resource.id, "sha256:abc123", "/cache/abc123")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.load_status == "loaded"
        assert updated.hash == "sha256:abc123"
        assert updated.content_ref == "/cache/abc123"

    def test_mark_loaded_sets_loaded_at(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_loaded should set loaded_at timestamp."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_loaded(resource.id, "sha256:abc123", None)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.loaded_at is not None

    def test_mark_loaded_clears_error(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_loaded should clear previous load_error."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            load_error="Previous error",
        )

        resource_repo.mark_loaded(resource.id, "sha256:abc123", None)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.load_error is None

    def test_mark_loaded_merges_metadata(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_loaded should merge metadata with existing."""
        import json

        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            resource_metadata={"existing": "value"},
        )

        resource_repo.mark_loaded(
            resource.id,
            "sha256:abc123",
            None,
            metadata={"new": "data"},
        )

        updated = resource_repo.get_by_id(resource.id)
        stored = json.loads(updated.resource_metadata)
        assert stored == {"existing": "value", "new": "data"}


class TestResourceMarkLoadFailed:
    """Tests for mark_load_failed state transition."""

    def test_mark_load_failed_sets_error_status(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_load_failed should set load_status='error'."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_load_failed(resource.id, "Connection timeout")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.load_status == "error"

    def test_mark_load_failed_records_error(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_load_failed should record error message."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_load_failed(resource.id, "Connection timeout")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.load_error == "Connection timeout"

    def test_mark_load_failed_updates_timestamp(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_load_failed should update updated_at."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")
        original = resource.updated_at

        resource_repo.mark_load_failed(resource.id, "Error")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.updated_at >= original


class TestResourceMarkIndexing:
    """Tests for mark_indexing state transition."""

    def test_mark_indexing_sets_state(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_indexing should set index_state='indexing'."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_indexing(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_state == "indexing"

    def test_mark_indexing_clears_error(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_indexing should clear previous index_error."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            index_state=IndexState.ERROR,
            index_error="Previous error",
        )

        resource_repo.mark_indexing(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_error is None

    def test_mark_indexing_updates_timestamp(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_indexing should update updated_at."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")
        original = resource.updated_at

        resource_repo.mark_indexing(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.updated_at >= original


class TestResourceMarkIndexed:
    """Tests for mark_indexed state transition."""

    def test_mark_indexed_sets_state(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_indexed should set index_state='indexed'."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_indexed(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_state == "indexed"

    def test_mark_indexed_sets_indexed_at(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_indexed should set indexed_at timestamp."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_indexed(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.indexed_at is not None

    def test_mark_indexed_records_method(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_indexed should record index_method if provided."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_indexed(resource.id, method="fts+embedding")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_method == "fts+embedding"

    def test_mark_indexed_clears_error(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_indexed should clear previous index_error."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            index_error="Previous error",
        )

        resource_repo.mark_indexed(resource.id)

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_error is None


class TestResourceMarkIndexFailed:
    """Tests for mark_index_failed state transition."""

    def test_mark_index_failed_sets_error_state(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_index_failed should set index_state='error'."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_index_failed(resource.id, "Embedding model unavailable")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_state == "error"

    def test_mark_index_failed_records_error(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_index_failed should record error message."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        resource_repo.mark_index_failed(resource.id, "Embedding model unavailable")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_error == "Embedding model unavailable"

    def test_mark_index_failed_updates_timestamp(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_index_failed should update updated_at."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")
        original = resource.updated_at

        resource_repo.mark_index_failed(resource.id, "Error")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.updated_at >= original


class TestResourceMarkStale:
    """Tests for mark_stale state transition."""

    def test_mark_stale_load_sets_load_status(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_stale with reason='load' should set load_status='stale'."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            load_status=LoadStatus.LOADED,
        )

        resource_repo.mark_stale(resource.id, "load")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.load_status == "stale"

    def test_mark_stale_index_sets_index_state(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_stale with reason='index' should set index_state='stale'."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            index_state=IndexState.INDEXED,
        )

        resource_repo.mark_stale(resource.id, "index")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.index_state == "stale"

    def test_mark_stale_invalid_reason_raises(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_stale with invalid reason should raise ValueError."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")

        with pytest.raises(ValueError, match="reason must be 'load' or 'index'"):
            resource_repo.mark_stale(resource.id, "invalid")

    def test_mark_stale_updates_timestamp(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """mark_stale should update updated_at."""
        resource = resource_repo.upsert(sample_collection.id, "file:///test.md")
        original = resource.updated_at

        resource_repo.mark_stale(resource.id, "load")

        updated = resource_repo.get_by_id(resource.id)
        assert updated.updated_at >= original


class TestResourceListNeedingIndex:
    """Tests for list_needing_index query method."""

    def test_list_needing_index_empty(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_needing_index should return empty list when no resources need indexing."""
        # Resource not loaded
        resource_repo.upsert(sample_collection.id, "file:///a.md")
        # Resource already indexed
        resource_repo.upsert(
            sample_collection.id,
            "file:///b.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.INDEXED,
        )

        resources = resource_repo.list_needing_index(sample_collection.id)

        assert resources == []

    def test_list_needing_index_finds_pending(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_needing_index should find loaded resources with pending index_state."""
        resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.PENDING,
        )

        resources = resource_repo.list_needing_index(sample_collection.id)

        assert len(resources) == 1
        assert resources[0].uri == "file:///test.md"

    def test_list_needing_index_finds_stale(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_needing_index should find loaded resources with stale index_state."""
        resource = resource_repo.upsert(
            sample_collection.id,
            "file:///test.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.INDEXED,
        )
        resource_repo.mark_stale(resource.id, "index")

        resources = resource_repo.list_needing_index(sample_collection.id)

        assert len(resources) == 1
        assert resources[0].uri == "file:///test.md"

    def test_list_needing_index_excludes_not_loaded(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_needing_index should exclude resources not yet loaded."""
        resource_repo.upsert(
            sample_collection.id,
            "file:///pending.md",
            load_status=LoadStatus.PENDING,
            index_state=IndexState.PENDING,
        )
        resource_repo.upsert(
            sample_collection.id,
            "file:///loaded.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.PENDING,
        )

        resources = resource_repo.list_needing_index(sample_collection.id)

        assert len(resources) == 1
        assert resources[0].uri == "file:///loaded.md"

    def test_list_needing_index_ordered_by_uri(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """list_needing_index should return resources ordered by uri."""
        resource_repo.upsert(
            sample_collection.id,
            "file:///z.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.PENDING,
        )
        resource_repo.upsert(
            sample_collection.id,
            "file:///a.md",
            load_status=LoadStatus.LOADED,
            index_state=IndexState.PENDING,
        )

        resources = resource_repo.list_needing_index(sample_collection.id)
        uris = [r.uri for r in resources]

        assert uris == ["file:///a.md", "file:///z.md"]


class TestResourceDeleteOrphaned:
    """Tests for delete_orphaned cleanup method."""

    def test_delete_orphaned_removes_invalid_uris(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """delete_orphaned should remove resources not in valid URIs set."""
        resource_repo.upsert(sample_collection.id, "file:///keep1.md")
        resource_repo.upsert(sample_collection.id, "file:///keep2.md")
        resource_repo.upsert(sample_collection.id, "file:///delete1.md")
        resource_repo.upsert(sample_collection.id, "file:///delete2.md")

        valid_uris = {"file:///keep1.md", "file:///keep2.md"}
        deleted = resource_repo.delete_orphaned(sample_collection.id, valid_uris)

        assert deleted == 2
        resources = resource_repo.list_by_collection(sample_collection.id)
        uris = {r.uri for r in resources}
        assert uris == valid_uris

    def test_delete_orphaned_returns_count(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """delete_orphaned should return number of deleted resources."""
        resource_repo.upsert(sample_collection.id, "file:///a.md")
        resource_repo.upsert(sample_collection.id, "file:///b.md")
        resource_repo.upsert(sample_collection.id, "file:///c.md")

        deleted = resource_repo.delete_orphaned(
            sample_collection.id, {"file:///a.md"}
        )

        assert deleted == 2

    def test_delete_orphaned_empty_valid_set_deletes_all(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """delete_orphaned with empty valid URIs should delete all resources."""
        resource_repo.upsert(sample_collection.id, "file:///a.md")
        resource_repo.upsert(sample_collection.id, "file:///b.md")

        deleted = resource_repo.delete_orphaned(sample_collection.id, set())

        assert deleted == 2
        assert resource_repo.count_by_collection(sample_collection.id) == 0

    def test_delete_orphaned_no_orphans(
        self, resource_repo: ResourceRepository, sample_collection
    ):
        """delete_orphaned should return 0 when no orphans exist."""
        resource_repo.upsert(sample_collection.id, "file:///a.md")
        resource_repo.upsert(sample_collection.id, "file:///b.md")

        deleted = resource_repo.delete_orphaned(
            sample_collection.id, {"file:///a.md", "file:///b.md"}
        )

        assert deleted == 0
        assert resource_repo.count_by_collection(sample_collection.id) == 2

    def test_delete_orphaned_only_affects_collection(
        self,
        resource_repo: ResourceRepository,
        collection_repo: SourceCollectionRepository,
        tmp_path: Path,
    ):
        """delete_orphaned should only affect specified collection."""
        # Create two collections
        collection1 = collection_repo.create("coll1", str(tmp_path), "**/*.md")
        collection2 = collection_repo.create("coll2", str(tmp_path / "other"), "**/*.md")

        resource_repo.upsert(collection1.id, "file:///a.md")
        resource_repo.upsert(collection2.id, "file:///b.md")

        # Delete orphans from collection1 only
        deleted = resource_repo.delete_orphaned(collection1.id, set())

        assert deleted == 1
        assert resource_repo.count_by_collection(collection1.id) == 0
        assert resource_repo.count_by_collection(collection2.id) == 1
