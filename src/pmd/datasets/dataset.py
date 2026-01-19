"""Dataset orchestration class for sync → materialize → index workflows.

The Dataset class provides high-level orchestration for managing a collection's
resources through their lifecycle:
1. sync_resources(): Fetch resources from source, update load state
2. materialize_documents(): Create/update Documents from loaded Resources
3. index(): Index documents (FTS + optional embeddings), update Resource.index_state
4. refresh(): Convenience method that runs all three steps
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pmd.sources import get_default_registry

from .results import DatasetIndexResult, MaterializeResult, SyncResult

if TYPE_CHECKING:
    from pmd.core.types import SourceCollection
    from pmd.store.caching import ResourceCacher
    from pmd.store.facade import DatasetFacade


class Dataset:
    """Orchestrates resource lifecycle for a single SourceCollection.

    The Dataset class wraps a SourceCollection and provides methods to:
    - Sync resources from the source (fetch new/changed content)
    - Materialize documents from loaded resources
    - Index documents for search

    Example:
        facade = DatasetFacade(db)
        cacher = ResourceCacher(config)
        collection = facade.get_collection_by_name("my-docs")

        dataset = Dataset(collection, facade, cacher)
        result = await dataset.sync_resources()
        print(f"Added {result.added} new resources")
    """

    def __init__(
        self,
        collection: SourceCollection,
        facade: DatasetFacade,
        cacher: ResourceCacher | None = None,
    ) -> None:
        """Initialize Dataset with collection and dependencies.

        Args:
            collection: The SourceCollection to manage.
            facade: DatasetFacade for database operations.
            cacher: Optional ResourceCacher for caching fetched content.
        """
        self._collection = collection
        self._facade = facade
        self._cacher = cacher
        self._registry = get_default_registry()

    @property
    def collection(self) -> SourceCollection:
        """The underlying source collection."""
        return self._collection

    @property
    def collection_id(self) -> int:
        """ID of the source collection."""
        return self._collection.id

    async def sync_resources(
        self,
        mode: Literal["full", "incremental"] = "incremental",
    ) -> SyncResult:
        """Synchronize resources from the source.

        Fetches resources from the source collection and updates the database.
        In incremental mode, only checks resources that are pending or stale.
        In full mode, re-checks all resources for changes.

        Args:
            mode: Sync mode - "incremental" (default) or "full".

        Returns:
            SyncResult with counts of added, updated, unchanged, and failed resources.
        """
        result = SyncResult()

        # Create source from collection
        source = self._registry.create_source(self._collection)

        # Get existing resources
        existing_resources = self._facade.list_resources_by_collection(self.collection_id)
        existing_by_uri = {r.uri: r for r in existing_resources}

        # Discover resources from source
        try:
            discovered_uris = set()
            async for item in source.discover():
                uri = item.uri
                discovered_uris.add(uri)

                existing = existing_by_uri.get(uri)

                if existing is None:
                    # New resource
                    try:
                        resource = self._facade.upsert_resource(
                            self.collection_id,
                            uri,
                            resource_type=getattr(item, "content_type", None),
                        )
                        self._facade.mark_loading(resource.id)

                        # Fetch content
                        content = await source.fetch(item)
                        content_hash = self._compute_hash(content)

                        # Cache if cacher available
                        content_ref = None
                        if self._cacher and self._cacher.enabled:
                            content_ref = self._cacher.cache_resource(
                                self._collection.name,
                                uri,
                                content,
                            )

                        self._facade.mark_loaded(resource.id, content_hash, content_ref)
                        result.added += 1
                    except Exception as e:
                        if existing is None:
                            # Create resource in error state
                            resource = self._facade.upsert_resource(self.collection_id, uri)
                        self._facade.mark_load_failed(resource.id, str(e))
                        result.failed += 1
                        result.errors.append((uri, str(e)))

                elif mode == "full" or existing.load_status in ("pending", "stale"):
                    # Check for changes
                    try:
                        self._facade.mark_loading(existing.id)
                        content = await source.fetch(item)
                        content_hash = self._compute_hash(content)

                        if content_hash != existing.hash:
                            # Content changed
                            content_ref = None
                            if self._cacher and self._cacher.enabled:
                                content_ref = self._cacher.cache_resource(
                                    self._collection.name,
                                    uri,
                                    content,
                                )
                            self._facade.mark_loaded(existing.id, content_hash, content_ref)
                            # Mark index as stale since content changed
                            self._facade.mark_stale(existing.id, "index")
                            result.updated += 1
                        else:
                            # No change
                            self._facade.mark_loaded(existing.id, content_hash, existing.content_ref)
                            result.unchanged += 1
                    except Exception as e:
                        self._facade.mark_load_failed(existing.id, str(e))
                        result.failed += 1
                        result.errors.append((uri, str(e)))
                else:
                    # Already loaded and not stale
                    result.unchanged += 1

            # Clean up orphaned resources
            deleted = self._facade.delete_orphaned_resources(self.collection_id, discovered_uris)
            if deleted > 0:
                # Note: orphan deletion is tracked separately, not in result counts
                pass

        except Exception as e:
            # Source-level error
            result.errors.append(("(source)", str(e)))

        return result

    async def materialize_documents(self) -> MaterializeResult:
        """Create or update documents from loaded resources.

        For each resource with load_status='loaded', creates or updates
        the corresponding document in the documents table.

        Returns:
            MaterializeResult with counts of created, updated, and skipped documents.
        """
        result = MaterializeResult()

        # Get all loaded resources
        resources = self._facade.list_resources_by_collection(
            self.collection_id,
            status="loaded",
        )

        for resource in resources:
            # Get content for this resource
            content = None
            if resource.content_ref and self._cacher:
                cached_path = self._cacher.get_cached_path(
                    self._collection.name,
                    resource.uri,
                )
                if cached_path:
                    content = cached_path.read_text(encoding="utf-8")

            if content is None:
                # Try to get from content repository via hash
                content = self._facade.get_content(resource.hash) if resource.hash else None

            if content is None:
                # Skip if we can't get content
                result.skipped += 1
                continue

            # Extract title from content (simple: first line or URI)
            title = self._extract_title(content, resource.uri)

            # Create/update document
            doc_result, is_new = self._facade.add_or_update_document(
                self.collection_id,
                resource.uri,  # Use URI as path
                title,
                content,
            )

            if is_new:
                result.created += 1
            else:
                result.updated += 1

        return result

    async def index(self) -> DatasetIndexResult:
        """Index documents and update resource index state.

        Indexes all resources that need indexing (load_status='loaded' and
        index_state in ('pending', 'stale')).

        Returns:
            DatasetIndexResult with counts of indexed and failed documents.
        """
        result = DatasetIndexResult()

        # Get resources needing indexing
        resources = self._facade.list_resources_needing_index(self.collection_id)

        for resource in resources:
            try:
                self._facade.mark_indexing(resource.id)

                # The actual indexing is done via IndexingService
                # For now, we just mark as indexed
                # In full integration, this would call IndexingService
                self._facade.mark_indexed(resource.id, method="dataset")
                result.indexed += 1

            except Exception as e:
                self._facade.mark_index_failed(resource.id, str(e))
                result.failed += 1
                result.errors.append((resource.uri, str(e)))

        return result

    async def refresh(self) -> tuple[SyncResult, MaterializeResult, DatasetIndexResult]:
        """Convenience method: sync + materialize + index.

        Runs the full resource lifecycle in order:
        1. sync_resources() - Fetch from source
        2. materialize_documents() - Create documents
        3. index() - Index for search

        Returns:
            Tuple of (SyncResult, MaterializeResult, DatasetIndexResult).
        """
        sync_result = await self.sync_resources()
        materialize_result = await self.materialize_documents()
        index_result = await self.index()
        return sync_result, materialize_result, index_result

    def _compute_hash(self, content: str) -> str:
        """Compute content hash for deduplication."""
        import hashlib
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract title from content or use fallback."""
        # Try to get first line as title
        lines = content.strip().split("\n")
        if lines:
            first_line = lines[0].strip()
            # Remove markdown heading prefix
            if first_line.startswith("#"):
                first_line = first_line.lstrip("#").strip()
            if first_line:
                return first_line[:200]  # Limit length
        return fallback
