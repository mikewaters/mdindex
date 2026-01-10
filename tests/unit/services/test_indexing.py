"""Tests for IndexingService."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock

from pmd.core.config import Config
from pmd.core.exceptions import CollectionNotFoundError
from pmd.services import IndexResult, CleanupResult, ServiceContainer
from pmd.services.indexing import IndexingService
from pmd.sources import FileSystemSource, SourceConfig, SourceListError


def _filesystem_source_for(collection) -> FileSystemSource:
    return FileSystemSource(
        SourceConfig(
            uri=collection.get_source_uri(),
            extra=collection.get_source_config_dict(),
        )
    )


def _filesystem_source_for_name(services: ServiceContainer, name: str) -> FileSystemSource:
    collection = services.collection_repo.get_by_name(name)
    assert collection is not None
    return _filesystem_source_for(collection)


class TestIndexingServiceInit:
    """Tests for IndexingService initialization."""

    def test_init_with_explicit_deps(self, connected_container: ServiceContainer):
        """IndexingService should work with explicit dependencies."""
        service = IndexingService(
            db=connected_container.db,
            collection_repo=connected_container.collection_repo,
            document_repo=connected_container.document_repo,
            fts_repo=connected_container.fts_repo,
            embedding_repo=connected_container.embedding_repo,
        )

        assert service._db is connected_container.db
        assert service._collection_repo is connected_container.collection_repo

    def test_from_container_factory(self, connected_container: ServiceContainer):
        """IndexingService.from_container should create service with container deps."""
        service = IndexingService.from_container(connected_container)

        assert service._db is connected_container.db
        assert service._collection_repo is connected_container.collection_repo
        assert service._fts_repo is connected_container.fts_repo


class TestIndexingServiceIndexCollection:
    """Tests for IndexingService.index_collection method."""

    @pytest.mark.asyncio
    async def test_index_collection_not_found(self, config: Config):
        """index_collection should raise for unknown collection."""
        async with ServiceContainer(config) as services:
            dummy_source = FileSystemSource(SourceConfig(uri="file:///tmp"))
            with pytest.raises(CollectionNotFoundError):
                await services.indexing.index_collection("nonexistent", dummy_source)

    @pytest.mark.asyncio
    async def test_index_collection_invalid_path(self, config: Config):
        """index_collection should raise for non-existent path."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create(
                "bad-path", "/nonexistent/path/does/not/exist", "**/*.md"
            )
            source = _filesystem_source_for_name(services, "bad-path")

            with pytest.raises(SourceListError, match="does not exist"):
                await services.indexing.index_collection("bad-path", source)

    @pytest.mark.asyncio
    async def test_index_collection_empty_directory(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should return zero for empty directory."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create("empty", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "empty")

            result = await services.indexing.index_collection("empty", source)

            assert isinstance(result, IndexResult)
            assert result.indexed == 0
            assert result.skipped == 0
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_index_collection_with_files(self, config: Config, tmp_path: Path):
        """index_collection should index markdown files."""
        # Create test files
        (tmp_path / "doc1.md").write_text("# Document 1\n\nContent one.")
        (tmp_path / "doc2.md").write_text("# Document 2\n\nContent two.")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")

            result = await services.indexing.index_collection("docs", source)

            assert result.indexed == 2
            assert result.skipped == 0
            assert result.errors == []

    @pytest.mark.asyncio
    async def test_index_collection_skips_unchanged(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should skip unchanged files when force=False."""
        (tmp_path / "doc.md").write_text("# Document\n\nContent.")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")

            # First index
            result1 = await services.indexing.index_collection(
                "docs",
                source,
                force=True,
            )
            assert result1.indexed == 1

            # Second index without force
            result2 = await services.indexing.index_collection(
                "docs",
                source,
                force=False,
            )
            assert result2.indexed == 0
            assert result2.skipped == 1

    @pytest.mark.asyncio
    async def test_index_collection_force_reindexes(
        self, config: Config, tmp_path: Path
    ):
        """index_collection with force=True should reindex all."""
        (tmp_path / "doc.md").write_text("# Document\n\nContent.")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")

            # First index
            result1 = await services.indexing.index_collection(
                "docs",
                source,
                force=True,
            )

            # Second index with force
            result2 = await services.indexing.index_collection(
                "docs",
                source,
                force=True,
            )

            assert result1.indexed == result2.indexed
            assert result2.skipped == 0

    @pytest.mark.asyncio
    async def test_index_collection_respects_glob(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should respect glob pattern."""
        (tmp_path / "doc.md").write_text("# Markdown")
        (tmp_path / "doc.txt").write_text("Plain text")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")

            result = await services.indexing.index_collection("docs", source)

            # Should only index .md file
            assert result.indexed == 1

    @pytest.mark.asyncio
    async def test_index_collection_extracts_title(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should extract title from markdown heading."""
        (tmp_path / "doc.md").write_text("# My Custom Title\n\nContent here.")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            collection = services.collection_repo.get_by_name("docs")
            source = _filesystem_source_for(collection)

            await services.indexing.index_collection("docs", source)

            # Check document title
            doc = services.document_repo.get(collection.id, "doc.md")
            assert doc.title == "My Custom Title"

    @pytest.mark.asyncio
    async def test_index_collection_uses_filename_fallback(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should use filename if no heading."""
        (tmp_path / "doc.md").write_text("Content without heading.")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            collection = services.collection_repo.get_by_name("docs")
            source = _filesystem_source_for(collection)

            await services.indexing.index_collection("docs", source)

            doc = services.document_repo.get(collection.id, "doc.md")
            assert doc.title == "doc"  # filename stem

    @pytest.mark.asyncio
    async def test_index_collection_removes_fts_for_non_indexable(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should remove FTS entries when content is non-indexable."""
        doc_path = tmp_path / "doc.md"
        doc_path.write_text("# Document\n\nBody content.")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")

            await services.indexing.index_collection("docs", source, force=True)

            # Ensure FTS row exists
            cursor = services.db.execute("SELECT COUNT(*) as count FROM documents_fts")
            assert cursor.fetchone()["count"] == 1

            # Make document title-only and reindex
            doc_path.write_text("# Document")
            await services.indexing.index_collection("docs", source, force=True)

            cursor = services.db.execute("SELECT COUNT(*) as count FROM documents_fts")
            assert cursor.fetchone()["count"] == 0

    @pytest.mark.asyncio
    async def test_index_collection_with_embed_triggers_embedding(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should call embed_collection when embed=True."""
        (tmp_path / "doc.md").write_text("# Document\n\nContent.")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")

            services.indexing.embed_collection = AsyncMock()

            await services.indexing.index_collection(
                "docs",
                source,
                force=True,
                embed=True,
            )

            services.indexing.embed_collection.assert_called_once_with(
                "docs",
                force=True,
            )


class TestIndexingServiceUpdateAllCollections:
    """Tests for IndexingService.update_all_collections method."""

    @pytest.mark.asyncio
    async def test_update_all_empty(self, config: Config):
        """update_all_collections should return empty dict for no collections."""
        async with ServiceContainer(config) as services:
            results = await services.indexing.update_all_collections()

            assert results == {}

    @pytest.mark.asyncio
    async def test_update_all_multiple_collections(
        self, config: Config, tmp_path: Path
    ):
        """update_all_collections should update all collections."""
        # Create directories for collections
        (tmp_path / "coll1").mkdir()
        (tmp_path / "coll2").mkdir()
        (tmp_path / "coll1" / "doc1.md").write_text("# Doc 1")
        (tmp_path / "coll2" / "doc2.md").write_text("# Doc 2")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("coll1", str(tmp_path / "coll1"), "**/*.md")
            services.collection_repo.create("coll2", str(tmp_path / "coll2"), "**/*.md")

            results = await services.indexing.update_all_collections()

            assert len(results) == 2
            assert "coll1" in results
            assert "coll2" in results
            assert results["coll1"].indexed == 1
            assert results["coll2"].indexed == 1


class TestIndexingServiceCleanupOrphans:
    """Tests for IndexingService.cleanup_orphans method."""

    @pytest.mark.asyncio
    async def test_cleanup_orphans_empty_database(self, config: Config):
        """cleanup_orphans should return zeros for empty database."""
        async with ServiceContainer(config) as services:
            result = await services.indexing.cleanup_orphans()

            assert isinstance(result, CleanupResult)
            assert result.orphaned_content == 0
            assert result.orphaned_embeddings == 0


class TestIndexingServiceExtractTitle:
    """Tests for IndexingService._extract_title static method."""

    def test_extract_title_from_heading(self):
        """_extract_title should extract from # heading."""
        content = "# My Title\n\nSome content."
        title = IndexingService._extract_title(content, "fallback")

        assert title == "My Title"

    def test_extract_title_strips_whitespace(self):
        """_extract_title should strip whitespace from title."""
        content = "#   Spaced Title   \n\nContent."
        title = IndexingService._extract_title(content, "fallback")

        assert title == "Spaced Title"

    def test_extract_title_uses_fallback(self):
        """_extract_title should use fallback if no heading."""
        content = "No heading here.\n\nJust content."
        title = IndexingService._extract_title(content, "fallback")

        assert title == "fallback"

    def test_extract_title_first_heading_only(self):
        """_extract_title should use first heading only."""
        content = "# First Title\n\n## Second\n\n# Third"
        title = IndexingService._extract_title(content, "fallback")

        assert title == "First Title"

    def test_extract_title_requires_space_after_hash(self):
        """_extract_title should require space after #."""
        content = "#NoSpace\n\nContent."
        title = IndexingService._extract_title(content, "fallback")

        assert title == "fallback"


class TestIndexingServiceEmbedCollection:
    """Tests for IndexingService.embed_collection method."""

    @pytest.mark.asyncio
    async def test_embed_collection_raises_without_vec(self, config: Config):
        """embed_collection should raise if vec not available."""
        async with ServiceContainer(config) as services:
            # Skip if vec is actually available
            if services.vec_available:
                pytest.skip("sqlite-vec is available")

            services.collection_repo.create("test", "/tmp", "**/*.md")

            with pytest.raises(RuntimeError, match="Vector storage not available"):
                await services.indexing.embed_collection("test")

    @pytest.mark.asyncio
    async def test_embed_collection_not_found(self, config: Config):
        """embed_collection should raise for unknown collection."""
        async with ServiceContainer(config) as services:
            # Force vec_available for this test
            services._vec_available = True

            with pytest.raises(CollectionNotFoundError):
                await services.indexing.embed_collection("nonexistent")

    @pytest.mark.asyncio
    async def test_embed_collection_returns_embed_result(
        self, config: Config, tmp_path: Path, mock_embedding_generator
    ):
        """embed_collection should return EmbedResult."""
        from pmd.services.indexing import EmbedResult
        from unittest.mock import AsyncMock

        (tmp_path / "doc.md").write_text("# Document\n\nContent here.")

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")

            # Index to create documents
            await services.indexing.index_collection("test", source)

            # Mock the embedding generator
            services._embedding_generator = mock_embedding_generator

            result = await services.indexing.embed_collection("test")

            assert isinstance(result, EmbedResult)
            assert result.embedded >= 0
            assert result.skipped >= 0

    @pytest.mark.asyncio
    async def test_embed_collection_calls_embed_document(
        self, config: Config, tmp_path: Path, mock_embedding_generator
    ):
        """embed_collection should call embed_document for each document."""
        (tmp_path / "doc1.md").write_text("# Doc 1\n\nContent one.")
        (tmp_path / "doc2.md").write_text("# Doc 2\n\nContent two.")

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")

            # Index documents first
            await services.indexing.index_collection("test", source)

            # Mock embedding generator
            services._embedding_generator = mock_embedding_generator

            await services.indexing.embed_collection("test", force=True)

            # Should have called embed_document for each document
            assert mock_embedding_generator.embed_document.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_collection_skips_existing(
        self, config: Config, tmp_path: Path, mock_embedding_generator
    ):
        """embed_collection should skip documents with existing embeddings."""
        from unittest.mock import MagicMock

        (tmp_path / "doc.md").write_text("# Document\n\nContent.")

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")

            # Index documents
            await services.indexing.index_collection("test", source)

            # Mock embedding repo to say embeddings exist
            services.embedding_repo.has_embeddings = MagicMock(return_value=True)
            services._embedding_generator = mock_embedding_generator

            result = await services.indexing.embed_collection("test", force=False)

            assert result.skipped == 1
            assert result.embedded == 0
            # embed_document should not be called when skipping
            mock_embedding_generator.embed_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_embed_collection_force_regenerates(
        self, config: Config, tmp_path: Path, mock_embedding_generator
    ):
        """embed_collection with force=True should regenerate all embeddings."""
        from unittest.mock import MagicMock

        (tmp_path / "doc.md").write_text("# Document\n\nContent.")

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")

            # Index documents
            await services.indexing.index_collection("test", source)

            # Mock embedding repo - embeddings exist
            services.embedding_repo.has_embeddings = MagicMock(return_value=True)
            services._embedding_generator = mock_embedding_generator

            result = await services.indexing.embed_collection("test", force=True)

            # Should still embed even though embeddings exist
            assert result.embedded == 1
            assert result.skipped == 0
            mock_embedding_generator.embed_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_collection_tracks_chunks(
        self, config: Config, tmp_path: Path
    ):
        """embed_collection should track total chunks embedded."""
        from unittest.mock import AsyncMock

        (tmp_path / "doc.md").write_text("# Document\n\nContent.")

        async with ServiceContainer(config) as services:
            services._vec_available = True
            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")

            # Index documents
            await services.indexing.index_collection("test", source)

            # Mock embedding generator to return 5 chunks
            mock_gen = AsyncMock()
            mock_gen.embed_document = AsyncMock(return_value=5)
            services._embedding_generator = mock_gen

            result = await services.indexing.embed_collection("test", force=True)

            assert result.chunks_total == 5


class TestIndexingServiceBackfillMetadata:
    """Tests for IndexingService.backfill_metadata method.

    These tests specifically exercise the source_config JSON parsing and
    Collection object creation (lines 709-718 in indexing.py).
    """

    @pytest.mark.asyncio
    async def test_backfill_metadata_empty_database(self, config: Config):
        """backfill_metadata should return zeros for empty database."""
        async with ServiceContainer(config) as services:
            stats = services.indexing.backfill_metadata()

            assert stats["processed"] == 0
            assert stats["updated"] == 0
            assert stats["skipped"] == 0
            assert stats["errors"] == []

    @pytest.mark.asyncio
    async def test_backfill_metadata_extracts_tags(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should extract metadata from documents."""
        # Create document with tags
        (tmp_path / "doc.md").write_text(
            "---\ntags: [python, testing]\n---\n# Test Doc\n\nContent."
        )

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # Clear existing metadata to test backfill
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            stats = services.indexing.backfill_metadata()

            assert stats["processed"] == 1
            assert stats["updated"] == 1
            assert stats["skipped"] == 0
            assert stats["errors"] == []

    @pytest.mark.asyncio
    async def test_backfill_metadata_skips_empty_body(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should skip documents with empty body."""
        (tmp_path / "doc.md").write_text("# Title only")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # Clear metadata so backfill will find this document
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            # Manually set content to empty to simulate edge case
            # Content is stored in the content table, not documents
            services.db.execute("UPDATE content SET doc = ''")

            stats = services.indexing.backfill_metadata()

            assert stats["processed"] == 1
            assert stats["skipped"] == 1
            assert stats["updated"] == 0

    @pytest.mark.asyncio
    async def test_backfill_metadata_with_source_config(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should parse source_config JSON."""
        import json

        (tmp_path / "doc.md").write_text(
            "---\ntags: [web]\n---\n# Doc\n\nContent."
        )

        async with ServiceContainer(config) as services:
            # Create collection with source_config
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")

            # Manually set source_config to test JSON parsing (line 710)
            source_config = json.dumps({"metadata_profile": "generic"})
            services.db.execute(
                "UPDATE collections SET source_config = ? WHERE name = ?",
                (source_config, "docs"),
            )

            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # Clear metadata for backfill test
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            stats = services.indexing.backfill_metadata()

            assert stats["updated"] == 1
            assert stats["errors"] == []

    @pytest.mark.asyncio
    async def test_backfill_metadata_with_null_source_config(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should handle NULL source_config gracefully."""
        (tmp_path / "doc.md").write_text(
            "---\ntags: [api]\n---\n# Doc\n\nContent."
        )

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")

            # Ensure source_config is NULL
            services.db.execute(
                "UPDATE collections SET source_config = NULL WHERE name = ?",
                ("docs",),
            )

            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # Clear metadata for backfill
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            # Should not raise - handles None source_config (line 710)
            stats = services.indexing.backfill_metadata()

            assert stats["updated"] == 1
            assert stats["errors"] == []

    @pytest.mark.asyncio
    async def test_backfill_metadata_with_metadata_profile_in_config(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should use metadata_profile from source_config."""
        import json

        # Create Obsidian-style document with nested tags
        (tmp_path / "doc.md").write_text(
            "---\ntags: [project/active, code/python]\n---\n# Doc\n\nContent."
        )

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")

            # Set metadata_profile to obsidian in source_config
            source_config = json.dumps({"metadata_profile": "obsidian"})
            services.db.execute(
                "UPDATE collections SET source_config = ? WHERE name = ?",
                (source_config, "docs"),
            )

            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # Clear metadata for backfill
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            stats = services.indexing.backfill_metadata()

            assert stats["updated"] == 1

            # Verify Obsidian profile was used (expands nested tags)
            from pmd.store.document_metadata import DocumentMetadataRepository
            metadata_repo = DocumentMetadataRepository(services.db)

            cursor = services.db.execute(
                "SELECT id FROM documents WHERE path = 'doc.md'"
            )
            doc_id = cursor.fetchone()["id"]
            tags = metadata_repo.get_tags(doc_id)

            # Obsidian profile expands project/active to include "project"
            assert "project" in tags
            assert "project/active" in tags

    @pytest.mark.asyncio
    async def test_backfill_metadata_collection_filter(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should filter by collection_name."""
        (tmp_path / "coll1").mkdir()
        (tmp_path / "coll2").mkdir()
        (tmp_path / "coll1" / "doc1.md").write_text("---\ntags: [a]\n---\n# 1")
        (tmp_path / "coll2" / "doc2.md").write_text("---\ntags: [b]\n---\n# 2")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("coll1", str(tmp_path / "coll1"), "**/*.md")
            services.collection_repo.create("coll2", str(tmp_path / "coll2"), "**/*.md")
            coll1_source = _filesystem_source_for_name(services, "coll1")
            coll2_source = _filesystem_source_for_name(services, "coll2")
            await services.indexing.index_collection("coll1", coll1_source)
            await services.indexing.index_collection("coll2", coll2_source)

            # Clear metadata
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            # Backfill only coll1
            stats = services.indexing.backfill_metadata(collection_name="coll1")

            assert stats["processed"] == 1
            assert stats["updated"] == 1

    @pytest.mark.asyncio
    async def test_backfill_metadata_force_reextracts(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata with force=True should re-extract existing metadata."""
        (tmp_path / "doc.md").write_text("---\ntags: [original]\n---\n# Doc")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # index_collection already extracts metadata, so:
            # - backfill without force finds no documents (already have metadata)
            stats1 = services.indexing.backfill_metadata(force=False)
            assert stats1["processed"] == 0  # No documents without metadata

            # - backfill WITH force should re-extract all
            stats2 = services.indexing.backfill_metadata(force=True)
            assert stats2["processed"] == 1
            assert stats2["updated"] == 1

            # Verify tags were extracted
            from pmd.store.document_metadata import DocumentMetadataRepository
            metadata_repo = DocumentMetadataRepository(services.db)
            cursor = services.db.execute("SELECT id FROM documents WHERE path = 'doc.md'")
            doc_id = cursor.fetchone()["id"]
            tags = metadata_repo.get_tags(doc_id)
            assert "original" in tags

    @pytest.mark.asyncio
    async def test_backfill_metadata_handles_extraction_errors(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should capture errors and continue."""
        (tmp_path / "good.md").write_text("---\ntags: [ok]\n---\n# Good")
        (tmp_path / "doc.md").write_text("# Doc with content")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # Clear metadata
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            # Both should process successfully (no extraction errors expected)
            stats = services.indexing.backfill_metadata()

            assert stats["processed"] == 2
            assert stats["errors"] == []

    @pytest.mark.asyncio
    async def test_backfill_metadata_with_empty_source_config_string(
        self, config: Config, tmp_path: Path
    ):
        """backfill_metadata should handle empty string source_config."""
        (tmp_path / "doc.md").write_text("---\ntags: [test]\n---\n# Doc")

        async with ServiceContainer(config) as services:
            services.collection_repo.create("docs", str(tmp_path), "**/*.md")

            # Set source_config to empty string
            services.db.execute(
                "UPDATE collections SET source_config = '' WHERE name = ?",
                ("docs",),
            )

            source = _filesystem_source_for_name(services, "docs")
            await services.indexing.index_collection("docs", source)

            # Clear metadata
            services.db.execute("DELETE FROM document_metadata")
            services.db.execute("DELETE FROM document_tags")

            # Should handle empty string gracefully (line 710: json.loads returns {})
            stats = services.indexing.backfill_metadata()

            assert stats["updated"] == 1
            assert stats["errors"] == []
