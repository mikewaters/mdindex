"""Tests for IndexingService."""

import pytest
from pathlib import Path

from pmd.core.config import Config
from pmd.core.exceptions import CollectionNotFoundError
from pmd.services import IndexResult, CleanupResult, ServiceContainer
from pmd.services.indexing import IndexingService


class TestIndexingServiceInit:
    """Tests for IndexingService initialization."""

    def test_init_stores_container(self, connected_container: ServiceContainer):
        """IndexingService should store container reference."""
        service = IndexingService(connected_container)

        assert service._container is connected_container


class TestIndexingServiceIndexCollection:
    """Tests for IndexingService.index_collection method."""

    @pytest.mark.asyncio
    async def test_index_collection_not_found(self, config: Config):
        """index_collection should raise for unknown collection."""
        async with ServiceContainer(config) as services:
            with pytest.raises(CollectionNotFoundError):
                await services.indexing.index_collection("nonexistent")

    @pytest.mark.asyncio
    async def test_index_collection_invalid_path(self, config: Config):
        """index_collection should raise for non-existent path."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create(
                "bad-path", "/nonexistent/path/does/not/exist", "**/*.md"
            )

            with pytest.raises(ValueError, match="does not exist"):
                await services.indexing.index_collection("bad-path")

    @pytest.mark.asyncio
    async def test_index_collection_empty_directory(
        self, config: Config, tmp_path: Path
    ):
        """index_collection should return zero for empty directory."""
        async with ServiceContainer(config) as services:
            services.collection_repo.create("empty", str(tmp_path), "**/*.md")

            result = await services.indexing.index_collection("empty")

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

            result = await services.indexing.index_collection("docs")

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

            # First index
            result1 = await services.indexing.index_collection("docs", force=True)
            assert result1.indexed == 1

            # Second index without force
            result2 = await services.indexing.index_collection("docs", force=False)
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

            # First index
            result1 = await services.indexing.index_collection("docs", force=True)

            # Second index with force
            result2 = await services.indexing.index_collection("docs", force=True)

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

            result = await services.indexing.index_collection("docs")

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

            await services.indexing.index_collection("docs")

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

            await services.indexing.index_collection("docs")

            doc = services.document_repo.get(collection.id, "doc.md")
            assert doc.title == "doc"  # filename stem


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

            # Index to create documents
            await services.indexing.index_collection("test")

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

            # Index documents first
            await services.indexing.index_collection("test")

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

            # Index documents
            await services.indexing.index_collection("test")

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

            # Index documents
            await services.indexing.index_collection("test")

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

            # Index documents
            await services.indexing.index_collection("test")

            # Mock embedding generator to return 5 chunks
            mock_gen = AsyncMock()
            mock_gen.embed_document = AsyncMock(return_value=5)
            services._embedding_generator = mock_gen

            result = await services.indexing.embed_collection("test", force=True)

            assert result.chunks_total == 5
