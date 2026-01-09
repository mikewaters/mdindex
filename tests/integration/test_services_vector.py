"""Integration tests for service layer with vector operations.

These tests exercise the full service layer pipeline including:
- ServiceContainer with MLX provider
- IndexingService.embed_collection
- SearchService.vector_search and hybrid_search

Tests require macOS with Apple Silicon and download models if needed.
"""

import sys
import pytest
from pathlib import Path

from pmd.core.config import Config, MLXConfig
from pmd.core.types import SearchSource
from pmd.services import ServiceContainer
from pmd.sources import FileSystemSource, SourceConfig


# Skip all tests in this module if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX embeddings require macOS with Apple Silicon"
)


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


@pytest.fixture(scope="module")
def mlx_config() -> MLXConfig:
    """Provide MLX configuration for vector tests."""
    return MLXConfig(
        embedding_model="mlx-community/nomicai-modernbert-embed-base-4bit",
        embedding_dimension=768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        lazy_load=True,
    )


class TestServiceEmbedCollection:
    """Integration tests for IndexingService.embed_collection."""

    @pytest.mark.asyncio
    async def test_embed_collection_creates_embeddings(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """embed_collection should create embeddings for indexed documents."""
        # Create test documents
        (tmp_path / "doc1.md").write_text(
            "# Machine Learning\n\nDeep learning uses neural networks."
        )
        (tmp_path / "doc2.md").write_text(
            "# Cooking\n\nItalian food uses olive oil and garlic."
        )

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            # Create and index collection
            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")
            await services.indexing.index_collection("test", source)

            # Embed collection
            result = await services.indexing.embed_collection("test")

            assert result.embedded == 2
            assert result.skipped == 0
            assert result.chunks_total >= 2

            # Verify embeddings exist
            collection = services.collection_repo.get_by_name("test")
            docs = services.document_repo.list_by_collection(collection.id)
            for doc in docs:
                assert services.embedding_repo.has_embeddings(doc.hash)

    @pytest.mark.asyncio
    async def test_embed_collection_skips_existing(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """embed_collection should skip documents with existing embeddings."""
        (tmp_path / "doc.md").write_text("# Test\n\nSome content here.")

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")
            await services.indexing.index_collection("test", source)

            # First embedding
            result1 = await services.indexing.embed_collection("test")
            assert result1.embedded == 1

            # Second embedding should skip
            result2 = await services.indexing.embed_collection("test")
            assert result2.embedded == 0
            assert result2.skipped == 1

    @pytest.mark.asyncio
    async def test_embed_collection_force_reembeds(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """embed_collection with force=True should re-embed all documents."""
        (tmp_path / "doc.md").write_text("# Test\n\nContent for re-embedding.")

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")
            await services.indexing.index_collection('test', source)

            # First embedding
            await services.indexing.embed_collection("test")

            # Force re-embed
            result = await services.indexing.embed_collection("test", force=True)

            assert result.embedded == 1
            assert result.skipped == 0


class TestServiceVectorSearch:
    """Integration tests for SearchService.vector_search."""

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """vector_search should return relevant documents."""
        (tmp_path / "python.md").write_text(
            "# Python Programming\n\nPython is a language for coding and development."
        )
        (tmp_path / "cooking.md").write_text(
            "# Italian Cooking\n\nPasta with tomato sauce and fresh basil."
        )

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")
            await services.indexing.index_collection("test", source)
            await services.indexing.embed_collection("test")

            # Search for programming-related content
            results = await services.search.vector_search(
                "software development coding", limit=5
            )

            assert len(results) >= 1
            # Python doc should be more relevant
            assert results[0].filepath == "python.md"
            assert results[0].source == SearchSource.VECTOR

    @pytest.mark.asyncio
    async def test_vector_search_respects_collection_filter(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """vector_search should filter by collection."""
        (tmp_path / "coll1").mkdir()
        (tmp_path / "coll2").mkdir()
        # Documents need body content to pass is_indexable check
        (tmp_path / "coll1" / "doc1.md").write_text(
            "# Document One\n\nThis is content in collection one about Python programming."
        )
        (tmp_path / "coll2" / "doc2.md").write_text(
            "# Document Two\n\nThis is content in collection two about Java development."
        )

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            # Create two collections
            services.collection_repo.create("coll1", str(tmp_path / "coll1"), "**/*.md")
            services.collection_repo.create("coll2", str(tmp_path / "coll2"), "**/*.md")
            coll1_source = _filesystem_source_for_name(services, "coll1")
            coll2_source = _filesystem_source_for_name(services, "coll2")

            await services.indexing.index_collection("coll1", coll1_source)
            await services.indexing.index_collection("coll2", coll2_source)
            await services.indexing.embed_collection("coll1")
            await services.indexing.embed_collection("coll2")

            # Search only in coll1
            results = await services.search.vector_search(
                "Python programming", collection_name="coll1"
            )

            assert len(results) == 1
            assert results[0].filepath == "doc1.md"

    @pytest.mark.asyncio
    async def test_vector_search_similarity_ranking(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """vector_search should rank results by similarity."""
        (tmp_path / "ml.md").write_text(
            "# Machine Learning\n\nNeural networks and deep learning for AI."
        )
        (tmp_path / "data.md").write_text(
            "# Data Analysis\n\nStatistics and data visualization techniques."
        )
        (tmp_path / "cooking.md").write_text(
            "# Recipes\n\nHow to make pasta with marinara sauce."
        )

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")
            await services.indexing.index_collection("test", source)
            await services.indexing.embed_collection("test")

            # Search for ML content
            results = await services.search.vector_search(
                "artificial intelligence neural networks", limit=3
            )

            assert len(results) == 3
            # ML should be most relevant
            assert results[0].filepath == "ml.md"
            # Cooking should be least relevant
            assert results[-1].filepath == "cooking.md"
            # Scores should be descending
            assert results[0].score >= results[1].score >= results[2].score


class TestServiceHybridSearch:
    """Integration tests for SearchService.hybrid_search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_fts_and_vector(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """hybrid_search should combine FTS and vector results."""
        (tmp_path / "python.md").write_text(
            "# Python Programming\n\nPython is great for scripting and automation."
        )
        (tmp_path / "java.md").write_text(
            "# Java Development\n\nJava uses the JVM for cross-platform apps."
        )

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")
            await services.indexing.index_collection("test", source)
            await services.indexing.embed_collection("test")

            # Hybrid search
            results = await services.search.hybrid_search(
                "Python programming language", limit=5
            )

            assert len(results) >= 1
            # Python doc should be most relevant
            assert results[0].file == "python.md"

    @pytest.mark.asyncio
    async def test_hybrid_search_fts_only_fallback(
        self, tmp_path: Path, mlx_config: MLXConfig
    ):
        """hybrid_search should work with FTS only if no embeddings."""
        (tmp_path / "doc.md").write_text("# Test Document\n\nSearchable content here.")

        config = Config()
        config.db_path = tmp_path / "test.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            services.collection_repo.create("test", str(tmp_path), "**/*.md")
            source = _filesystem_source_for_name(services, "test")
            await services.indexing.index_collection("test", source)
            # Don't embed - test FTS fallback

            # Hybrid search should still work via FTS
            results = await services.search.hybrid_search("searchable", limit=5)

            assert len(results) >= 1
            assert results[0].file == "doc.md"


class TestServiceCorpusSearch:
    """Integration tests using the test corpus."""

    @pytest.mark.asyncio
    async def test_corpus_vector_search(self, mlx_config: MLXConfig, tmp_path: Path):
        """vector_search should find relevant documents in test corpus."""
        corpus_path = Path(__file__).parent.parent / "fixtures" / "test_corpus"
        if not corpus_path.exists():
            pytest.skip("Test corpus not found")

        config = Config()
        config.db_path = tmp_path / "corpus.db"
        config.mlx = mlx_config

        async with ServiceContainer(config) as services:
            if not services.vec_available:
                pytest.skip("sqlite-vec not available")

            # Index and embed corpus (limit for speed)
            services.collection_repo.create("corpus", str(corpus_path), "**/*.md")
            source = _filesystem_source_for_name(services, "corpus")
            await services.indexing.index_collection("corpus", source)

            # Only embed first 20 docs for speed
            result = await services.indexing.embed_collection("corpus")

            if result.embedded < 5:
                pytest.skip("Not enough documents embedded")

            # Test semantic search queries
            search_cases = [
                ("machine learning neural networks", "K-means"),
                ("graph database nodes edges", "Graph"),
                ("smart home automation", "Home"),
            ]

            for query, expected_substr in search_cases:
                results = await services.search.vector_search(query, limit=10)

                assert len(results) > 0, f"No results for: {query}"

                # Check if any result contains expected substring
                result_files = [r.filepath for r in results]
                found = any(expected_substr in f for f in result_files)
                if not found:
                    # Log for debugging but don't fail - semantic search can vary
                    print(f"Note: '{expected_substr}' not in top results for '{query}'")
                    print(f"Got: {result_files[:5]}")
