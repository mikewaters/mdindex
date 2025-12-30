"""Integration tests for hybrid search against the test corpus.

These tests index the real test corpus and perform hybrid searches
to verify that combining FTS and vector search improves retrieval quality.

The tests focus on:
- Hybrid search finding documents with both keyword and semantic matching
- FTS-only matches (exact keywords) being included
- Vector-only matches (semantic similarity) being included
- Ranking quality when both signals agree
- Score attribution from both sources

Tests require macOS with Apple Silicon and download models if needed.
"""

import sys
import pytest
from pathlib import Path

from pmd.core.config import Config, MLXConfig
from pmd.store.database import Database
from pmd.store.collections import CollectionRepository
from pmd.store.documents import DocumentRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.search import FTS5SearchRepository
from pmd.llm.embeddings import EmbeddingGenerator
from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig


# Skip all tests in this module if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX embeddings require macOS with Apple Silicon"
)


# Path to test corpus
CORPUS_PATH = Path(__file__).parent.parent / "fixtures" / "test_corpus"


@pytest.fixture(scope="module")
def mlx_config() -> MLXConfig:
    """Provide MLX configuration for corpus tests."""
    return MLXConfig(
        embedding_model="mlx-community/nomicai-modernbert-embed-base-4bit",
        embedding_dimension=768,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        lazy_load=True,
    )


@pytest.fixture(scope="module")
def mlx_provider(mlx_config: MLXConfig):
    """Provide an MLX provider instance.

    Module-scoped to avoid reloading the model for each test.
    """
    from pmd.llm.mlx_provider import MLXProvider

    provider = MLXProvider(mlx_config)
    yield provider
    provider.unload_all()


@pytest.fixture(scope="module")
def indexed_corpus(mlx_provider, mlx_config, tmp_path_factory):
    """Index the test corpus and return database components.

    This fixture indexes all documents in the test corpus with both
    FTS and vector embeddings, creating a searchable database.

    Module-scoped to avoid re-indexing for each test.
    """
    # Create temp directory for this module
    tmp_path = tmp_path_factory.mktemp("hybrid_test")
    db_path = tmp_path / "corpus.db"

    # Set up database with embedding dimension from config
    db = Database(db_path, embedding_dimension=mlx_config.embedding_dimension)
    db.connect()

    if not db.vec_available:
        pytest.skip("sqlite-vec extension not available")

    # Create repositories
    collection_repo = CollectionRepository(db)
    document_repo = DocumentRepository(db)
    embedding_repo = EmbeddingRepository(db)
    fts_repo = FTS5SearchRepository(db)

    # Create collection for corpus
    collection = collection_repo.create(
        "test_corpus",
        str(CORPUS_PATH),
        "**/*.md"
    )

    # Create config for embedding generator
    config = Config()
    config.db_path = db_path
    config.chunk.max_bytes = 2000
    config.chunk.min_chunk_size = 100

    embedding_generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)

    # Index documents from corpus
    import asyncio

    async def index_corpus():
        indexed_count = 0
        corpus_files = list(CORPUS_PATH.glob("**/*.md"))

        for filepath in corpus_files:
            try:
                content = filepath.read_text(encoding="utf-8")
                if not content.strip():
                    continue

                # Extract title from first line
                lines = content.split("\n")
                title = lines[0].lstrip("#").strip() if lines else filepath.stem

                # Add document
                doc_result, _ = document_repo.add_or_update(
                    collection.id,
                    filepath.name,
                    title,
                    content,
                )

                # Get document ID for FTS indexing
                cursor = db.execute(
                    "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
                    (collection.id, filepath.name),
                )
                row = cursor.fetchone()
                if row:
                    doc_id = row["id"]
                    fts_repo.index_document(doc_id, filepath.name, content)

                # Generate embeddings
                await embedding_generator.embed_document(doc_result.hash, content)
                indexed_count += 1

            except Exception as e:
                # Skip files that fail to index
                print(f"Failed to index {filepath.name}: {e}")
                continue

        return indexed_count

    indexed_count = asyncio.new_event_loop().run_until_complete(index_corpus())
    print(f"\nIndexed {indexed_count} documents from corpus")

    yield {
        "db": db,
        "collection": collection,
        "document_repo": document_repo,
        "embedding_repo": embedding_repo,
        "fts_repo": fts_repo,
        "embedding_generator": embedding_generator,
        "indexed_count": indexed_count,
    }

    # Cleanup
    db.close()


class TestHybridSearchBasics:
    """Basic tests for hybrid search functionality."""

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, indexed_corpus):
        """Hybrid search should return results for valid queries."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search("machine learning algorithm", limit=10)

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_hybrid_has_both_score_types(self, indexed_corpus):
        """Hybrid results should have both FTS and vector scores."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search("graph database neo4j", limit=10)

        # At least one result should have both signals
        has_both = any(
            r.fts_score is not None and r.fts_score > 0 and
            r.vec_score is not None and r.vec_score > 0
            for r in results
        )
        assert has_both, "Expected at least one result with both FTS and vector scores"


class TestHybridSearchAccuracy:
    """Tests for hybrid search finding relevant documents."""

    @pytest.mark.asyncio
    async def test_find_kmeans_document(self, indexed_corpus):
        """Should find K-means clustering doc with ML query."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search(
            "unsupervised clustering algorithm centroids",
            limit=10
        )

        result_files = [r.file for r in results]
        assert "K-means clustering.md" in result_files

    @pytest.mark.asyncio
    async def test_find_graph_databases_document(self, indexed_corpus):
        """Should find Graph Databases doc with database query."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search("weaviate neo4j graph database", limit=10)

        result_files = [r.file for r in results]
        assert "Graph Databases.md" in result_files

    @pytest.mark.asyncio
    async def test_find_knn_document(self, indexed_corpus):
        """Should find k-NN doc when searching for neighbor algorithms."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Avoid hyphens which cause FTS5 parsing issues
        results = await pipeline.search(
            "nearest neighbors classification distance supervised",
            limit=10
        )

        result_files = [r.file for r in results]
        assert "k-Nearest Neighbors (k-NN) .md" in result_files

    @pytest.mark.asyncio
    async def test_find_web_scraping_document(self, indexed_corpus):
        """Should find web ingestion doc with scraping query."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search("firecrawl jina web scraping", limit=10)

        result_files = [r.file for r in results]
        assert "Ingestion of Web content using LLMs.md" in result_files

    @pytest.mark.asyncio
    async def test_find_keyboard_lubing_document(self, indexed_corpus):
        """Should find keyboard switch doc with lubing query."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search(
            "keyboard switch lubrication gateron cherry",
            limit=10
        )

        result_files = [r.file for r in results]
        assert "Keyboard Switch lubricating guide.md" in result_files


class TestHybridFTSStrength:
    """Tests where FTS exact matching should help."""

    @pytest.mark.asyncio
    async def test_exact_term_surrealdb(self, indexed_corpus):
        """Exact term 'SurrealDB' should find Graph Databases doc via FTS."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Very specific term that FTS should match exactly
        results = await pipeline.search("SurrealDB", limit=10)

        result_files = [r.file for r in results]
        assert "Graph Databases.md" in result_files

    @pytest.mark.asyncio
    async def test_exact_term_tigergraph(self, indexed_corpus):
        """Exact term 'TigerGraph' should find Graph Databases doc."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search("TigerGraph pyTigerGraph", limit=10)

        result_files = [r.file for r in results]
        assert "Graph Databases.md" in result_files

    @pytest.mark.asyncio
    async def test_exact_term_zealios(self, indexed_corpus):
        """Exact term 'Zealios' should find keyboard lubing doc."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search("Zealios Gateron switch", limit=10)

        result_files = [r.file for r in results]
        assert "Keyboard Switch lubricating guide.md" in result_files


class TestHybridSemanticStrength:
    """Tests where vector semantic matching should help."""

    @pytest.mark.asyncio
    async def test_semantic_partitioning_data(self, indexed_corpus):
        """Semantic query about partitioning should find K-means doc."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Query about clustering concepts - should find K-means
        results = await pipeline.search(
            "unsupervised machine learning partitioning data clusters",
            limit=10
        )

        result_files = [r.file for r in results]
        # Should find clustering document
        assert "K-means clustering.md" in result_files

    @pytest.mark.asyncio
    async def test_semantic_relationships_storage(self, indexed_corpus):
        """Semantic query about relationships should find graph DB doc."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Query about concept without using exact terms
        results = await pipeline.search(
            "storing connections between entities nodes edges",
            limit=10
        )

        result_files = [r.file for r in results]
        assert "Graph Databases.md" in result_files

    @pytest.mark.asyncio
    async def test_semantic_autism_perception(self, indexed_corpus):
        """Semantic query about perception should find autism doc."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search(
            "sensory overload heightened sensitivity perception",
            limit=10
        )

        result_files = [r.file for r in results]
        assert "Intense World Theory Autism.md" in result_files


class TestHybridRanking:
    """Tests for ranking quality in hybrid search."""

    @pytest.mark.asyncio
    async def test_relevant_ranked_higher(self, indexed_corpus):
        """More relevant documents should rank higher."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Query specifically about clustering (avoid hyphens for FTS5)
        results = await pipeline.search(
            "clustering euclidean distance centroids algorithm",
            limit=20
        )

        # Results should be sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # K-means should be in top 3
        top_files = [r.file for r in results[:3]]
        assert "K-means clustering.md" in top_files

    @pytest.mark.asyncio
    async def test_high_confidence_when_both_agree(self, indexed_corpus):
        """Results with both FTS and vector matches should rank well."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Query that should match both lexically and semantically
        results = await pipeline.search("graph database neo4j cypher", limit=10)

        # The top result should have both signals
        if results:
            top_result = results[0]
            # At least one of the top results should have strong signals from both
            top_results = results[:3]
            has_strong_hybrid = any(
                r.fts_score is not None and r.fts_score > 0 and
                r.vec_score is not None and r.vec_score > 0
                for r in top_results
            )
            assert has_strong_hybrid


class TestHybridCollectionFiltering:
    """Tests for collection-scoped hybrid search."""

    @pytest.mark.asyncio
    async def test_respects_collection_filter(self, indexed_corpus):
        """Hybrid search should respect collection_id filter."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]
        collection = indexed_corpus["collection"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Search with collection filter
        results = await pipeline.search(
            "machine learning",
            limit=10,
            collection_id=collection.id,
        )

        # Should return results (we only have one collection)
        assert len(results) > 0


class TestHybridScoreNormalization:
    """Tests for score normalization in hybrid search."""

    @pytest.mark.asyncio
    async def test_scores_normalized_when_enabled(self, indexed_corpus):
        """Scores should be normalized to 0-1 range when enabled."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
            normalize_final_scores=True,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search("database query language", limit=10)

        # All scores should be in 0-1 range when normalized
        for result in results:
            assert 0 <= result.score <= 1.0

    @pytest.mark.asyncio
    async def test_min_score_filter(self, indexed_corpus):
        """Results below min_score should be filtered out."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
            normalize_final_scores=True,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # High min_score should filter out many results
        results = await pipeline.search(
            "random unrelated gibberish query",
            limit=10,
            min_score=0.9,
        )

        # Either no results or all above threshold
        for result in results:
            assert result.score >= 0.9


class TestHybridEdgeCases:
    """Edge case tests for hybrid search."""

    @pytest.mark.asyncio
    async def test_single_word_query(self, indexed_corpus):
        """Single word query should work correctly."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Single word query
        results = await pipeline.search("database", limit=10)

        # Should return results
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_very_long_query(self, indexed_corpus):
        """Very long query should still work."""
        fts_repo = indexed_corpus["fts_repo"]
        embedding_generator = indexed_corpus["embedding_generator"]

        config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=config,
            embedding_generator=embedding_generator,
        )

        # Very long query with many terms
        long_query = " ".join([
            "machine learning clustering algorithm unsupervised",
            "graph database neo4j weaviate surreal tigergraph",
            "keyboard switch mechanical lubrication gateron",
            "web scraping firecrawl jina extract content",
        ])

        results = await pipeline.search(long_query, limit=10)

        # Should return results without error
        assert isinstance(results, list)
        assert len(results) > 0
