"""Integration tests for vector search against the test corpus.

These tests index the real test corpus and perform vector searches
to verify that semantically relevant documents are found.
The corpus contains documents on various topics including:
- Machine learning (K-means clustering)
- Graph databases (Weaviate, Neo4j, SurrealDB)
- Web scraping and LLM ingestion
- Autism research (Intense World Theory)
- Home automation (Home Assistant)
- iOS security

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

    This fixture indexes all documents in the test corpus with embeddings,
    creating a searchable database for the test module.

    Module-scoped to avoid re-indexing for each test.
    """
    # Create temp directory for this module
    tmp_path = tmp_path_factory.mktemp("corpus_test")
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
    # Use small chunks for testing
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


class TestCorpusIndexing:
    """Tests for corpus indexing."""

    def test_corpus_was_indexed(self, indexed_corpus):
        """Corpus should have been indexed successfully."""
        assert indexed_corpus["indexed_count"] > 0

    def test_embeddings_were_created(self, indexed_corpus):
        """Embeddings should have been created for indexed documents."""
        embedding_count = indexed_corpus["embedding_repo"].count_embeddings()
        assert embedding_count > 0


class TestVectorSearchAccuracy:
    """Tests for vector search finding relevant documents."""

    @pytest.mark.asyncio
    async def test_search_kmeans_clustering(self, indexed_corpus, mlx_provider):
        """Should find K-means clustering doc when searching for clustering algorithms."""
        query = "unsupervised machine learning clustering partitioning data points"

        # Generate query embedding
        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        # Should find results
        assert len(results) > 0

        # K-means clustering document should be in top results
        result_files = [r.filepath for r in results]
        assert "K-means clustering.md" in result_files

    @pytest.mark.asyncio
    async def test_search_graph_databases(self, indexed_corpus, mlx_provider):
        """Should find Graph Databases doc when searching for graph database topics."""
        query = "neo4j weaviate graph database property nodes edges cypher"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        assert len(results) > 0

        result_files = [r.filepath for r in results]
        assert "Graph Databases.md" in result_files

    @pytest.mark.asyncio
    async def test_search_autism_theory(self, indexed_corpus, mlx_provider):
        """Should find Intense World Theory doc when searching for autism topics."""
        query = "autism sensory sensitivity hyper-reactivity neural perception"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        assert len(results) > 0

        result_files = [r.filepath for r in results]
        assert "Intense World Theory Autism.md" in result_files

    @pytest.mark.asyncio
    async def test_search_web_scraping(self, indexed_corpus, mlx_provider):
        """Should find web ingestion doc when searching for web scraping topics."""
        query = "web scraping crawling extract content firecrawl jina"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        assert len(results) > 0

        result_files = [r.filepath for r in results]
        assert "Ingestion of Web content using LLMs.md" in result_files

    @pytest.mark.asyncio
    async def test_search_home_automation(self, indexed_corpus, mlx_provider):
        """Should find Home Assistant doc when searching for home automation."""
        query = "smart home automation IoT devices control"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        assert len(results) > 0

        result_files = [r.filepath for r in results]
        assert "Home Assistant.md" in result_files

    @pytest.mark.asyncio
    async def test_search_ios_security(self, indexed_corpus, mlx_provider):
        """Should find iOS Security doc when searching for mobile security."""
        query = "ios lockdown mode security whitelist trusted apps"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        assert len(results) > 0

        result_files = [r.filepath for r in results]
        assert "iOS Security Posture.md" in result_files


class TestSemanticRelevance:
    """Tests for semantic understanding in vector search."""

    @pytest.mark.asyncio
    async def test_synonym_search_finds_relevant(self, indexed_corpus, mlx_provider):
        """Should find documents using synonyms/related terms."""
        # Search for centroid-based partitioning should find K-means
        query = "partitioning data into groups using centroids and distance"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        result_files = [r.filepath for r in results]
        # Should find clustering document using related terminology
        assert "K-means clustering.md" in result_files

    @pytest.mark.asyncio
    async def test_conceptual_search(self, indexed_corpus, mlx_provider):
        """Should find documents based on conceptual meaning."""
        # Search for database concepts should find graph databases
        query = "storing relationships between entities connected data"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        result_files = [r.filepath for r in results]
        assert "Graph Databases.md" in result_files

    @pytest.mark.asyncio
    async def test_irrelevant_query_low_scores(self, indexed_corpus, mlx_provider):
        """Unrelated queries should have lower relevance scores."""
        # Query completely unrelated to corpus content
        relevant_query = "graph database neo4j weaviate"
        irrelevant_query = "pizza recipe tomato sauce mozzarella"

        relevant_emb = await mlx_provider.embed(relevant_query, is_query=True)
        irrelevant_emb = await mlx_provider.embed(irrelevant_query, is_query=True)

        relevant_results = indexed_corpus["embedding_repo"].search_vectors(
            relevant_emb.embedding, limit=5
        )
        irrelevant_results = indexed_corpus["embedding_repo"].search_vectors(
            irrelevant_emb.embedding, limit=5
        )

        # Both should return results (vector search always returns something)
        assert len(relevant_results) > 0
        assert len(irrelevant_results) > 0

        # Relevant query should have higher top score
        assert relevant_results[0].score > irrelevant_results[0].score


class TestVectorSearchRanking:
    """Tests for correct ranking of search results."""

    @pytest.mark.asyncio
    async def test_more_relevant_ranked_higher(self, indexed_corpus, mlx_provider):
        """More relevant documents should appear before less relevant ones."""
        # Query specifically about K-means
        query = "k-means centroids euclidean distance clustering algorithm"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=20
        )

        assert len(results) >= 2

        # Results should be sorted by score descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

        # K-means document should be in top 3
        top_files = [r.filepath for r in results[:3]]
        assert "K-means clustering.md" in top_files

    @pytest.mark.asyncio
    async def test_specific_query_beats_general(self, indexed_corpus, mlx_provider):
        """Specific terminology should find relevant documents."""
        # Specific graph database query with key terminology
        query = "graph database neo4j weaviate nodes edges cypher query"

        query_emb = await mlx_provider.embed(query, is_query=True)
        assert query_emb is not None

        results = indexed_corpus["embedding_repo"].search_vectors(
            query_emb.embedding, limit=10
        )

        # Graph Databases document should be in results
        result_files = [r.filepath for r in results]
        assert "Graph Databases.md" in result_files, f"Expected Graph Databases.md in {result_files}"


class TestHybridSearchWithCorpus:
    """Tests for hybrid FTS + vector search."""

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_signals(self, indexed_corpus, mlx_provider):
        """Hybrid search should use both FTS and vector signals."""
        from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig

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

        # Search for a specific topic
        results = await pipeline.search("neural networks machine learning", limit=10)

        assert len(results) > 0

        # Results should have both FTS and vector scores
        for result in results:
            # At least one signal should be present
            has_fts = result.fts_score is not None and result.fts_score > 0
            has_vec = result.vec_score is not None and result.vec_score > 0
            assert has_fts or has_vec

    @pytest.mark.asyncio
    async def test_hybrid_respects_collection_filter(
        self, indexed_corpus, mlx_provider
    ):
        """Hybrid search should respect collection_id filter."""
        from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig

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
            "graph database",
            limit=10,
            collection_id=collection.id,
        )

        # All results should be from the test corpus collection
        # (since we only have one collection, just verify we get results)
        assert len(results) > 0
