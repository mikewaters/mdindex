"""Integration tests for vector search using MLX.

These tests exercise the full vector search pipeline including:
- Embedding generation with MLX
- Vector storage in sqlite-vec
- Vector similarity search
- Hybrid search pipeline
- LLM provider lifecycle (including close())

Tests require macOS with Apple Silicon and download models if needed.
"""

import sys
import pytest
from pathlib import Path

from pmd.core.config import Config, MLXConfig
from pmd.core.types import EmbeddingResult, SearchSource
from pmd.store.database import Database
from pmd.store.collections import CollectionRepository
from pmd.store.documents import DocumentRepository
from pmd.store.embeddings import EmbeddingRepository
from pmd.store.search import FTS5SearchRepository


# Skip all tests in this module if not on macOS
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="MLX embeddings require macOS with Apple Silicon"
)


def get_document_id(db: Database, collection_id: int, path: str) -> int:
    """Get document ID from database by path.

    Args:
        db: Database instance.
        collection_id: Collection ID.
        path: Document path.

    Returns:
        Document ID.
    """
    cursor = db.execute(
        "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
        (collection_id, path),
    )
    row = cursor.fetchone()
    if not row:
        raise ValueError(f"Document not found: {path}")
    return row["id"]


@pytest.fixture(scope="module")
def mlx_config() -> MLXConfig:
    """Provide MLX configuration for vector search tests."""
    return MLXConfig(
        embedding_model="mlx-community/multilingual-e5-small-mlx",
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
    # Clean up - unload models to free memory
    provider.unload_all()


class TestVectorSearchBasics:
    """Tests for basic vector search functionality."""

    @pytest.fixture
    def vector_db(self, tmp_path: Path) -> Database:
        """Provide a connected database with vector support."""
        db_path = tmp_path / "vector_test.db"
        database = Database(db_path)
        database.connect()
        yield database
        database.close()

    @pytest.fixture
    def embedding_repo(self, vector_db: Database) -> EmbeddingRepository:
        """Provide an EmbeddingRepository instance."""
        return EmbeddingRepository(vector_db)

    @pytest.fixture
    def fts_repo(self, vector_db: Database) -> FTS5SearchRepository:
        """Provide an FTS5 search repository."""
        return FTS5SearchRepository(vector_db)

    @pytest.fixture
    def collection_repo(self, vector_db: Database) -> CollectionRepository:
        """Provide a CollectionRepository instance."""
        return CollectionRepository(vector_db)

    @pytest.fixture
    def document_repo(self, vector_db: Database) -> DocumentRepository:
        """Provide a DocumentRepository instance."""
        return DocumentRepository(vector_db)

    @pytest.mark.asyncio
    async def test_vector_search_returns_results(
        self,
        mlx_provider,
        vector_db: Database,
        embedding_repo: EmbeddingRepository,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        tmp_path: Path,
    ):
        """Vector search should return relevant documents."""
        #if not vector_db.vec_available:
        #    pytest.skip("sqlite-vec extension not available")

        # Create collection and documents
        collection = collection_repo.create("test", str(tmp_path), "**/*.md")

        doc1, _ = document_repo.add_or_update(
            collection.id,
            "python.md",
            "Python Programming",
            "Python is a programming language known for its simplicity and readability.",
        )

        doc2, _ = document_repo.add_or_update(
            collection.id,
            "cooking.md",
            "Italian Cooking",
            "Italian cooking uses olive oil, garlic, and fresh herbs.",
        )

        # Generate and store embeddings
        emb1 = await mlx_provider.embed(
            "Python is a programming language known for its simplicity."
        )
        emb2 = await mlx_provider.embed(
            "Italian cooking uses olive oil, garlic, and fresh herbs."
        )

        assert emb1 is not None
        assert emb2 is not None

        embedding_repo.store_embedding(
            doc1.hash, 0, 0, emb1.embedding, mlx_provider.get_default_embedding_model()
        )
        embedding_repo.store_embedding(
            doc2.hash, 0, 0, emb2.embedding, mlx_provider.get_default_embedding_model()
        )

        # Search for programming-related query
        query_emb = await mlx_provider.embed("coding and software development", is_query=True)
        assert query_emb is not None

        results = embedding_repo.search_vectors(query_emb.embedding, limit=5)

        assert len(results) >= 1
        # Python doc should be more relevant to coding query
        assert any(r.filepath == "python.md" for r in results)

    @pytest.mark.asyncio
    async def test_vector_search_with_collection_filter(
        self,
        mlx_provider,
        vector_db: Database,
        embedding_repo: EmbeddingRepository,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        tmp_path: Path,
    ):
        """Vector search should respect collection_id filter."""
        #if not vector_db.vec_available:
        #    pytest.skip("sqlite-vec extension not available")

        # Create two collections
        coll1 = collection_repo.create("coll1", str(tmp_path), "**/*.md")
        coll2 = collection_repo.create("coll2", str(tmp_path / "coll2"), "**/*.md")

        # Add document to collection 1
        doc1, _ = document_repo.add_or_update(
            coll1.id,
            "doc1.md",
            "Document One",
            "This is a document about testing.",
        )

        # Add document to collection 2
        doc2, _ = document_repo.add_or_update(
            coll2.id,
            "doc2.md",
            "Document Two",
            "This is another document about testing.",
        )

        # Generate embeddings
        emb1 = await mlx_provider.embed("This is a document about testing.")
        emb2 = await mlx_provider.embed("This is another document about testing.")
        assert emb1 is not None
        assert emb2 is not None

        embedding_repo.store_embedding(
            doc1.hash, 0, 0, emb1.embedding, mlx_provider.get_default_embedding_model()
        )
        embedding_repo.store_embedding(
            doc2.hash, 0, 0, emb2.embedding, mlx_provider.get_default_embedding_model()
        )

        # Search with collection filter
        query_emb = await mlx_provider.embed("testing documents", is_query=True)
        assert query_emb is not None

        # Search only collection 1
        results = embedding_repo.search_vectors(
            query_emb.embedding, limit=5, collection_id=coll1.id
        )

        # Should only find document from collection 1
        assert len(results) == 1
        assert results[0].filepath == "doc1.md"

    @pytest.mark.asyncio
    async def test_vector_search_similarity_ranking(
        self,
        mlx_provider,
        vector_db: Database,
        embedding_repo: EmbeddingRepository,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        tmp_path: Path,
    ):
        """More relevant documents should have higher similarity scores."""
        #if not vector_db.vec_available:
        #    pytest.skip("sqlite-vec extension not available")

        collection = collection_repo.create("test", str(tmp_path), "**/*.md")

        # Create documents with varying relevance to "machine learning"
        docs = [
            ("ml.md", "Machine Learning", "Deep learning and neural networks for AI."),
            ("data.md", "Data Science", "Statistics and data analysis techniques."),
            ("cooking.md", "Recipes", "How to make pasta with tomato sauce."),
        ]

        for path, title, content in docs:
            doc, _ = document_repo.add_or_update(collection.id, path, title, content)
            emb = await mlx_provider.embed(content)
            assert emb is not None
            embedding_repo.store_embedding(
                doc.hash, 0, 0, emb.embedding, mlx_provider.get_default_embedding_model()
            )

        # Search for ML-related content
        query_emb = await mlx_provider.embed(
            "artificial intelligence machine learning", is_query=True
        )
        assert query_emb is not None

        results = embedding_repo.search_vectors(query_emb.embedding, limit=3)

        assert len(results) == 3
        # ML document should be first (most relevant)
        assert results[0].filepath == "ml.md"
        # Cooking should be last (least relevant)
        assert results[-1].filepath == "cooking.md"
        # Scores should be in descending order
        assert results[0].score >= results[1].score >= results[2].score

    @pytest.mark.asyncio
    async def test_vector_search_with_full_corpus(
        self,
        mlx_provider,
        vector_db: Database,
        embedding_repo: EmbeddingRepository,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        tmp_path: Path,
    ):
        """Vector search should find relevant documents from the full test corpus."""
        #if not vector_db.vec_available:
        #    pytest.skip("sqlite-vec extension not available")

        # Path to test corpus
        corpus_path = Path(__file__).parent.parent / "fixtures" / "test_corpus"
        if not corpus_path.exists():
            pytest.skip("Test corpus not found")

        # Create collection for corpus
        collection = collection_repo.create("corpus", str(corpus_path), "**/*.md")

        # Index a selection of documents from the corpus
        indexed_docs = []
        corpus_files = list(corpus_path.glob("**/*.md"))[:50]  # Limit to 50 for speed

        for filepath in corpus_files:
            try:
                content = filepath.read_text(encoding="utf-8")
                if not content.strip() or len(content) < 50:
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

                # Generate and store embedding (use first 2000 chars for speed)
                emb = await mlx_provider.embed(content[:2000])
                if emb is not None:
                    embedding_repo.store_embedding(
                        doc_result.hash,
                        0,
                        0,
                        emb.embedding,
                        mlx_provider.get_default_embedding_model(),
                    )
                    indexed_docs.append(filepath.name)

            except Exception:
                continue

        assert len(indexed_docs) > 10, "Should have indexed at least 10 documents"

        # Test searches for known corpus content
        search_cases = [
            # (query, expected_file_substring)
            ("unsupervised machine learning clustering centroids", "K-means"),
            ("graph database neo4j weaviate property nodes", "Graph"),
            ("autism sensory hyper-reactivity neural", "Autism"),
            ("web scraping firecrawl jina content extraction", "Ingestion"),
            ("smart home automation IoT devices", "Home"),
        ]

        for query, expected_substr in search_cases:
            query_emb = await mlx_provider.embed(query, is_query=True)
            assert query_emb is not None

            results = embedding_repo.search_vectors(query_emb.embedding, limit=10)
            print(results)

            # Should find results
            assert len(results) > 0, f"No results for query: {query}"

            # At least one result should contain the expected substring
            result_files = [r.filepath for r in results]
            found = any(expected_substr in f for f in result_files)
            assert found, (
                f"Expected file containing '{expected_substr}' not found in results "
                f"for query '{query}'. Got: {result_files[:5]}"
            )


class TestEmbeddingGenerator:
    """Tests for the EmbeddingGenerator class."""

    @pytest.fixture
    def vector_db(self, tmp_path: Path) -> Database:
        """Provide a connected database with vector support."""
        db_path = tmp_path / "embedding_gen_test.db"
        database = Database(db_path)
        database.connect()
        yield database
        database.close()

    @pytest.fixture
    def config(self, vector_db: Database, tmp_path: Path) -> Config:
        """Provide config pointing to test database."""
        cfg = Config()
        cfg.db_path = tmp_path / "embedding_gen_test.db"
        return cfg

    @pytest.fixture
    def embedding_repo(self, vector_db: Database) -> EmbeddingRepository:
        """Provide an EmbeddingRepository instance."""
        return EmbeddingRepository(vector_db)

    @pytest.mark.asyncio
    async def test_embed_query_returns_vector(
        self, mlx_provider, embedding_repo: EmbeddingRepository, config: Config
    ):
        """embed_query should return a valid embedding vector."""
        from pmd.llm.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)
        embedding = await generator.embed_query("test query for embedding")

        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 384  # e5-small dimension
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_document_stores_embeddings(
        self,
        mlx_provider,
        embedding_repo: EmbeddingRepository,
        config: Config,
        vector_db: Database,
    ):
        """embed_document should store embeddings in the repository."""
        from pmd.llm.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)

        # Create a test hash and content
        test_hash = "a" * 64  # Fake SHA256 hash
        content = "This is test content for embedding generation."

        chunks_embedded = await generator.embed_document(test_hash, content)

        assert chunks_embedded > 0
        assert embedding_repo.has_embeddings(test_hash)

    @pytest.mark.asyncio
    async def test_embed_document_skips_existing(
        self,
        mlx_provider,
        embedding_repo: EmbeddingRepository,
        config: Config,
    ):
        """embed_document should skip if embeddings already exist."""
        from pmd.llm.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)

        test_hash = "b" * 64
        content = "Test content for skipping test."

        # First embedding
        first_count = await generator.embed_document(test_hash, content)
        assert first_count > 0

        # Second embedding should skip (return 0)
        second_count = await generator.embed_document(test_hash, content)
        assert second_count == 0

    @pytest.mark.asyncio
    async def test_embed_document_force_regeneration(
        self,
        mlx_provider,
        embedding_repo: EmbeddingRepository,
        config: Config,
    ):
        """embed_document with force=True should regenerate embeddings."""
        from pmd.llm.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)

        test_hash = "c" * 64
        content = "Content for force regeneration test."

        # First embedding
        first_count = await generator.embed_document(test_hash, content)
        assert first_count > 0

        # Second embedding with force=True should re-embed
        second_count = await generator.embed_document(test_hash, content, force=True)
        assert second_count > 0


class TestHybridSearchPipeline:
    """Tests for the HybridSearchPipeline with vector search."""

    @pytest.fixture
    def vector_db(self, tmp_path: Path) -> Database:
        """Provide a connected database with vector support."""
        db_path = tmp_path / "hybrid_test.db"
        database = Database(db_path)
        database.connect()
        yield database
        database.close()

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Provide config pointing to test database."""
        cfg = Config()
        cfg.db_path = tmp_path / "hybrid_test.db"
        return cfg

    @pytest.fixture
    def embedding_repo(self, vector_db: Database) -> EmbeddingRepository:
        """Provide an EmbeddingRepository instance."""
        return EmbeddingRepository(vector_db)

    @pytest.fixture
    def fts_repo(self, vector_db: Database) -> FTS5SearchRepository:
        """Provide an FTS5 search repository."""
        return FTS5SearchRepository(vector_db)

    @pytest.fixture
    def collection_repo(self, vector_db: Database) -> CollectionRepository:
        """Provide a CollectionRepository instance."""
        return CollectionRepository(vector_db)

    @pytest.fixture
    def document_repo(self, vector_db: Database) -> DocumentRepository:
        """Provide a DocumentRepository instance."""
        return DocumentRepository(vector_db)

    @pytest.mark.asyncio
    async def test_hybrid_search_combines_fts_and_vector(
        self,
        mlx_provider,
        vector_db: Database,
        config: Config,
        embedding_repo: EmbeddingRepository,
        fts_repo: FTS5SearchRepository,
        collection_repo: CollectionRepository,
        document_repo: DocumentRepository,
        tmp_path: Path,
    ):
        """Hybrid search should combine FTS and vector results."""
        if not vector_db.vec_available:
            pytest.skip("sqlite-vec extension not available")

        from pmd.llm.embeddings import EmbeddingGenerator
        from pmd.search.pipeline import HybridSearchPipeline, SearchPipelineConfig

        collection = collection_repo.create("hybrid", str(tmp_path), "**/*.md")

        # Create documents
        docs = [
            ("python.md", "Python Guide", "Python programming with classes and functions."),
            ("java.md", "Java Guide", "Java programming with objects and methods."),
        ]

        for path, title, content in docs:
            doc, _ = document_repo.add_or_update(collection.id, path, title, content)
            # Get document ID for FTS indexing
            doc_id = get_document_id(vector_db, collection.id, path)
            # Index in FTS
            fts_repo.index_document(doc_id, path, content)
            # Generate and store embedding
            emb = await mlx_provider.embed(content)
            assert emb is not None
            embedding_repo.store_embedding(
                doc.hash, 0, 0, emb.embedding, mlx_provider.get_default_embedding_model()
            )

        # Create embedding generator and pipeline
        embedding_generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)

        pipeline_config = SearchPipelineConfig(
            enable_query_expansion=False,
            enable_reranking=False,
        )

        pipeline = HybridSearchPipeline(
            fts_repo,
            config=pipeline_config,
            embedding_generator=embedding_generator,
        )

        # Search
        results = await pipeline.search("Python programming language", limit=5)

        assert len(results) >= 1
        # Python doc should be most relevant
        # RankedResult uses 'file' instead of 'filepath'
        assert results[0].file == "python.md"


class TestProviderLifecycle:
    """Tests for LLM provider lifecycle including close()."""

    @pytest.mark.asyncio
    async def test_provider_close_unloads_models(self, mlx_config: MLXConfig):
        """close() should unload all models to free memory."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        # Load embedding model by generating an embedding
        result = await provider.embed("Test text")
        assert result is not None

        # Provider should have loaded model
        assert provider._embedding_model_loaded

        # Close provider
        await provider.close()

        # Models should be unloaded
        assert not provider._embedding_model_loaded
        assert provider._embedding_model is None
        assert provider._embedding_tokenizer is None

    @pytest.mark.asyncio
    async def test_provider_close_can_be_called_multiple_times(
        self, mlx_config: MLXConfig
    ):
        """close() should be safe to call multiple times."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        # Close multiple times - should not raise
        await provider.close()
        await provider.close()
        await provider.close()

    @pytest.mark.asyncio
    async def test_provider_usable_after_reload(self, mlx_config: MLXConfig):
        """Provider should be usable again after close and reload."""
        from pmd.llm.mlx_provider import MLXProvider

        provider = MLXProvider(mlx_config)

        # First use
        result1 = await provider.embed("First embedding")
        assert result1 is not None

        # Close
        await provider.close()
        assert not provider._embedding_model_loaded

        # Use again (should reload model)
        result2 = await provider.embed("Second embedding")
        assert result2 is not None
        assert provider._embedding_model_loaded

        # Clean up
        await provider.close()


class TestVectorSearchWithChunking:
    """Tests for vector search with document chunking."""

    @pytest.fixture
    def vector_db(self, tmp_path: Path) -> Database:
        """Provide a connected database with vector support."""
        db_path = tmp_path / "chunking_test.db"
        database = Database(db_path)
        database.connect()
        yield database
        database.close()

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        """Provide config with chunking settings."""
        cfg = Config()
        cfg.db_path = tmp_path / "chunking_test.db"
        # Very small max_bytes to force chunking in tests
        cfg.chunk.max_bytes = 200
        cfg.chunk.min_chunk_size = 50
        return cfg

    @pytest.fixture
    def embedding_repo(self, vector_db: Database) -> EmbeddingRepository:
        """Provide an EmbeddingRepository instance."""
        return EmbeddingRepository(vector_db)

    @pytest.mark.asyncio
    async def test_embed_document_creates_multiple_chunks(
        self, mlx_provider, embedding_repo: EmbeddingRepository, config: Config
    ):
        """Long documents should be split into multiple chunks."""
        from pmd.llm.embeddings import EmbeddingGenerator

        generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)

        # Create long content that should be split with 200 byte max chunks
        long_content = "This is paragraph one with some content. " * 20 + "\n\n"
        long_content += "This is paragraph two with different content. " * 20

        test_hash = "d" * 64
        chunks_embedded = await generator.embed_document(test_hash, long_content)

        # Should have multiple chunks (content is ~1.6KB, chunks max 200 bytes)
        assert chunks_embedded > 1

        # Check all chunks are stored
        embeddings = embedding_repo.get_embeddings_for_content(test_hash)
        assert len(embeddings) == chunks_embedded

    @pytest.mark.asyncio
    async def test_vector_search_finds_relevant_chunk(
        self,
        mlx_provider,
        vector_db: Database,
        embedding_repo: EmbeddingRepository,
        config: Config,
        tmp_path: Path,
    ):
        """Vector search should find documents based on chunk content."""
        if not vector_db.vec_available:
            pytest.skip("sqlite-vec extension not available")

        from pmd.llm.embeddings import EmbeddingGenerator
        from pmd.store.collections import CollectionRepository
        from pmd.store.documents import DocumentRepository

        collection_repo = CollectionRepository(vector_db)
        document_repo = DocumentRepository(vector_db)

        collection = collection_repo.create("chunking", str(tmp_path), "**/*.md")

        # Document with distinct paragraphs
        content = """
# Introduction

This document covers general topics.

# Machine Learning Section

Deep learning uses neural networks to learn patterns from data.
Convolutional networks are great for image recognition.

# Cooking Section

Italian cuisine uses olive oil, garlic, and fresh basil.
Pasta should be cooked al dente for best texture.
"""
        doc, _ = document_repo.add_or_update(
            collection.id, "mixed.md", "Mixed Topics", content
        )

        generator = EmbeddingGenerator(mlx_provider, embedding_repo, config)
        await generator.embed_document(doc.hash, content)

        # Search for ML content
        query_emb = await mlx_provider.embed("neural networks deep learning", is_query=True)
        assert query_emb is not None

        results = embedding_repo.search_vectors(query_emb.embedding, limit=1)

        assert len(results) == 1
        assert results[0].filepath == "mixed.md"
        # chunk_pos should indicate which part of document matched
        assert results[0].chunk_pos is not None
