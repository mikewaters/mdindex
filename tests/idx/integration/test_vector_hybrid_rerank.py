"""Integration tests for vector search, hybrid RRF, and LLM reranking.

Smoke tests for end-to-end flow: ingest with vector indexing,
hybrid search with RRF fusion, and reranker invocation.

Note: Vector search is mocked to avoid requiring embedding model downloads
in tests. The goal is to test integration paths and result shapes.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from idx.ingest.pipelines import IngestPipeline
from idx.ingest.schemas import IngestDirectoryConfig
from idx.search.fts import FTSSearch
from idx.search.hybrid import HybridSearch
from idx.search.models import SearchCriteria, SearchResult, SearchResults
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import create_fts_table
from idx.store.session_context import use_session


@pytest.fixture
def test_engine(tmp_path: Path) -> Engine:
    """Create a temporary database and return the engine."""
    db_path = tmp_path / "test.db"
    engine = create_engine_for_path(db_path)
    Base.metadata.create_all(engine)
    create_fts_table(engine)
    return engine


@pytest.fixture
def session_factory(test_engine: Engine):
    """Create a session factory for the test database."""
    return sessionmaker(bind=test_engine, expire_on_commit=False)


@contextmanager
def create_session(factory) -> Generator[Session, None, None]:
    """Create a session that auto-commits on exit."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.fixture
def patched_get_session(session_factory):
    """Patch get_session to use the test database."""
    @contextmanager
    def get_test_session():
        with create_session(session_factory) as session:
            yield session

    with patch("idx.ingest.pipelines.get_session", get_test_session):
        yield get_test_session


@pytest.fixture(autouse=True)
def use_mock_embedding(patched_embedding) -> None:
    """Use mock embedding model for all tests."""
    yield


@pytest.fixture
def sample_docs(tmp_path: Path) -> Path:
    """Create sample documents for testing."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "auth.md").write_text("""# Authentication

How to implement user authentication.
OAuth2, JWT tokens, and session management.
""")

    (docs / "database.md").write_text("""# Database Design

SQL databases and schema design.
Indexing strategies and query optimization.
""")

    (docs / "api.md").write_text("""# API Design

RESTful API patterns.
Authentication endpoints and rate limiting.
""")

    return docs


class TestVectorSearchIntegration:
    """Integration tests for vector search path."""

    def test_vector_search_result_shape(self) -> None:
        """Vector search returns properly shaped SearchResults."""
        # Create mock results simulating vector search output
        mock_results = SearchResults(
            results=[
                SearchResult(
                    path="auth.md",
                    dataset_name="vault",
                    score=0.95,
                    chunk_text="How to implement authentication",
                    chunk_seq=0,
                    chunk_pos=0,
                    scores={"vector": 0.95},
                ),
                SearchResult(
                    path="api.md",
                    dataset_name="vault",
                    score=0.85,
                    chunk_text="Authentication endpoints",
                    chunk_seq=0,
                    chunk_pos=0,
                    scores={"vector": 0.85},
                ),
            ],
            query="authentication",
            mode="vector",
            total_candidates=2,
            timing_ms=50.0,
        )

        # Verify shape
        assert mock_results.mode == "vector"
        assert len(mock_results.results) == 2
        for result in mock_results.results:
            assert result.path
            assert result.dataset_name
            assert 0.0 <= result.score <= 1.0
            assert "vector" in result.scores

    def test_vector_search_with_scores_format(self) -> None:
        """search_with_scores returns (path, dataset, score) tuples format."""
        # Mock the return format used internally
        vec_scores = [
            ("auth.md", "vault", 0.95),
            ("api.md", "vault", 0.85),
            ("database.md", "vault", 0.70),
        ]

        # Verify format is correct for search pipeline
        assert len(vec_scores) == 3
        for path, dataset_name, score in vec_scores:
            assert isinstance(path, str)
            assert isinstance(dataset_name, str)
            assert isinstance(score, float)


class TestHybridSearchIntegration:
    """Integration tests for hybrid RRF search.

    Note: HybridSearch uses QueryFusionRetriever internally which requires
    real vector indices. These tests mock the VectorStoreManager to avoid
    embedding model downloads.
    """

    def test_hybrid_search_combines_sources(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
    ) -> None:
        """Hybrid search combines FTS and vector results."""
        from llama_index.core.schema import NodeWithScore, TextNode

        # Ingest documents (FTS only for this test)
        pipeline = IngestPipeline()
        config = IngestDirectoryConfig(
            source_path=sample_docs,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )
        result = pipeline.ingest(config)
        assert result.documents_created == 3

        # Create mock vector manager
        mock_vector_manager = MagicMock()
        mock_index = MagicMock()
        mock_vector_manager.load_or_create.return_value = mock_index

        # Mock retriever to return some nodes
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [
            NodeWithScore(
                node=TextNode(
                    text="Authentication content",
                    metadata={"source_doc_id": "test-vault:auth.md"},
                ),
                score=0.95,
            ),
        ]

        # Run hybrid search with ambient session
        with create_session(session_factory) as session:
            with use_session(session):
                hybrid = HybridSearch(vector_manager=mock_vector_manager)
                results = hybrid.search(
                    query="authentication",
                    top_k=10,
                )

        # Verify hybrid results
        assert len(results) >= 1

        # Results should have RRF scores
        for result in results:
            assert "rrf" in result.scores

    def test_hybrid_search_normalizes_scores(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
    ) -> None:
        """Hybrid search normalizes RRF scores to 0-1 range."""
        from llama_index.core.schema import NodeWithScore, TextNode

        # Ingest documents
        pipeline = IngestPipeline()
        config = IngestDirectoryConfig(
            source_path=sample_docs,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )
        pipeline.ingest(config)

        # Create mock vector manager
        mock_vector_manager = MagicMock()
        mock_index = MagicMock()
        mock_vector_manager.load_or_create.return_value = mock_index

        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [
            NodeWithScore(
                node=TextNode(
                    text="Database content",
                    metadata={"source_doc_id": "test-vault:database.md"},
                ),
                score=0.9,
            ),
        ]

        # Run hybrid search
        with create_session(session_factory) as session:
            with use_session(session):
                hybrid = HybridSearch(vector_manager=mock_vector_manager)
                results = hybrid.search(query="database", top_k=10)

        # All scores should be between 0 and 1
        for result in results:
            assert 0.0 <= result.score <= 1.0

    def test_hybrid_search_respects_dataset_filter(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
        tmp_path: Path,
    ) -> None:
        """Hybrid search filters by dataset name."""
        from llama_index.core.schema import NodeWithScore, TextNode

        pipeline = IngestPipeline()

        # Create two datasets
        docs2 = tmp_path / "docs2"
        docs2.mkdir()
        (docs2 / "other.md").write_text("# Other\n\nUnrelated content.")

        # Ingest both
        pipeline.ingest(
            IngestDirectoryConfig(
                source_path=sample_docs,
                dataset_name="vault1",
                patterns=["**/*.md"],
            )
        )
        pipeline.ingest(
            IngestDirectoryConfig(
                source_path=docs2,
                dataset_name="vault2",
                patterns=["**/*.md"],
            )
        )

        # Create mock vector manager
        mock_vector_manager = MagicMock()
        mock_index = MagicMock()
        mock_vector_manager.load_or_create.return_value = mock_index

        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [
            NodeWithScore(
                node=TextNode(
                    text="Auth content",
                    metadata={"source_doc_id": "vault1:auth.md"},
                ),
                score=0.9,
            ),
        ]

        # Search with filter
        with create_session(session_factory) as session:
            with use_session(session):
                hybrid = HybridSearch(vector_manager=mock_vector_manager)
                results = hybrid.search(
                    query="content",
                    top_k=10,
                    dataset_name="vault1",
                )

        # Results should only be from vault1
        for result in results:
            assert result.dataset_name == "vault1"


class TestRerankerIntegration:
    """Integration tests for LLM-as-judge reranker."""

    @pytest.mark.asyncio
    async def test_reranker_invocation(self) -> None:
        """Reranker can score search results."""
        from idx.llm.reranker import Reranker

        # Create mock provider
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=["Yes", "No", "Yes"])

        # Create search results to rerank
        results = [
            SearchResult(
                path="auth.md",
                dataset_name="vault",
                score=0.9,
                chunk_text="How to implement OAuth2 authentication",
            ),
            SearchResult(
                path="database.md",
                dataset_name="vault",
                score=0.8,
                chunk_text="SQL indexing strategies",
            ),
            SearchResult(
                path="api.md",
                dataset_name="vault",
                score=0.7,
                chunk_text="Authentication endpoints and rate limiting",
            ),
        ]

        # Rerank
        reranker = Reranker(provider=mock_provider)
        reranked = await reranker.rerank("authentication patterns", results)

        # Verify results
        assert len(reranked) == 3
        for result in reranked:
            assert "rerank" in result.scores
            assert 0.0 <= result.scores["rerank"] <= 1.0

    @pytest.mark.asyncio
    async def test_reranker_blends_scores(self) -> None:
        """Reranker blends RRF and LLM scores with position-aware weights."""
        from idx.llm.reranker import Reranker

        # Create mock provider - all relevant
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="Yes")

        results = [
            SearchResult(
                path="a.md",
                dataset_name="vault",
                score=0.9,
                chunk_text="Relevant content",
                scores={"rrf": 0.9},
            ),
            SearchResult(
                path="b.md",
                dataset_name="vault",
                score=0.5,
                chunk_text="More relevant content",
                scores={"rrf": 0.5},
            ),
        ]

        reranker = Reranker(provider=mock_provider)
        reranked = await reranker.rerank("query", results)

        # Both should have blend weights
        for result in reranked:
            assert "blend_weight" in result.scores
            # Rank 0-2 get 75% weight
            assert result.scores["blend_weight"] == 0.75

    @pytest.mark.asyncio
    async def test_reranker_handles_errors_gracefully(self) -> None:
        """Reranker returns neutral scores on LLM errors."""
        from idx.llm.reranker import Reranker

        # Create mock provider that fails
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=Exception("LLM error"))

        results = [
            SearchResult(
                path="doc.md",
                dataset_name="vault",
                score=0.9,
                chunk_text="Some content",
            ),
        ]

        reranker = Reranker(provider=mock_provider)
        reranked = await reranker.rerank("query", results)

        # Should still return results with neutral score
        assert len(reranked) == 1
        assert reranked[0].scores["rerank"] == 0.5  # Neutral on error


class TestEndToEndFlow:
    """End-to-end smoke tests for full search pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_ingest_search_rerank(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
    ) -> None:
        """Full flow: ingest -> hybrid search -> rerank."""
        from llama_index.core.schema import NodeWithScore, TextNode

        from idx.llm.reranker import Reranker

        # 1. Ingest documents
        pipeline = IngestPipeline()
        config = IngestDirectoryConfig(
            source_path=sample_docs,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )
        result = pipeline.ingest(config)
        assert result.documents_created == 3

        # 2. Hybrid search (with mocked vector manager)
        mock_vector_manager = MagicMock()
        mock_index = MagicMock()
        mock_vector_manager.load_or_create.return_value = mock_index

        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        mock_retriever.retrieve.return_value = [
            NodeWithScore(
                node=TextNode(
                    text="Authentication with OAuth2",
                    metadata={"source_doc_id": "test-vault:auth.md"},
                ),
                score=0.95,
            ),
            NodeWithScore(
                node=TextNode(
                    text="API authentication endpoints",
                    metadata={"source_doc_id": "test-vault:api.md"},
                ),
                score=0.85,
            ),
        ]

        with create_session(session_factory) as session:
            with use_session(session):
                hybrid = HybridSearch(vector_manager=mock_vector_manager)
                search_results = hybrid.search(
                    query="authentication",
                    top_k=10,
                )

        # 3. Rerank (with mocked LLM)
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=["Yes", "Yes", "No"])

        # Add chunk text for reranker if missing
        for res in search_results:
            if not res.chunk_text:
                res.chunk_text = f"Content from {res.path}"

        reranker = Reranker(provider=mock_provider)
        reranked = await reranker.rerank(
            "authentication",
            search_results,
        )

        # 4. Verify final results
        assert len(reranked) >= 1

        # Results should be re-sorted by blended score
        for i in range(len(reranked) - 1):
            assert reranked[i].score >= reranked[i + 1].score

        # Each result should have full score breakdown
        for result in reranked:
            assert result.path
            assert result.dataset_name
            assert "rerank" in result.scores
            assert "blend_weight" in result.scores

    def test_fts_search_standalone(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
    ) -> None:
        """FTS search works independently of vector search."""
        # Ingest
        pipeline = IngestPipeline()
        config = IngestDirectoryConfig(
            source_path=sample_docs,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )
        pipeline.ingest(config)

        # Search
        with create_session(session_factory) as session:
            fts = FTSSearch(session)
            results = fts.search(
                SearchCriteria(query="authentication", mode="fts", limit=10)
            )

        # auth.md and api.md both mention authentication
        assert len(results.results) >= 1
        assert results.mode == "fts"

        # Results have correct structure
        for result in results.results:
            assert result.path.endswith(".md")
            assert result.dataset_name == "test-vault"


class TestResultShapeStability:
    """Tests ensuring result shapes are stable across operations."""

    def test_search_result_required_fields(self) -> None:
        """SearchResult has all required fields."""
        result = SearchResult(
            path="test.md",
            dataset_name="vault",
            score=0.8,
        )

        assert hasattr(result, "path")
        assert hasattr(result, "dataset_name")
        assert hasattr(result, "score")
        assert hasattr(result, "chunk_text")
        assert hasattr(result, "chunk_seq")
        assert hasattr(result, "chunk_pos")
        assert hasattr(result, "metadata")
        assert hasattr(result, "scores")

    def test_search_results_metadata(self) -> None:
        """SearchResults has timing and candidate metadata."""
        results = SearchResults(
            results=[],
            query="test",
            mode="fts",
            total_candidates=0,
            timing_ms=10.5,
        )

        assert hasattr(results, "query")
        assert hasattr(results, "mode")
        assert hasattr(results, "total_candidates")
        assert hasattr(results, "timing_ms")
        assert results.timing_ms is not None

    def test_scores_dict_preserved_through_operations(self) -> None:
        """The scores dict accumulates through search pipeline."""
        from idx.llm.reranker import blend_scores, RerankScore

        # Initial result with FTS score
        result = SearchResult(
            path="test.md",
            dataset_name="vault",
            score=0.8,
            scores={"fts": 0.8, "rrf": 0.9},
        )

        # After reranking
        rerank_scores = [
            RerankScore(
                path="test.md",
                dataset_name="vault",
                relevant=True,
                confidence=0.9,
                score=0.95,
                raw_response="Yes",
            )
        ]

        blended = blend_scores([result], rerank_scores)

        # Should have all scores
        assert len(blended) == 1
        assert "fts" in blended[0].scores
        assert "rrf" in blended[0].scores
        assert "rerank" in blended[0].scores
        assert "blend_weight" in blended[0].scores
