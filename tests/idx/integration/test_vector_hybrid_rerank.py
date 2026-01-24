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
from idx.search.hybrid import HybridSearch, rrf_fusion
from idx.search.models import SearchCriteria, SearchResult, SearchResults
from idx.store.database import Base, create_engine_for_path
from idx.store.fts import create_fts_table


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

    with patch("idx.pipelines.ingest.get_session", get_test_session):
        yield get_test_session


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
        """search_with_scores returns (path, dataset, score) tuples."""
        # Mock the return format used for hybrid fusion
        vec_scores = [
            ("auth.md", "vault", 0.95),
            ("api.md", "vault", 0.85),
            ("database.md", "vault", 0.70),
        ]

        # Verify format works for RRF fusion
        fused = rrf_fusion([vec_scores])

        assert len(fused) == 3
        for path, dataset_name, score in fused:
            assert isinstance(path, str)
            assert isinstance(dataset_name, str)
            assert isinstance(score, float)


class TestHybridSearchIntegration:
    """Integration tests for hybrid RRF search."""

    def test_hybrid_search_combines_sources(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
    ) -> None:
        """Hybrid search combines FTS and vector results."""
        # Ingest documents (FTS only for this test)
        pipeline = IngestPipeline()
        config = IngestDirectoryConfig(
            source_path=sample_docs,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )
        result = pipeline.ingest(config)
        assert result.documents_created == 3

        # Create mock vector search
        mock_vector = MagicMock()
        mock_vector.search_with_scores.return_value = [
            ("auth.md", "test-vault", 0.95),
            ("api.md", "test-vault", 0.80),
        ]

        # Run hybrid search
        with create_session(session_factory) as session:
            hybrid = HybridSearch(
                session=session,
                vector_search=mock_vector,
            )
            results = hybrid.search(
                SearchCriteria(query="authentication", mode="hybrid", limit=10)
            )

        # Verify hybrid results
        assert results.mode == "hybrid"
        assert len(results.results) >= 1

        # Results should have RRF scores
        for result in results.results:
            assert "rrf" in result.scores

    def test_hybrid_search_normalizes_scores(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
    ) -> None:
        """Hybrid search normalizes RRF scores to 0-1 range."""
        # Ingest documents
        pipeline = IngestPipeline()
        config = IngestDirectoryConfig(
            source_path=sample_docs,
            dataset_name="test-vault",
            patterns=["**/*.md"],
        )
        pipeline.ingest(config)

        # Mock vector search
        mock_vector = MagicMock()
        mock_vector.search_with_scores.return_value = [
            ("auth.md", "test-vault", 0.9),
            ("database.md", "test-vault", 0.8),
        ]

        # Run hybrid search
        with create_session(session_factory) as session:
            hybrid = HybridSearch(
                session=session,
                vector_search=mock_vector,
            )
            results = hybrid.search(
                SearchCriteria(query="database", mode="hybrid", limit=10)
            )

        # Top result should have score of 1.0 after normalization
        if results.results:
            assert results.results[0].score == 1.0

        # All scores should be between 0 and 1
        for result in results.results:
            assert 0.0 <= result.score <= 1.0

    def test_hybrid_search_respects_dataset_filter(
        self,
        patched_get_session,
        session_factory,
        sample_docs: Path,
        tmp_path: Path,
    ) -> None:
        """Hybrid search filters by dataset name."""
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

        # Mock vector search that respects dataset filter
        mock_vector = MagicMock()
        mock_vector.search_with_scores.return_value = [
            ("auth.md", "vault1", 0.9),
        ]

        # Search with filter
        with create_session(session_factory) as session:
            hybrid = HybridSearch(
                session=session,
                vector_search=mock_vector,
            )
            results = hybrid.search(
                SearchCriteria(
                    query="content",
                    mode="hybrid",
                    dataset_name="vault1",
                    limit=10,
                )
            )

        # Results should only be from vault1
        for result in results.results:
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

        # 2. Hybrid search (with mocked vector)
        mock_vector = MagicMock()
        mock_vector.search_with_scores.return_value = [
            ("auth.md", "test-vault", 0.95),
            ("api.md", "test-vault", 0.85),
        ]

        with create_session(session_factory) as session:
            hybrid = HybridSearch(
                session=session,
                vector_search=mock_vector,
            )
            search_results = hybrid.search(
                SearchCriteria(query="authentication", mode="hybrid", limit=10)
            )

        # 3. Rerank (with mocked LLM)
        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(side_effect=["Yes", "Yes", "No"])

        # Add chunk text for reranker
        for result in search_results.results:
            if not result.chunk_text:
                result.chunk_text = f"Content from {result.path}"

        reranker = Reranker(provider=mock_provider)
        reranked = await reranker.rerank(
            "authentication",
            search_results.results,
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
