"""Search commands for PMD CLI.

Provides three search commands:
- `pmd search`: BM25 full-text search (FTS5)
- `pmd vsearch`: Vector semantic search
- `pmd query`: Hybrid search with FTS + vector + LLM reranking
"""

import asyncio

from ...core.config import Config
from ...llm import QueryExpander, DocumentReranker, create_llm_provider
from ...llm.embeddings import EmbeddingGenerator
from ...search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from ...store.database import Database
from ...store.embeddings import EmbeddingRepository
from ...store.search import FTS5SearchRepository


def add_search_arguments(parser) -> None:
    """Add arguments for search commands.

    Args:
        parser: Argument parser for the command.
    """
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "-l",
        "--limit",
        type=int,
        default=5,
        help="Maximum results to return (default: 5)",
    )
    parser.add_argument(
        "-c",
        "--collection",
        help="Limit search to specific collection",
    )
    parser.add_argument(
        "-s",
        "--score",
        type=float,
        default=0.0,
        help="Minimum score threshold (default: 0.0)",
    )


def handle_search(args, config: Config) -> None:
    """Handle search command (FTS5 only).

    Uses FTS5SearchRepository for BM25 full-text search.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        fts_repo = FTS5SearchRepository(db)

        # Get collection ID if specified
        collection_id = None
        if args.collection:
            from .collection import _get_collection_id

            collection_id = _get_collection_id(db, args.collection)

        # Perform FTS5 search using the unified search() method
        results = fts_repo.search(
            args.query,
            limit=args.limit,
            collection_id=collection_id,
            min_score=args.score,
        )

        _print_search_results(results, "FTS5 Search")
    finally:
        db.close()


def handle_vsearch(args, config: Config) -> None:
    """Handle vsearch command (vector search).

    Uses EmbeddingRepository for semantic similarity search.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_vsearch_async(args, config))


async def _handle_vsearch_async(args, config: Config) -> None:
    """Async handler for vector search.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    llm_provider = None
    try:
        embedding_repo = EmbeddingRepository(db)

        # Check if vector search is available
        if not db.vec_available:
            raise RuntimeError("Vector search not available (sqlite-vec extension not loaded)")

        # Get collection ID if specified
        collection_id = None
        if args.collection:
            from .collection import _get_collection_id

            collection_id = _get_collection_id(db, args.collection)

        # Generate query embedding via LLM
        llm_provider = create_llm_provider(config)
        embedding_generator = EmbeddingGenerator(llm_provider, embedding_repo, config)

        query_embedding = await embedding_generator.embed_query(args.query)

        if not query_embedding:
            print("Failed to generate query embedding (is LLM provider running?)")
            return

        # Perform vector search via EmbeddingRepository
        results = embedding_repo.search_vectors(
            query_embedding,
            limit=args.limit,
            collection_id=collection_id,
            min_score=args.score,
        )

        _print_search_results(results, "Vector Search")
    finally:
        if llm_provider:
            await llm_provider.close()
        db.close()


def handle_query(args, config: Config) -> None:
    """Handle query command (hybrid search).

    Uses HybridSearchPipeline with FTS5 and EmbeddingRepository for vector search.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_query_async(args, config))


async def _handle_query_async(args, config: Config) -> None:
    """Async handler for hybrid search.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    llm_provider = None
    try:
        # Create repositories for FTS and embeddings
        embedding_repo = EmbeddingRepository(db)
        fts_repo = FTS5SearchRepository(db)

        # Get collection ID if specified
        collection_id = None
        if args.collection:
            from .collection import _get_collection_id

            collection_id = _get_collection_id(db, args.collection)

        # Initialize LLM components
        llm_provider = create_llm_provider(config)
        query_expander = QueryExpander(llm_provider)
        reranker = DocumentReranker(llm_provider)
        embedding_generator = EmbeddingGenerator(llm_provider, embedding_repo, config)

        # Run hybrid search pipeline (embedding_generator provides vector search)
        pipeline_config = SearchPipelineConfig(
            fts_weight=config.search.fts_weight,
            vec_weight=config.search.vec_weight,
            rrf_k=config.search.rrf_k,
            rerank_candidates=config.search.rerank_candidates,
            enable_query_expansion=True,
            enable_reranking=True,
        )
        pipeline = HybridSearchPipeline(
            fts_repo,
            config=pipeline_config,
            query_expander=query_expander,
            reranker=reranker,
            embedding_generator=embedding_generator,
        )

        results = await pipeline.search(
            args.query,
            limit=args.limit,
            collection_id=collection_id,
            min_score=args.score,
        )

        _print_search_results(results, "Hybrid Search")
    finally:
        if llm_provider:
            await llm_provider.close()
        db.close()


def _print_search_results(results, title: str) -> None:
    """Print search results in a formatted way.

    Args:
        results: List of search results.
        title: Title for the results section.
    """
    print(f"\n{title} Results ({len(results)})")
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, 1):
        score_display = f"{result.score:.3f}" if hasattr(result, "score") else "N/A"
        print(f"{rank}. {result.title} [{score_display}]")
        print(f"   File: {result.display_path}")

        # Source provenance (for hybrid search)
        sources = []
        ranks = []
        if hasattr(result, "fts_rank") and result.fts_rank is not None:
            sources.append("FTS")
            ranks.append(f"FTS#{result.fts_rank + 1}")
        if hasattr(result, "vec_rank") and result.vec_rank is not None:
            sources.append("VEC")
            ranks.append(f"VEC#{result.vec_rank + 1}")

        if sources:
            sources_str = "+".join(sources)
            ranks_str = ", ".join(ranks)
            print(f"   Sources: {sources_str} ({ranks_str})")

        # Scores breakdown
        if hasattr(result, "fts_score") and result.fts_score is not None:
            print(f"   FTS Score: {result.fts_score:.3f}")

        if hasattr(result, "vec_score") and result.vec_score is not None:
            print(f"   Vector Score: {result.vec_score:.3f}")

        if hasattr(result, "rerank_score") and result.rerank_score is not None:
            rerank_str = f"Rerank: {result.rerank_score:.3f}"

            # Add relevance judgment if available
            if hasattr(result, "relevant") and result.relevant is not None:
                rel_str = "Yes" if result.relevant else "No"
                rerank_str += f" ({rel_str})"

            # Add confidence if available
            if hasattr(result, "rerank_confidence") and result.rerank_confidence is not None:
                rerank_str += f" conf={result.rerank_confidence:.2f}"

            print(f"   {rerank_str}")

        # Blend weight if available (shows position-aware weighting)
        if hasattr(result, "blend_weight") and result.blend_weight is not None:
            rrf_pct = int(result.blend_weight * 100)
            rerank_pct = 100 - rrf_pct
            print(f"   Blend: {rrf_pct}% RRF + {rerank_pct}% rerank")

        print()
