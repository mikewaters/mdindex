"""Search commands for PMD CLI.

Provides three search commands:
- `pmd search`: BM25 full-text search (FTS5)
- `pmd vsearch`: Vector semantic search
- `pmd query`: Hybrid search with FTS + vector + LLM reranking
"""

import asyncio

from pmd.core.config import Config
from pmd.app import create_application


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

    Uses SearchService for BM25 full-text search.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_search_async(args, config))


async def _handle_search_async(args, config: Config) -> None:
    """Async handler for FTS search.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    async with await create_application(config) as app:
        results = app.search.fts_search(
            args.query,
            limit=args.limit,
            collection_name=args.collection,
            min_score=args.score,
        )

        _print_search_results(results, "FTS5 Search")


def handle_vsearch(args, config: Config) -> None:
    """Handle vsearch command (vector search).

    Uses SearchService for semantic similarity search.

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
    async with await create_application(config) as app:
        # Check if vector search is available
        if not app.vec_available:
            print("Vector search not available (sqlite-vec extension not loaded)")
            return

        # Check if LLM is available
        if not await app.is_llm_available():
            print("LLM provider not available (is it running?)")
            return

        results = await app.search.vector_search(
            args.query,
            limit=args.limit,
            collection_name=args.collection,
            min_score=args.score,
        )

        _print_search_results(results, "Vector Search")


def handle_query(args, config: Config) -> None:
    """Handle query command (hybrid search).

    Uses SearchService for hybrid FTS + vector search with optional reranking.

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
    async with await create_application(config) as app:
        # Check if LLM is available for query expansion/reranking
        llm_available = await app.is_llm_available()

        results = await app.search.hybrid_search(
            args.query,
            limit=args.limit,
            collection_name=args.collection,
            min_score=args.score,
            enable_query_expansion=llm_available,
            enable_reranking=llm_available,
        )

        _print_search_results(results, "Hybrid Search")


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

        # Handle both SearchResult and RankedResult
        if hasattr(result, "title"):
            title_text = result.title
        else:
            title_text = getattr(result, "file", "Unknown")

        if hasattr(result, "display_path"):
            path_text = result.display_path
        elif hasattr(result, "file"):
            path_text = result.file
        else:
            path_text = "Unknown"

        print(f"{rank}. {title_text} [{score_display}]")
        print(f"   File: {path_text}")

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
