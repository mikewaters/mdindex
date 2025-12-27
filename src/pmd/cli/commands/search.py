"""Search commands for PMD CLI."""

from ...core.config import Config
from ...search.pipeline import HybridSearchPipeline, SearchPipelineConfig
from ...store.database import Database
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

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        search_repo = FTS5SearchRepository(db)

        # Get collection ID if specified
        collection_id = None
        if args.collection:
            from .collections import _get_collection_id

            collection_id = _get_collection_id(db, args.collection)

        # Perform FTS5 search
        results = search_repo.search_fts(
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

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        search_repo = FTS5SearchRepository(db)

        # Get collection ID if specified
        collection_id = None
        if args.collection:
            from .collections import _get_collection_id

            collection_id = _get_collection_id(db, args.collection)

        # Vector search (placeholder in Phase 2)
        results = search_repo.search_vec(
            [],  # Empty embedding in Phase 2
            limit=args.limit,
            collection_id=collection_id,
            min_score=args.score,
        )

        if not results:
            print("Vector search not yet available (Phase 3)")
        else:
            _print_search_results(results, "Vector Search")
    finally:
        db.close()


def handle_query(args, config: Config) -> None:
    """Handle query command (hybrid search).

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        search_repo = FTS5SearchRepository(db)

        # Get collection ID if specified
        collection_id = None
        if args.collection:
            from .collections import _get_collection_id

            collection_id = _get_collection_id(db, args.collection)

        # Run hybrid search pipeline
        pipeline_config = SearchPipelineConfig(
            fts_weight=config.search.fts_weight,
            vec_weight=config.search.vec_weight,
            rrf_k=config.search.rrf_k,
            rerank_candidates=config.search.rerank_candidates,
        )
        pipeline = HybridSearchPipeline(search_repo, pipeline_config)

        results = pipeline.search(
            args.query,
            limit=args.limit,
            collection_id=collection_id,
            min_score=args.score,
        )

        _print_search_results(results, "Hybrid Search")
    finally:
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

        if hasattr(result, "fts_score") and result.fts_score is not None:
            print(f"   FTS Score: {result.fts_score:.3f}")

        if hasattr(result, "vec_score") and result.vec_score is not None:
            print(f"   Vector Score: {result.vec_score:.3f}")

        if hasattr(result, "rerank_score") and result.rerank_score is not None:
            print(f"   Rerank Score: {result.rerank_score:.3f}")

        print()
