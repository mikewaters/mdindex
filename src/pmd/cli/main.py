"""CLI entry point for PMD."""

import argparse
import sys
from typing import NoReturn

from loguru import logger

from ..core.config import Config
from . import commands


def configure_logging(level: str) -> None:
    """Configure loguru logging with the specified level.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
    """
    # Remove default handler
    logger.remove()

    # Add custom handler with formatting
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=(
            "<green>{time:HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="pmd",
        description="Python Markdown Search - Hybrid search for markdown documents",
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")
    parser.add_argument(
        "-L",
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Set logging level (default: WARNING)",
    )
    parser.add_argument(
        "-c",
        "--config",
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--phoenix-tracing",
        action="store_true",
        help="Enable OpenTelemetry tracing to Arize Phoenix",
    )
    parser.add_argument(
        "--phoenix-endpoint",
        help="Phoenix OTLP endpoint (default: http://localhost:6006/v1/traces)",
    )

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Collection commands
    coll_parser = subparsers.add_parser(
        "collection", help="Manage collections"
    )
    coll_subparsers = coll_parser.add_subparsers(dest="collection_cmd", required=True)

    # collection add
    add_parser = coll_subparsers.add_parser("add", help="Add a collection")
    add_parser.add_argument("name", help="Collection name")
    add_parser.add_argument("path", help="Directory path or URL to index")
    add_parser.add_argument(
        "-g",
        "--glob",
        default="**/*.md",
        help="File glob pattern for filesystem sources (default: **/*.md)",
    )
    add_parser.add_argument(
        "-s",
        "--source",
        choices=["filesystem", "http", "entity"],
        default="filesystem",
        help="Source type (default: filesystem)",
    )
    add_parser.add_argument(
        "--sitemap",
        help="Sitemap URL for HTTP sources",
    )
    add_parser.add_argument(
        "--auth-type",
        choices=["none", "bearer", "basic", "api_key"],
        default="none",
        help="Authentication type for remote sources",
    )
    add_parser.add_argument(
        "--auth-token",
        help="Auth token/password (or $ENV:VAR_NAME reference)",
    )
    add_parser.add_argument(
        "--username",
        help="Username for basic auth",
    )

    # collection list
    coll_subparsers.add_parser("list", help="List all collections")

    # collection show
    show_parser = coll_subparsers.add_parser("show", help="Show collection details")
    show_parser.add_argument("name", help="Collection name")

    # collection remove
    remove_parser = coll_subparsers.add_parser("remove", help="Remove a collection")
    remove_parser.add_argument("name", help="Collection name")

    # collection rename
    rename_parser = coll_subparsers.add_parser("rename", help="Rename a collection")
    rename_parser.add_argument("old_name", help="Current collection name")
    rename_parser.add_argument("new_name", help="New collection name")

    # Search commands
    search_parser = subparsers.add_parser("search", help="BM25 keyword search")
    commands.add_search_arguments(search_parser)

    vsearch_parser = subparsers.add_parser("vsearch", help="Vector semantic search")
    commands.add_search_arguments(vsearch_parser)

    query_parser = subparsers.add_parser("query", help="Hybrid search with reranking")
    commands.add_search_arguments(query_parser)

    # Indexing commands
    update_all_parser = subparsers.add_parser("update-all", help="Update all collections")
    update_all_parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate embeddings after indexing",
    )

    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    commands.add_index_arguments(embed_parser)

    index_parser = subparsers.add_parser("index", help="Index a collection")
    commands.add_index_arguments(index_parser)

    subparsers.add_parser("cleanup", help="Clean cache and orphaned data")

    # Metadata backfill command
    backfill_parser = subparsers.add_parser(
        "backfill-metadata", help="Backfill document metadata for existing documents"
    )
    commands.add_backfill_arguments(backfill_parser)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show index status")
    commands.add_status_arguments(status_parser)

    return parser


def main() -> NoReturn:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Configure logging before anything else
    configure_logging(args.log_level)

    config = Config.from_env_or_file(args.config)
    logger.debug(f"Loaded config, db_path={config.db_path}")

    # Apply CLI tracing overrides (CLI takes precedence over config/env)
    if args.phoenix_tracing:
        config.tracing.enabled = True
    if args.phoenix_endpoint:
        config.tracing.phoenix_endpoint = args.phoenix_endpoint

    # Initialize Phoenix tracing if enabled
    if config.tracing.enabled:
        from ..core.instrumentation import configure_phoenix_tracing
        configure_phoenix_tracing(config.tracing)

    try:
        if args.command == "collection":
            commands.handle_collection(args, config)
        elif args.command == "search":
            commands.handle_search(args, config)
        elif args.command == "vsearch":
            commands.handle_vsearch(args, config)
        elif args.command == "query":
            commands.handle_query(args, config)
        elif args.command == "index":
            commands.handle_index_collection(args, config)
        elif args.command == "update-all":
            commands.handle_update_all(args, config)
        elif args.command == "embed":
            commands.handle_embed(args, config)
        elif args.command == "cleanup":
            commands.handle_cleanup(args, config)
        elif args.command == "backfill-metadata":
            commands.handle_backfill_metadata(args, config)
        elif args.command == "status":
            commands.handle_status(args, config)
        else:
            parser.print_help()

        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
