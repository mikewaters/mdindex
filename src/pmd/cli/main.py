"""CLI entry point for PMD."""

import argparse
import sys
from typing import NoReturn

from ..core.config import Config
from . import commands


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="pmd",
        description="Python Markdown Search - Hybrid search for markdown documents",
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Collection commands
    coll_parser = subparsers.add_parser(
        "collection", help="Manage collections"
    )
    coll_subparsers = coll_parser.add_subparsers(dest="collection_cmd", required=True)

    # collection add
    add_parser = coll_subparsers.add_parser("add", help="Add a collection")
    add_parser.add_argument("name", help="Collection name")
    add_parser.add_argument("path", help="Directory path to index")
    add_parser.add_argument(
        "-g",
        "--glob",
        default="**/*.md",
        help="File glob pattern (default: **/*.md)",
    )

    # collection list
    coll_subparsers.add_parser("list", help="List all collections")

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
    subparsers.add_parser("update-all", help="Update all collections")

    embed_parser = subparsers.add_parser("embed", help="Generate embeddings")
    commands.add_index_arguments(embed_parser)

    index_parser = subparsers.add_parser("index", help="Index a collection")
    commands.add_index_arguments(index_parser)

    subparsers.add_parser("cleanup", help="Clean cache and orphaned data")

    # Status command
    subparsers.add_parser("status", help="Show index status")

    return parser


def main() -> NoReturn:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    config = Config.from_env()

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
