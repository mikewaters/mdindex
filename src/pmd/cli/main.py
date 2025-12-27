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
