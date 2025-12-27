"""CLI entry point for PMD."""

import argparse
import sys
from typing import NoReturn

from ..core.config import Config


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="pmd",
        description="Python Markdown Search - Hybrid search for markdown documents"
    )
    parser.add_argument("-v", "--version", action="version", version="%(prog)s 1.0.0")

    subparsers = parser.add_subparsers(dest="command", required=False)

    # Placeholder for commands - to be implemented
    subparsers.add_parser("status", help="Show index status")

    return parser


def main() -> NoReturn:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    config = Config.from_env()

    if args.command == "status":
        print("Index status: Not yet implemented")
    else:
        parser.print_help()

    sys.exit(0)


if __name__ == "__main__":
    main()
