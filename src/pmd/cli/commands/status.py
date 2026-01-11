"""Status command for PMD CLI."""

import asyncio

from pmd.core.config import Config
from pmd.app import create_application


def add_status_arguments(parser) -> None:
    """Add arguments for status commands.

    Args:
        parser: Argument parser for the command.
    """
    parser.add_argument(
        "--check-sync",
        action="store_true",
        help="Check FTS and vector index synchronization",
    )
    parser.add_argument(
        "--collection",
        help="Limit sync check to a specific collection",
    )
    parser.add_argument(
        "--sync-limit",
        type=int,
        default=20,
        help="Max sample paths per sync category (default: 20)",
    )


def handle_status(args, config: Config) -> None:
    """Handle status command.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_status_async(args, config))


async def _handle_status_async(args, config: Config) -> None:
    """Async handler for status.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    async with await create_application(config) as app:
        status = app.status.get_index_status()
        _print_status(status)

        if args.check_sync:
            report = app.status.get_index_sync_report(
                collection_name=args.collection,
                limit=args.sync_limit,
            )
            _print_sync_report(report)


def _print_status(status) -> None:
    """Print status information.

    Args:
        status: IndexStatus object to display.
    """
    print("PMD Index Status")
    print("=" * 50)
    print(f"Collections: {len(status.source_collections)}")
    print(f"Documents: {status.total_documents}")
    print(f"Embedded: {status.embedded_documents}")
    print(f"Index Size: {status.index_size_bytes} bytes")
    print(f"Embeddings: {status.cache_entries}")
    print()

    if status.source_collections:
        print("Collections:")
        for coll in status.source_collections:
            print(f"  • {coll.name} ({coll.pwd})")
    else:
        print("No collections indexed yet.")


def _print_sync_report(report: dict) -> None:
    """Print FTS/vector sync report."""
    print("\nIndex Sync Report")
    print("=" * 50)

    if "error" in report:
        print(report["error"])
        return

    if report.get("collection"):
        print(f"Collection: {report['collection']}")

    print(f"Missing FTS entries: {report['missing_fts_count']}")
    for path in report.get("missing_fts_paths", []):
        print(f"  • {path}")

    print(f"Missing vector entries: {report['missing_vectors_count']}")
    for path in report.get("missing_vectors_paths", []):
        print(f"  • {path}")

    print(f"Orphaned vector hashes: {report['orphan_vectors_count']}")
    print(f"Orphaned FTS rows: {report['orphan_fts_count']}")
