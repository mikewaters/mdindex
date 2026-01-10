"""Document indexing commands for PMD CLI."""

import asyncio

from pmd.core.config import Config
from pmd.core.exceptions import CollectionNotFoundError
from pmd.services import ServiceContainer
from pmd.sources import get_default_registry


def add_index_arguments(parser) -> None:
    """Add arguments for index commands.

    Args:
        parser: Argument parser for the command.
    """
    parser.add_argument(
        "collection",
        help="Collection name to index",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force reindex of all documents",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate embeddings after indexing",
    )


def handle_index_collection(args, config: Config) -> None:
    """Index all documents in a collection.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_index_async(args, config))


async def _handle_index_async(args, config: Config) -> None:
    """Async handler for indexing.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    async with ServiceContainer(config) as services:
        try:
            collection = services.collection_repo.get_by_name(args.collection)
            if not collection:
                raise CollectionNotFoundError(f"Collection '{args.collection}' not found")

            registry = get_default_registry()
            source = registry.create_source(collection)

            result = await services.indexing.index_collection(
                args.collection,
                force=args.force,
                embed=args.embed,
                source=source,
            )

            print(f"✓ Indexed {result.indexed} documents in '{args.collection}'")
            if result.skipped > 0:
                print(f"  Skipped {result.skipped} unchanged documents")
            if result.errors:
                print(f"  Errors: {len(result.errors)}")
                for path, error in result.errors[:5]:  # Show first 5 errors
                    print(f"    {path}: {error}")

        except CollectionNotFoundError as e:
            print(f"Error: {e}")
            raise SystemExit(1)


def handle_embed(args, config: Config) -> None:
    """Generate embeddings for documents.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_embed_async(args, config))


async def _handle_embed_async(args, config: Config) -> None:
    """Async handler for embedding generation.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    async with ServiceContainer(config) as services:
        try:
            result = await services.indexing.embed_collection(
                args.collection,
                force=args.force,
            )

            print(f"✓ Embedded {result.embedded} documents ({result.skipped} skipped)")
            if result.chunks_total > 0:
                print(f"  Total chunks: {result.chunks_total}")

        except CollectionNotFoundError as e:
            print(f"Error: {e}")
            raise SystemExit(1)
        except RuntimeError as e:
            print(f"Error: {e}")
            raise SystemExit(1)


def handle_cleanup(args, config: Config) -> None:
    """Clean up cache and orphaned data.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_cleanup_async(args, config))


async def _handle_cleanup_async(args, config: Config) -> None:
    """Async handler for cleanup.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    async with ServiceContainer(config) as services:
        result = await services.indexing.cleanup_orphans()

        print(f"✓ Cleaned up {result.orphaned_content} orphaned content hashes")
        if result.orphaned_embeddings > 0:
            print(f"  Removed {result.orphaned_embeddings} orphaned embeddings")


def handle_update_all(args, config: Config) -> None:
    """Update all collections.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_update_all_async(args, config))


async def _handle_update_all_async(args, config: Config) -> None:
    """Async handler for update all.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    async with ServiceContainer(config) as services:
        results = await services.indexing.update_all_collections(embed=args.embed)

        total = sum(r.indexed for r in results.values())
        for name, result in results.items():
            print(f"  {name}: {result.indexed} indexed, {result.skipped} skipped")

        print(f"✓ Updated {len(results)} collections ({total} documents indexed)")


def add_backfill_arguments(parser) -> None:
    """Add arguments for metadata backfill command.

    Args:
        parser: Argument parser for the command.
    """
    parser.add_argument(
        "collection",
        nargs="?",
        default=None,
        help="Collection name to backfill (all collections if not specified)",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Re-extract metadata even if already present",
    )


def handle_backfill_metadata(args, config: Config) -> None:
    """Backfill document metadata for existing documents.

    This migration extracts and stores metadata (tags, attributes) for
    documents that were indexed before the metadata tables existed.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    asyncio.run(_handle_backfill_async(args, config))


async def _handle_backfill_async(args, config: Config) -> None:
    """Async handler for metadata backfill.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    async with ServiceContainer(config) as services:
        stats = services.indexing.backfill_metadata(
            collection_name=args.collection,
            force=args.force,
        )

        print(f"Metadata backfill complete:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Updated: {stats['updated']}")
        print(f"  Skipped: {stats['skipped']}")
        if stats["errors"]:
            print(f"  Errors: {len(stats['errors'])}")
            for path, error in stats["errors"][:5]:  # Show first 5 errors
                print(f"    {path}: {error}")
