"""Status command for PMD CLI."""

import asyncio

from ...core.config import Config
from ...services import ServiceContainer


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
    async with ServiceContainer(config) as services:
        status = services.status.get_index_status()
        _print_status(status)


def _print_status(status) -> None:
    """Print status information.

    Args:
        status: IndexStatus object to display.
    """
    print("PMD Index Status")
    print("=" * 50)
    print(f"Collections: {len(status.collections)}")
    print(f"Documents: {status.total_documents}")
    print(f"Embedded: {status.embedded_documents}")
    print(f"Index Size: {status.index_size_bytes} bytes")
    print(f"Embeddings: {status.cache_entries}")
    print()

    if status.collections:
        print("Collections:")
        for coll in status.collections:
            print(f"  • {coll.name} ({coll.pwd})")
    else:
        print("No collections indexed yet.")
