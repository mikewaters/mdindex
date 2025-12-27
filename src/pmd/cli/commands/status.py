"""Status command for PMD CLI."""

from ...core.config import Config
from ...core.types import IndexStatus
from ...store.collections import CollectionRepository
from ...store.database import Database


def handle_status(args, config: Config) -> None:
    """Handle status command.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        repo = CollectionRepository(db)
        status = _get_index_status(db, repo, config)
        _print_status(status)
    finally:
        db.close()


def _get_index_status(db: Database, repo: CollectionRepository, config: Config) -> IndexStatus:
    """Get current index status.

    Args:
        db: Database instance.
        repo: CollectionRepository instance.
        config: Application configuration.

    Returns:
        IndexStatus object with current state.
    """
    collections = repo.list_all()

    # Count total documents
    cursor = db.execute("SELECT COUNT(*) as count FROM documents WHERE active = 1")
    total_documents = cursor.fetchone()["count"]

    # Count embedded documents (Phase 2)
    embedded_documents = 0

    # Get database file size
    try:
        index_size_bytes = config.db_path.stat().st_size
    except (OSError, AttributeError):
        index_size_bytes = 0

    # Count cache entries (Phase 2)
    cache_entries = 0

    return IndexStatus(
        collections=collections,
        total_documents=total_documents,
        embedded_documents=embedded_documents,
        index_size_bytes=index_size_bytes,
        cache_entries=cache_entries,
        ollama_available=False,  # Phase 2
        models_available={},  # Phase 2
    )


def _print_status(status: IndexStatus) -> None:
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
    print(f"Cache Entries: {status.cache_entries}")
    print()

    if status.collections:
        print("Collections:")
        for coll in status.collections:
            print(f"  • {coll.name} ({coll.pwd})")
    else:
        print("No collections indexed yet.")
