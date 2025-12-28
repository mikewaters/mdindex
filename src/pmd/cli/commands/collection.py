"""Collection management commands for PMD CLI."""

from pathlib import Path

from ...core.config import Config
from ...core.exceptions import CollectionExistsError, CollectionNotFoundError
from ...store.collections import CollectionRepository
from ...store.database import Database


def _get_collection_id(db: Database, name: str) -> int:
    """Get collection ID by name.

    Args:
        db: Database instance.
        name: Collection name.

    Returns:
        Collection ID.

    Raises:
        CollectionNotFoundError: If collection not found.
    """
    repo = CollectionRepository(db)
    collection = repo.get_by_name(name)
    if not collection:
        raise CollectionNotFoundError(f"Collection '{name}' not found")
    return collection.id


def handle_collection(args, config: Config) -> None:
    """Handle collection subcommands.

    Args:
        args: Parsed command arguments.
        config: Application configuration.

    Raises:
        Various PMD exceptions.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        repo = CollectionRepository(db)

        if args.collection_cmd == "add":
            _add_collection(repo, args)
        elif args.collection_cmd == "list":
            _list_collections(repo)
        elif args.collection_cmd == "remove":
            _remove_collection(repo, args)
        elif args.collection_cmd == "rename":
            _rename_collection(repo, args)
    finally:
        db.close()


def _add_collection(repo: CollectionRepository, args) -> None:
    """Add a new collection.

    Args:
        repo: CollectionRepository instance.
        args: Parsed arguments with name, path, glob.
    """
    # Resolve to absolute path to avoid issues when indexing from different directories
    resolved_path = str(Path(args.path).resolve())

    if not Path(resolved_path).exists():
        print(f"✗ Path does not exist: {resolved_path}")
        raise ValueError(f"Path does not exist: {resolved_path}")

    try:
        collection = repo.create(args.name, resolved_path, args.glob)
        print(f"✓ Created collection '{collection.name}'")
        print(f"  Path: {collection.pwd}")
        print(f"  Pattern: {collection.glob_pattern}")
    except CollectionExistsError as e:
        print(f"✗ {e}")
        raise


def _list_collections(repo: CollectionRepository) -> None:
    """List all collections.

    Args:
        repo: CollectionRepository instance.
    """
    collections = repo.list_all()

    if not collections:
        print("No collections found.")
        return

    print(f"Collections ({len(collections)}):")
    for coll in collections:
        print(f"  {coll.name}")
        print(f"    Path: {coll.pwd}")
        print(f"    Pattern: {coll.glob_pattern}")
        print(f"    Updated: {coll.updated_at}")


def _remove_collection(repo: CollectionRepository, args) -> None:
    """Remove a collection.

    Args:
        repo: CollectionRepository instance.
        args: Parsed arguments with name.
    """
    collection = repo.get_by_name(args.name)
    if not collection:
        raise CollectionNotFoundError(f"Collection '{args.name}' not found")

    docs_deleted, hashes_cleaned = repo.remove(collection.id)
    print(f"✓ Removed collection '{args.name}'")
    print(f"  Documents deleted: {docs_deleted}")
    print(f"  Orphaned hashes cleaned: {hashes_cleaned}")


def _rename_collection(repo: CollectionRepository, args) -> None:
    """Rename a collection.

    Args:
        repo: CollectionRepository instance.
        args: Parsed arguments with old_name, new_name.
    """
    collection = repo.get_by_name(args.old_name)
    if not collection:
        raise CollectionNotFoundError(f"Collection '{args.old_name}' not found")

    repo.rename(collection.id, args.new_name)
    print(f"✓ Renamed collection '{args.old_name}' to '{args.new_name}'")
