"""Collection management commands for PMD CLI."""

import json
from pathlib import Path
from typing import Any

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
        elif args.collection_cmd == "show":
            _show_collection(repo, db, args)
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
        args: Parsed arguments with name, path, glob, source, etc.
    """
    source_type = getattr(args, "source", "filesystem")

    if source_type == "filesystem":
        # Resolve to absolute path for filesystem sources
        resolved_path = str(Path(args.path).resolve())

        if not Path(resolved_path).exists():
            print(f"✗ Path does not exist: {resolved_path}")
            raise ValueError(f"Path does not exist: {resolved_path}")

        try:
            collection = repo.create(
                args.name,
                resolved_path,
                args.glob,
                source_type="filesystem",
            )
            print(f"✓ Created filesystem collection '{collection.name}'")
            print(f"  Path: {collection.pwd}")
            print(f"  Pattern: {collection.glob_pattern}")
        except CollectionExistsError as e:
            print(f"✗ {e}")
            raise

    elif source_type == "http":
        # Build source config for HTTP source
        source_config: dict[str, Any] = {
            "base_url": args.path,
        }

        if getattr(args, "sitemap", None):
            source_config["sitemap_url"] = args.sitemap

        if getattr(args, "auth_type", "none") != "none":
            source_config["auth_type"] = args.auth_type
            if getattr(args, "auth_token", None):
                source_config["auth_token"] = args.auth_token
            if getattr(args, "username", None):
                source_config["username"] = args.username

        try:
            collection = repo.create(
                args.name,
                args.path,  # Store original URL
                args.glob,
                source_type="http",
                source_config=source_config,
            )
            print(f"✓ Created HTTP collection '{collection.name}'")
            print(f"  URL: {args.path}")
            if source_config.get("sitemap_url"):
                print(f"  Sitemap: {source_config['sitemap_url']}")
            if source_config.get("auth_type"):
                print(f"  Auth: {source_config['auth_type']}")
        except CollectionExistsError as e:
            print(f"✗ {e}")
            raise

    elif source_type == "entity":
        # Build source config for entity source
        source_config = {
            "uri": args.path,
        }

        if getattr(args, "auth_type", "none") != "none":
            source_config["auth_type"] = args.auth_type
            if getattr(args, "auth_token", None):
                source_config["auth_token"] = args.auth_token

        try:
            collection = repo.create(
                args.name,
                args.path,
                args.glob,
                source_type="entity",
                source_config=source_config,
            )
            print(f"✓ Created entity collection '{collection.name}'")
            print(f"  URI: {args.path}")
        except CollectionExistsError as e:
            print(f"✗ {e}")
            raise

    else:
        print(f"✗ Unknown source type: {source_type}")
        raise ValueError(f"Unknown source type: {source_type}")


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
        source_indicator = {
            "filesystem": "📁",
            "http": "🌐",
            "entity": "🔗",
        }.get(coll.source_type, "❓")

        print(f"  {source_indicator} {coll.name}")
        if coll.source_type == "filesystem":
            print(f"    Path: {coll.pwd}")
            print(f"    Pattern: {coll.glob_pattern}")
        elif coll.source_type == "http":
            print(f"    URL: {coll.pwd}")
        else:
            print(f"    URI: {coll.pwd}")
        print(f"    Updated: {coll.updated_at}")


def _show_collection(repo: CollectionRepository, db: Database, args) -> None:
    """Show detailed collection information.

    Args:
        repo: CollectionRepository instance.
        db: Database instance.
        args: Parsed arguments with name.
    """
    collection = repo.get_by_name(args.name)
    if not collection:
        raise CollectionNotFoundError(f"Collection '{args.name}' not found")

    # Get document count
    cursor = db.execute(
        "SELECT COUNT(*) as count FROM documents WHERE collection_id = ? AND active = 1",
        (collection.id,),
    )
    doc_count = cursor.fetchone()["count"]

    # Get embedded document count
    cursor = db.execute(
        """
        SELECT COUNT(DISTINCT d.id) as count
        FROM documents d
        JOIN content_vectors cv ON d.hash = cv.hash
        WHERE d.collection_id = ? AND d.active = 1
        """,
        (collection.id,),
    )
    embedded_count = cursor.fetchone()["count"]

    print(f"Collection: {collection.name}")
    print(f"  ID: {collection.id}")
    print(f"  Source Type: {collection.source_type}")

    if collection.source_type == "filesystem":
        print(f"  Path: {collection.pwd}")
        print(f"  Pattern: {collection.glob_pattern}")
    elif collection.source_type == "http":
        print(f"  URL: {collection.pwd}")
        if collection.source_config:
            if collection.source_config.get("sitemap_url"):
                print(f"  Sitemap: {collection.source_config['sitemap_url']}")
            if collection.source_config.get("auth_type"):
                print(f"  Auth: {collection.source_config['auth_type']}")
    else:
        print(f"  URI: {collection.pwd}")

    print(f"  Documents: {doc_count}")
    print(f"  Embedded: {embedded_count}")
    print(f"  Created: {collection.created_at}")
    print(f"  Updated: {collection.updated_at}")

    if collection.source_config:
        # Show config without secrets
        safe_config = {
            k: "***" if "token" in k.lower() or "password" in k.lower() else v
            for k, v in collection.source_config.items()
        }
        print(f"  Config: {json.dumps(safe_config, indent=4)}")


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
