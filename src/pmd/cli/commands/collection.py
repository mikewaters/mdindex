"""Collection management commands for PMD CLI."""

import json
from pathlib import Path
from typing import Any

from pmd.core.config import Config
from pmd.core.exceptions import SourceCollectionExistsError, SourceCollectionNotFoundError
from pmd.store.repositories.collections import SourceCollectionRepository
from pmd.store.database import Database


def _print_patterns(patterns: list[str]) -> None:
    """Print glob patterns in a readable format.

    Args:
        patterns: List of glob patterns.
    """
    if len(patterns) == 1:
        print(f"  Pattern: {patterns[0]}")
    else:
        print("  Patterns:")
        for pattern in patterns:
            prefix = "    !" if pattern.startswith("!") else "    "
            display = pattern[1:] if pattern.startswith("!") else pattern
            if pattern.startswith("!"):
                print(f"    !{display} (exclude)")
            else:
                print(f"    {pattern}")


def _format_patterns_brief(patterns: list[str]) -> str:
    """Format patterns for brief display (list command).

    Args:
        patterns: List of glob patterns.

    Returns:
        Comma-separated pattern string, truncated if needed.
    """
    if len(patterns) == 1:
        return patterns[0]

    includes = [p for p in patterns if not p.startswith("!")]
    excludes = [p for p in patterns if p.startswith("!")]

    parts = includes[:2]  # Show up to 2 include patterns
    if len(includes) > 2:
        parts.append(f"+{len(includes) - 2} more")
    if excludes:
        parts.append(f"{len(excludes)} excludes")

    return ", ".join(parts)


def _get_source_collection_id(db: Database, name: str) -> int:
    """Get source collection ID by name.

    Args:
        db: Database instance.
        name: Source collection name.

    Returns:
        Source collection ID.

    Raises:
        SourceCollectionNotFoundError: If source collection not found.
    """
    repo = SourceCollectionRepository(db)
    source_collection = repo.get_by_name(name)
    if not source_collection:
        raise SourceCollectionNotFoundError(f"Source collection '{name}' not found")
    return source_collection.id


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
        repo = SourceCollectionRepository(db)

        if args.collection_cmd == "add":
            _add_source_collection(repo, args)
        elif args.collection_cmd == "list":
            _list_source_collections(repo)
        elif args.collection_cmd == "show":
            _show_source_collection(repo, db, args)
        elif args.collection_cmd == "remove":
            _remove_source_collection(repo, args)
        elif args.collection_cmd == "rename":
            _rename_source_collection(repo, args)
    finally:
        db.close()


def _add_source_collection(repo: SourceCollectionRepository, args) -> None:
    """Add a new source collection.

    Args:
        repo: SourceCollectionRepository instance.
        args: Parsed arguments with name, path, globs, source, etc.
    """
    source_type = getattr(args, "source", "filesystem")

    # Get glob patterns from args (may be None if not specified)
    glob_patterns = getattr(args, "globs", None) or ["**/*.md"]

    if source_type == "filesystem":
        # Resolve to absolute path for filesystem sources
        resolved_path = str(Path(args.path).resolve())

        if not Path(resolved_path).exists():
            print(f"Path does not exist: {resolved_path}")
            raise ValueError(f"Path does not exist: {resolved_path}")

        try:
            source_collection = repo.create(
                args.name,
                resolved_path,
                glob_patterns,
                source_type="filesystem",
            )
            print(f"Created filesystem collection '{source_collection.name}'")
            print(f"  Path: {source_collection.pwd}")
            _print_patterns(source_collection.glob_patterns)
        except SourceCollectionExistsError as e:
            print(f"Error: {e}")
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
            source_collection = repo.create(
                args.name,
                args.path,  # Store original URL
                glob_patterns,
                source_type="http",
                source_config=source_config,
            )
            print(f"Created HTTP collection '{source_collection.name}'")
            print(f"  URL: {args.path}")
            if source_config.get("sitemap_url"):
                print(f"  Sitemap: {source_config['sitemap_url']}")
            if source_config.get("auth_type"):
                print(f"  Auth: {source_config['auth_type']}")
        except SourceCollectionExistsError as e:
            print(f"Error: {e}")
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
            source_collection = repo.create(
                args.name,
                args.path,
                glob_patterns,
                source_type="entity",
                source_config=source_config,
            )
            print(f"Created entity collection '{source_collection.name}'")
            print(f"  URI: {args.path}")
        except SourceCollectionExistsError as e:
            print(f"Error: {e}")
            raise

    else:
        print(f"Unknown source type: {source_type}")
        raise ValueError(f"Unknown source type: {source_type}")


def _list_source_collections(repo: SourceCollectionRepository) -> None:
    """List all source collections.

    Args:
        repo: SourceCollectionRepository instance.
    """
    source_collections = repo.list_all()

    if not source_collections:
        print("No collections found.")
        return

    print(f"Collections ({len(source_collections)}):")
    for coll in source_collections:
        source_indicator = {
            "filesystem": "[fs]",
            "http": "[http]",
            "entity": "[entity]",
        }.get(coll.source_type, "[?]")

        print(f"  {source_indicator} {coll.name}")
        if coll.source_type == "filesystem":
            print(f"    Path: {coll.pwd}")
            print(f"    Pattern: {_format_patterns_brief(coll.glob_patterns)}")
        elif coll.source_type == "http":
            print(f"    URL: {coll.pwd}")
        else:
            print(f"    URI: {coll.pwd}")
        print(f"    Updated: {coll.updated_at}")


def _show_source_collection(repo: SourceCollectionRepository, db: Database, args) -> None:
    """Show detailed source collection information.

    Args:
        repo: SourceCollectionRepository instance.
        db: Database instance.
        args: Parsed arguments with name.
    """
    source_collection = repo.get_by_name(args.name)
    if not source_collection:
        raise SourceCollectionNotFoundError(f"Source collection '{args.name}' not found")

    # Get document count
    cursor = db.execute(
        "SELECT COUNT(*) as count FROM documents WHERE source_collection_id = ? AND active = 1",
        (source_collection.id,),
    )
    doc_count = cursor.fetchone()["count"]

    # Get embedded document count
    cursor = db.execute(
        """
        SELECT COUNT(DISTINCT d.id) as count
        FROM documents d
        JOIN content_vectors cv ON d.hash = cv.hash
        WHERE d.source_collection_id = ? AND d.active = 1
        """,
        (source_collection.id,),
    )
    embedded_count = cursor.fetchone()["count"]

    print(f"Collection: {source_collection.name}")
    print(f"  ID: {source_collection.id}")
    print(f"  Source Type: {source_collection.source_type}")

    if source_collection.source_type == "filesystem":
        print(f"  Path: {source_collection.pwd}")
        _print_patterns(source_collection.glob_patterns)
    elif source_collection.source_type == "http":
        print(f"  URL: {source_collection.pwd}")
        if source_collection.source_config:
            if source_collection.source_config.get("sitemap_url"):
                print(f"  Sitemap: {source_collection.source_config['sitemap_url']}")
            if source_collection.source_config.get("auth_type"):
                print(f"  Auth: {source_collection.source_config['auth_type']}")
    else:
        print(f"  URI: {source_collection.pwd}")

    print(f"  Documents: {doc_count}")
    print(f"  Embedded: {embedded_count}")
    print(f"  Created: {source_collection.created_at}")
    print(f"  Updated: {source_collection.updated_at}")

    if source_collection.source_config:
        # Show config without secrets
        safe_config = {
            k: "***" if "token" in k.lower() or "password" in k.lower() else v
            for k, v in source_collection.source_config.items()
        }
        print(f"  Config: {json.dumps(safe_config, indent=4)}")


def _remove_source_collection(repo: SourceCollectionRepository, args) -> None:
    """Remove a source collection.

    Args:
        repo: SourceCollectionRepository instance.
        args: Parsed arguments with name.
    """
    source_collection = repo.get_by_name(args.name)
    if not source_collection:
        raise SourceCollectionNotFoundError(f"Source collection '{args.name}' not found")

    docs_deleted, hashes_cleaned = repo.remove(source_collection.id)
    print(f"Removed collection '{args.name}'")
    print(f"  Documents deleted: {docs_deleted}")
    print(f"  Orphaned hashes cleaned: {hashes_cleaned}")


def _rename_source_collection(repo: SourceCollectionRepository, args) -> None:
    """Rename a source collection.

    Args:
        repo: SourceCollectionRepository instance.
        args: Parsed arguments with old_name, new_name.
    """
    source_collection = repo.get_by_name(args.old_name)
    if not source_collection:
        raise SourceCollectionNotFoundError(f"Source collection '{args.old_name}' not found")

    repo.rename(source_collection.id, args.new_name)
    print(f"Renamed collection '{args.old_name}' to '{args.new_name}'")
