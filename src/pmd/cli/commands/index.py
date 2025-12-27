"""Document indexing commands for PMD CLI."""

from pathlib import Path

from ...core.config import Config
from ...core.exceptions import CollectionNotFoundError
from ...store.collections import CollectionRepository
from ...store.database import Database
from ...store.documents import DocumentRepository
from ...store.search import FTS5SearchRepository


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


def handle_index_collection(args, config: Config) -> None:
    """Index all documents in a collection.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        coll_repo = CollectionRepository(db)
        doc_repo = DocumentRepository(db)
        search_repo = FTS5SearchRepository(db)

        # Get collection
        collection = coll_repo.get_by_name(args.collection)
        if not collection:
            raise CollectionNotFoundError(f"Collection '{args.collection}' not found")

        # Index documents from filesystem
        count = _index_collection_files(
            collection,
            doc_repo,
            search_repo,
            args.force,
        )

        print(f"✓ Indexed {count} documents in '{args.collection}'")
    finally:
        db.close()


def handle_embed(args, config: Config) -> None:
    """Generate embeddings for documents.

    Phase 3: Will use Ollama to generate embeddings.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    print("Embedding not yet available (Phase 3)")


def handle_cleanup(args, config: Config) -> None:
    """Clean up cache and orphaned data.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        coll_repo = CollectionRepository(db)

        # Find orphaned hashes (referenced by no documents)
        cursor = db.execute(
            """
            SELECT COUNT(*) as count FROM content
            WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
            """
        )
        orphaned_count = cursor.fetchone()["count"]

        if orphaned_count > 0:
            cursor = db.execute(
                """
                SELECT hash FROM content
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents WHERE active = 1)
                """
            )

            with db.transaction() as cursor:
                for row in cursor.fetchall():
                    cursor.execute("DELETE FROM content WHERE hash = ?", (row["hash"],))
                    cursor.execute(
                        "DELETE FROM content_vectors WHERE hash = ?", (row["hash"],)
                    )

        print(f"✓ Cleaned up {orphaned_count} orphaned hashes")
    finally:
        db.close()


def handle_update_all(args, config: Config) -> None:
    """Update all collections.

    Args:
        args: Parsed command arguments.
        config: Application configuration.
    """
    db = Database(config.db_path)
    db.connect()

    try:
        coll_repo = CollectionRepository(db)
        doc_repo = DocumentRepository(db)
        search_repo = FTS5SearchRepository(db)

        collections = coll_repo.list_all()
        total = 0

        for collection in collections:
            count = _index_collection_files(
                collection,
                doc_repo,
                search_repo,
                force=False,
            )
            print(f"  {collection.name}: {count} documents")
            total += count

        print(f"✓ Updated {len(collections)} collections ({total} documents)")
    finally:
        db.close()


def _index_collection_files(
    collection,
    doc_repo: DocumentRepository,
    search_repo: FTS5SearchRepository,
    force: bool = False,
) -> int:
    """Index all files in a collection.

    Args:
        collection: Collection object to index.
        doc_repo: DocumentRepository instance.
        search_repo: SearchRepository instance.
        force: Force reindex of all documents.

    Returns:
        Number of documents indexed.
    """
    collection_path = Path(collection.pwd)
    if not collection_path.exists():
        raise ValueError(f"Collection path does not exist: {collection_path}")

    indexed_count = 0

    # Find all matching files
    glob_pattern = collection.glob_pattern or "**/*.md"
    for file_path in collection_path.glob(glob_pattern):
        if not file_path.is_file():
            continue

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
        except (UnicodeDecodeError, IOError):
            continue

        # Get relative path
        relative_path = str(file_path.relative_to(collection_path))

        # Extract title from first line or filename
        lines = content.split("\n")
        title = None
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        if not title:
            title = file_path.stem

        # Check if document has been modified
        if not force:
            modified = doc_repo.check_if_modified(collection.id, relative_path, None)
            if not modified:
                continue

        # Store document
        doc_result, is_new = doc_repo.add_or_update(
            collection.id,
            relative_path,
            title,
            content,
        )

        # Index in FTS5
        search_repo.index_document(
            doc_result.filepath,
            relative_path,
            content,
        )

        indexed_count += 1

    return indexed_count
