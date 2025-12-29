"""Document indexing commands for PMD CLI."""

import asyncio

from ...core.config import Config
from ...core.exceptions import CollectionNotFoundError
from ...llm import create_llm_provider
from ...llm.embeddings import EmbeddingGenerator
from ...store.collections import CollectionRepository
from ...store.database import Database
from ...store.documents import DocumentRepository
from ...store.embeddings import EmbeddingRepository
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
        embedding_repo = EmbeddingRepository(db)
        search_repo = FTS5SearchRepository(db)

        # Get collection
        collection = coll_repo.get_by_name(args.collection)
        if not collection:
            raise CollectionNotFoundError(f"Collection '{args.collection}' not found")

        # Index documents from filesystem using repository method
        result = coll_repo.index_documents(
            collection.id,
            doc_repo,
            search_repo,
            args.force,
        )

        print(f"✓ Indexed {result.indexed} documents in '{args.collection}'")
        if result.skipped > 0:
            print(f"  Skipped {result.skipped} unchanged documents")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")
            for path, error in result.errors[:5]:  # Show first 5 errors
                print(f"    {path}: {error}")
    finally:
        db.close()


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
    db = Database(config.db_path)
    db.connect()

    llm_provider = None
    try:
        # Check if sqlite-vec is available
        if not db.vec_available:
            print("Vector storage not available (sqlite-vec extension not loaded)")
            return

        coll_repo = CollectionRepository(db)
        doc_repo = DocumentRepository(db)
        embedding_repo = EmbeddingRepository(db)

        # Get collection
        collection = coll_repo.get_by_name(args.collection)
        if not collection:
            raise CollectionNotFoundError(f"Collection '{args.collection}' not found")

        # Initialize LLM provider for embeddings
        llm_provider = create_llm_provider(config)

        # Check if LLM is available
        if not await llm_provider.is_available():
            print("LLM provider not available (is it running?)")
            return

        embedding_generator = EmbeddingGenerator(llm_provider, embedding_repo, config)

        # Get all documents in collection
        cursor = db.execute(
            """
            SELECT d.path, d.hash, c.doc
            FROM documents d
            JOIN content c ON d.hash = c.hash
            WHERE d.collection_id = ? AND d.active = 1
            """,
            (collection.id,),
        )
        documents = cursor.fetchall()

        embedded_count = 0
        skipped_count = 0

        for doc in documents:
            # Check if already embedded (unless force)
            if not args.force and embedding_repo.has_embeddings(doc["hash"]):
                skipped_count += 1
                continue

            # Generate and store embeddings
            chunks_embedded = await embedding_generator.embed_document(
                doc["hash"],
                doc["doc"],
            )

            if chunks_embedded > 0:
                embedded_count += 1
                print(f"  Embedded: {doc['path']} ({chunks_embedded} chunks)")

        print(f"✓ Embedded {embedded_count} documents ({skipped_count} skipped)")
    finally:
        if llm_provider:
            await llm_provider.close()
        db.close()


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
        embedding_repo = EmbeddingRepository(db)
        search_repo = FTS5SearchRepository(db)

        collections = coll_repo.list_all()
        total = 0

        for collection in collections:
            result = coll_repo.index_documents(
                collection.id,
                doc_repo,
                search_repo,
                force=False,
            )
            print(f"  {collection.name}: {result.indexed} indexed, {result.skipped} skipped")
            total += result.indexed

        print(f"✓ Updated {len(collections)} collections ({total} documents indexed)")
    finally:
        db.close()


