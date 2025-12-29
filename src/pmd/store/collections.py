"""Collection CRUD operations for PMD."""

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from ..core.exceptions import CollectionExistsError, CollectionNotFoundError
from ..core.types import Collection
from ..search.text import normalize_content
from .database import Database

if TYPE_CHECKING:
    from .documents import DocumentRepository
    from .search import FTS5SearchRepository


@dataclass
class IndexResult:
    """Result of indexing a collection."""

    indexed: int
    skipped: int
    errors: list[tuple[str, str]]


class CollectionRepository:
    """Repository for collection operations."""

    def __init__(self, db: Database):
        """Initialize with database connection.

        Args:
            db: Database instance to use for operations.
        """
        self.db = db

    def list_all(self) -> list[Collection]:
        """Get all collections.

        Returns:
            List of all Collection objects.
        """
        cursor = self.db.execute("SELECT * FROM collections ORDER BY name")
        return [self._row_to_collection(row) for row in cursor.fetchall()]

    def get_by_name(self, name: str) -> Collection | None:
        """Get collection by name.

        Args:
            name: Collection name to search for.

        Returns:
            Collection object if found, None otherwise.
        """
        cursor = self.db.execute("SELECT * FROM collections WHERE name = ?", (name,))
        row = cursor.fetchone()
        return self._row_to_collection(row) if row else None

    def get_by_id(self, collection_id: int) -> Collection | None:
        """Get collection by ID.

        Args:
            collection_id: Collection ID to search for.

        Returns:
            Collection object if found, None otherwise.
        """
        cursor = self.db.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
        row = cursor.fetchone()
        return self._row_to_collection(row) if row else None

    def create(
        self, name: str, pwd: str, glob_pattern: str = "**/*.md"
    ) -> Collection:
        """Create a new collection.

        Args:
            name: Unique name for the collection.
            pwd: Base directory path to index.
            glob_pattern: File pattern to match (default: **/*.md).

        Returns:
            Created Collection object.

        Raises:
            CollectionExistsError: If collection with this name already exists.
        """
        logger.debug(f"Creating collection: name={name!r}, pwd={pwd!r}, pattern={glob_pattern!r}")

        if self.get_by_name(name):
            raise CollectionExistsError(f"Collection '{name}' already exists")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            cursor.execute(
                """
                INSERT INTO collections (name, pwd, glob_pattern, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, pwd, glob_pattern, now, now),
            )
            collection_id = cursor.lastrowid

        logger.info(f"Collection created: id={collection_id}, name={name!r}")

        return Collection(
            id=collection_id,  # type: ignore
            name=name,
            pwd=pwd,
            glob_pattern=glob_pattern,
            created_at=now,
            updated_at=now,
        )

    def remove(self, collection_id: int) -> tuple[int, int]:
        """Remove a collection and clean up associated data.

        Args:
            collection_id: ID of collection to remove.

        Returns:
            Tuple of (documents_deleted, orphaned_hashes_cleaned).

        Raises:
            CollectionNotFoundError: If collection does not exist.
        """
        collection = self.get_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")

        logger.debug(f"Removing collection: id={collection_id}, name={collection.name!r}")

        with self.db.transaction() as cursor:
            # Get count of documents to delete
            cursor.execute("SELECT COUNT(*) FROM documents WHERE collection_id = ?", (collection_id,))
            docs_count = cursor.fetchone()[0]

            # Delete documents
            cursor.execute("DELETE FROM documents WHERE collection_id = ?", (collection_id,))

            # Delete path contexts
            cursor.execute("DELETE FROM path_contexts WHERE collection_id = ?", (collection_id,))

            # Find and delete orphaned content hashes
            cursor.execute(
                """
                SELECT DISTINCT hash FROM content
                WHERE hash NOT IN (SELECT DISTINCT hash FROM documents)
                """
            )
            orphaned_hashes = [row[0] for row in cursor.fetchall()]

            for hash_val in orphaned_hashes:
                cursor.execute("DELETE FROM content WHERE hash = ?", (hash_val,))
                cursor.execute("DELETE FROM content_vectors WHERE hash = ?", (hash_val,))

            # Delete the collection
            cursor.execute("DELETE FROM collections WHERE id = ?", (collection_id,))

        logger.info(f"Collection removed: name={collection.name!r}, docs={docs_count}, orphans={len(orphaned_hashes)}")

        return (docs_count, len(orphaned_hashes))

    def rename(self, collection_id: int, new_name: str) -> None:
        """Rename a collection.

        Args:
            collection_id: ID of collection to rename.
            new_name: New name for the collection.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            CollectionExistsError: If new name already exists.
        """
        collection = self.get_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")

        if new_name != collection.name and self.get_by_name(new_name):
            raise CollectionExistsError(f"Collection '{new_name}' already exists")

        logger.debug(f"Renaming collection: {collection.name!r} -> {new_name!r}")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            cursor.execute(
                "UPDATE collections SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, now, collection_id),
            )

        logger.info(f"Collection renamed: {collection.name!r} -> {new_name!r}")

    def update_collection_path(
        self, collection_id: int, pwd: str, glob_pattern: str | None = None
    ) -> None:
        """Update a collection's path and/or glob pattern.

        Args:
            collection_id: ID of collection to update.
            pwd: New base directory path.
            glob_pattern: New glob pattern (optional).

        Raises:
            CollectionNotFoundError: If collection does not exist.
        """
        collection = self.get_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")

        logger.debug(f"Updating collection path: name={collection.name!r}, pwd={pwd!r}, pattern={glob_pattern!r}")

        now = datetime.utcnow().isoformat()

        with self.db.transaction() as cursor:
            if glob_pattern:
                cursor.execute(
                    "UPDATE collections SET pwd = ?, glob_pattern = ?, updated_at = ? WHERE id = ?",
                    (pwd, glob_pattern, now, collection_id),
                )
            else:
                cursor.execute(
                    "UPDATE collections SET pwd = ?, updated_at = ? WHERE id = ?",
                    (pwd, now, collection_id),
                )

        logger.info(f"Collection path updated: name={collection.name!r}")

    @staticmethod
    def _row_to_collection(row: tuple) -> Collection:
        """Convert database row to Collection object.

        Args:
            row: Database row from sqlite3.Row.

        Returns:
            Collection object.
        """
        return Collection(
            id=row["id"],  # type: ignore
            name=row["name"],  # type: ignore
            pwd=row["pwd"],  # type: ignore
            glob_pattern=row["glob_pattern"],  # type: ignore
            created_at=row["created_at"],  # type: ignore
            updated_at=row["updated_at"],  # type: ignore
        ) 

    def _get_document_id(self, collection_id: int, path: str) -> int | None:
        """Get the database ID for a document by path.

        Args:
            collectio   n_id: Collection ID.
            path: Document path relative to collection.

        Returns:
            Document ID (integer primary key) or None if not found.
        """
        cursor = self.db.execute(
            "SELECT id FROM documents WHERE collection_id = ? AND path = ?",
            (collection_id, path),
        )
        row = cursor.fetchone()
        return row["id"] if row else None

    def index_documents(
        self,
        collection_id: int,
        doc_repo: "DocumentRepository",
        search_repo: "FTS5SearchRepository",
        force: bool = False,
    ) -> IndexResult:
        """Index all documents in a collection from the filesystem.

        Scans the collection's directory using its glob pattern, reads matching
        files, and stores them in the database with FTS5 indexing.

        Args:
            collection_id: ID of the collection to index.
            doc_repo: DocumentRepository for storing documents.
            search_repo: FTS5SearchRepository for full-text indexing.
            force: If True, reindex all documents even if unchanged.

        Returns:
            IndexResult with counts of indexed, skipped, and errored files.

        Raises:
            CollectionNotFoundError: If collection does not exist.
            ValueError: If collection path does not exist on filesystem.
        """
        collection = self.get_by_id(collection_id)
        if not collection:
            raise CollectionNotFoundError(f"Collection {collection_id} not found")

        collection_path = Path(collection.pwd)
        if not collection_path.exists():
            raise ValueError(f"Collection path does not exist: {collection_path}")

        logger.info(f"Indexing collection: name={collection.name!r}, path={collection_path}, force={force}")
        start_time = time.perf_counter()

        indexed_count = 0
        skipped_count = 0
        errors: list[tuple[str, str]] = []

        glob_pattern = collection.glob_pattern or "**/*.md"

        for file_path in collection_path.glob(glob_pattern):

            if not file_path.is_file():
                continue

            relative_path = str(file_path.relative_to(collection_path))

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (UnicodeDecodeError, IOError) as e:
                errors.append((relative_path, str(e)))
                logger.warning(f"Failed to read file: {relative_path}: {e}")
                continue

            # Extract title from first markdown heading or use filename
            title = self._extract_title(content, file_path.stem)

            # Check if document has been modified (skip if unchanged and not forcing)
            if not force:
                from ..utils.hashing import sha256_hash

                content_hash = sha256_hash(content)
                if not doc_repo.check_if_modified(collection_id, relative_path, content_hash):
                    skipped_count += 1
                    continue

            # Store document in database
            doc_result, _ = doc_repo.add_or_update(
                collection_id,
                relative_path,
                title,
                content,
            )

            # Get the document ID for FTS5 indexing
            # Use normalized fts_body so title-only docs remain searchable
            doc_id = self._get_document_id(collection_id, relative_path)
            if doc_id is not None:
                normalized = normalize_content(content)
                search_repo.index_document(doc_id, relative_path, normalized.fts_body)

            indexed_count += 1
            logger.debug(f"Indexed: {relative_path} ({len(content)} chars)")

        elapsed = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"Indexing complete: name={collection.name!r}, indexed={indexed_count}, "
            f"skipped={skipped_count}, errors={len(errors)}, {elapsed:.1f}ms"
        )

        return IndexResult(
            indexed=indexed_count,
            skipped=skipped_count,
            errors=errors,
        )

    @staticmethod
    def _extract_title(content: str, fallback: str) -> str:
        """Extract title from markdown content.

        Looks for the first line starting with '# ' and uses that as the title.
        Falls back to the provided fallback (typically the filename stem).

        Args:
            content: Markdown content to extract title from.
            fallback: Fallback title if no heading found.

        Returns:
            Extracted or fallback title.
        """
        for line in content.split("\n"):
            if line.startswith("# "):
                return line[2:].strip()
        return fallback
