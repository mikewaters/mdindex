"""idx.store.repositories - Repository classes for data access.

Repositories abstract database access patterns for Dataset and Document models.
Supports both explicit session injection and ambient session via contextvars.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from idx.store.models import Dataset, Document
from idx.store.session_context import current_session

__all__ = [
    "DatasetRepository",
    "DocumentRepository",
]


class DatasetRepository:
    """Repository for Dataset model operations.

    Can be initialized with an explicit session or use the ambient session
    from the current context (set via `use_session()`).
    """

    def __init__(self, session: Session | None = None) -> None:
        """Initialize with a database session.

        Args:
            session: SQLAlchemy session instance. If None, uses the
                ambient session from current_session().
        """
        self._explicit_session = session

    @property
    def _session(self) -> Session:
        """Get the session to use for database operations."""
        if self._explicit_session is not None:
            return self._explicit_session
        return current_session()

    def create(
        self,
        name: str,
        uri: str,
        source_type: str,
        source_path: str,
    ) -> Dataset:
        """Create a new dataset.

        Args:
            name: Unique name identifier.
            uri: Full URI for the dataset.
            source_type: Type of source.
            source_path: Filesystem path to the source.

        Returns:
            The created Dataset instance.
        """
        dataset = Dataset(
            name=name,
            uri=uri,
            source_type=source_type,
            source_path=source_path,
        )
        self._session.add(dataset)
        return dataset

    def get_by_id(self, dataset_id: int) -> Dataset | None:
        """Get a dataset by ID.

        Args:
            dataset_id: The dataset's primary key.

        Returns:
            The Dataset if found, None otherwise.
        """
        return self._session.get(Dataset, dataset_id)

    def get_by_name(self, name: str) -> Dataset | None:
        """Get a dataset by name.

        Args:
            name: The dataset's unique name.

        Returns:
            The Dataset if found, None otherwise.
        """
        stmt = select(Dataset).where(Dataset.name == name)
        return self._session.execute(stmt).scalar_one_or_none()

    def get_by_uri(self, uri: str) -> Dataset | None:
        """Get a dataset by URI.

        Args:
            uri: The dataset's URI.

        Returns:
            The Dataset if found, None otherwise.
        """
        stmt = select(Dataset).where(Dataset.uri == uri)
        return self._session.execute(stmt).scalar_one_or_none()

    def list_all(self) -> list[Dataset]:
        """List all datasets.

        Returns:
            List of all Dataset instances.
        """
        stmt = select(Dataset).order_by(Dataset.name)
        return list(self._session.execute(stmt).scalars().all())

    def delete(self, dataset: Dataset) -> None:
        """Delete a dataset.

        Args:
            dataset: The dataset to delete.
        """
        self._session.delete(dataset)

    def exists_by_name(self, name: str) -> bool:
        """Check if a dataset exists with the given name.

        Args:
            name: The dataset name to check.

        Returns:
            True if a dataset with this name exists.
        """
        stmt = select(func.count()).select_from(Dataset).where(Dataset.name == name)
        count = self._session.execute(stmt).scalar()
        return count is not None and count > 0


class DocumentRepository:
    """Repository for Document model operations.

    Can be initialized with an explicit session or use the ambient session
    from the current context (set via `use_session()`).
    """

    def __init__(self, session: Session | None = None) -> None:
        """Initialize with a database session.

        Args:
            session: SQLAlchemy session instance. If None, uses the
                ambient session from current_session().
        """
        self._explicit_session = session

    @property
    def _session(self) -> Session:
        """Get the session to use for database operations."""
        if self._explicit_session is not None:
            return self._explicit_session
        return current_session()

    def create(
        self,
        dataset_id: int,
        path: str,
        content_hash: str,
        body: str,
        *,
        etag: str | None = None,
        last_modified: datetime | None = None,
        metadata_json: str | None = None,
    ) -> Document:
        """Create a new document.

        Args:
            dataset_id: Parent dataset ID.
            path: Relative path within the dataset.
            content_hash: SHA256 hash of content.
            body: Full normalized text content.
            etag: Optional source etag.
            last_modified: Optional source modification time.
            metadata_json: Optional JSON metadata string.

        Returns:
            The created Document instance.
        """
        doc = Document(
            dataset_id=dataset_id,
            path=path,
            content_hash=content_hash,
            body=body,
            etag=etag,
            last_modified=last_modified,
            metadata_json=metadata_json,
        )
        self._session.add(doc)
        return doc

    def get_by_id(self, doc_id: int) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: The document's primary key.

        Returns:
            The Document if found, None otherwise.
        """
        return self._session.get(Document, doc_id)

    def get_by_path(self, dataset_id: int, path: str) -> Document | None:
        """Get a document by dataset and path.

        Args:
            dataset_id: The parent dataset's ID.
            path: The document's path within the dataset.

        Returns:
            The Document if found, None otherwise.
        """
        stmt = select(Document).where(
            Document.dataset_id == dataset_id,
            Document.path == path,
        )
        return self._session.execute(stmt).scalar_one_or_none()

    def list_by_dataset(
        self,
        dataset_id: int,
        *,
        active_only: bool = False,
    ) -> list[Document]:
        """List documents in a dataset.

        Args:
            dataset_id: The parent dataset's ID.
            active_only: If True, only return active documents.

        Returns:
            List of Document instances.
        """
        stmt = select(Document).where(Document.dataset_id == dataset_id)
        if active_only:
            stmt = stmt.where(Document.active == True)  # noqa: E712
        stmt = stmt.order_by(Document.path)
        return list(self._session.execute(stmt).scalars().all())

    def list_paths_by_dataset(
        self,
        dataset_id: int,
        *,
        active_only: bool = False,
    ) -> set[str]:
        """List document paths in a dataset.

        Args:
            dataset_id: The parent dataset's ID.
            active_only: If True, only return active documents.

        Returns:
            Set of document paths.
        """
        stmt = select(Document.path).where(Document.dataset_id == dataset_id)
        if active_only:
            stmt = stmt.where(Document.active == True)  # noqa: E712
        return set(self._session.execute(stmt).scalars().all())

    def update(
        self,
        doc: Document,
        *,
        content_hash: str | None = None,
        body: str | None = None,
        etag: str | None = None,
        last_modified: datetime | None = None,
        metadata_json: str | None = None,
        active: bool | None = None,
    ) -> Document:
        """Update a document.

        Args:
            doc: The document to update.
            content_hash: New content hash if changed.
            body: New text content if changed.
            etag: New etag if changed.
            last_modified: New modification time if changed.
            metadata_json: New metadata JSON if changed.
            active: New active status if changed.

        Returns:
            The updated Document instance.
        """
        if content_hash is not None:
            doc.content_hash = content_hash
        if body is not None:
            doc.body = body
        if etag is not None:
            doc.etag = etag
        if last_modified is not None:
            doc.last_modified = last_modified
        if metadata_json is not None:
            doc.metadata_json = metadata_json
        if active is not None:
            doc.active = active
        return doc

    def soft_delete(self, doc: Document) -> Document:
        """Soft-delete a document by setting active=False.

        Args:
            doc: The document to soft-delete.

        Returns:
            The updated Document instance.
        """
        doc.active = False
        return doc

    def soft_delete_by_paths(
        self,
        dataset_id: int,
        paths: set[str],
    ) -> int:
        """Soft-delete multiple documents by path.

        Args:
            dataset_id: The parent dataset's ID.
            paths: Set of paths to soft-delete.

        Returns:
            Number of documents soft-deleted.
        """
        if not paths:
            return 0

        stmt = (
            select(Document)
            .where(
                Document.dataset_id == dataset_id,
                Document.path.in_(paths),
                Document.active == True,  # noqa: E712
            )
        )
        docs = self._session.execute(stmt).scalars().all()
        for doc in docs:
            doc.active = False
        return len(docs)

    def hard_delete_by_paths(
        self,
        dataset_id: int,
        paths: set[str],
    ) -> int:
        """Hard-delete multiple documents by path.

        Permanently removes documents from the database.

        Args:
            dataset_id: The parent dataset's ID.
            paths: Set of paths to delete.

        Returns:
            Number of documents deleted.
        """
        if not paths:
            return 0

        stmt = (
            select(Document)
            .where(
                Document.dataset_id == dataset_id,
                Document.path.in_(paths),
            )
        )
        docs = list(self._session.execute(stmt).scalars().all())
        for doc in docs:
            self._session.delete(doc)
        return len(docs)

    def delete(self, doc: Document) -> None:
        """Hard-delete a document.

        Args:
            doc: The document to delete.
        """
        self._session.delete(doc)

    def count_by_dataset(
        self,
        dataset_id: int,
        *,
        active_only: bool = False,
    ) -> int:
        """Count documents in a dataset.

        Args:
            dataset_id: The parent dataset's ID.
            active_only: If True, only count active documents.

        Returns:
            Number of documents.
        """
        stmt = select(func.count()).select_from(Document).where(
            Document.dataset_id == dataset_id
        )
        if active_only:
            stmt = stmt.where(Document.active == True)  # noqa: E712
        count = self._session.execute(stmt).scalar()
        return count or 0
