"""idx.store.service - DatasetService for high-level data operations.

The DatasetService provides a Pydantic model-based API for managing
datasets and documents, delegating to repositories for persistence.
"""

import json
import re
from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy.orm import Session

from idx.core.logging import get_logger
from idx.store.database import get_session
from idx.store.models import Dataset, Document
from idx.store.repositories import DatasetRepository, DocumentRepository
from idx.store.schemas import (
    DatasetCreate,
    DatasetInfo,
    DocumentCreate,
    DocumentInfo,
    DocumentUpdate,
)

__all__ = [
    "DatasetService",
    "DatasetNotFoundError",
    "DatasetExistsError",
    "DocumentNotFoundError",
    "normalize_dataset_name",
]

logger = get_logger(__name__)


class DatasetNotFoundError(Exception):
    """Raised when a dataset is not found."""

    def __init__(self, identifier: str | int) -> None:
        self.identifier = identifier
        super().__init__(f"Dataset not found: {identifier}")


class DatasetExistsError(Exception):
    """Raised when a dataset already exists with the given name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Dataset already exists: {name}")


class DocumentNotFoundError(Exception):
    """Raised when a document is not found."""

    def __init__(self, identifier: str | int, dataset_id: int | None = None) -> None:
        self.identifier = identifier
        self.dataset_id = dataset_id
        if dataset_id:
            super().__init__(f"Document not found: {identifier} in dataset {dataset_id}")
        else:
            super().__init__(f"Document not found: {identifier}")


def normalize_dataset_name(name: str) -> str:
    """Normalize a dataset name to URI-acceptable format.

    Converts the name to lowercase, replaces spaces and special characters
    with hyphens, and removes any leading/trailing hyphens.

    Args:
        name: The raw dataset name.

    Returns:
        The normalized dataset name.

    Example:
        >>> normalize_dataset_name("My Obsidian Vault")
        'my-obsidian-vault'
    """
    # Lowercase
    normalized = name.lower()
    # Replace spaces and underscores with hyphens
    normalized = re.sub(r"[\s_]+", "-", normalized)
    # Remove any characters that aren't alphanumeric or hyphens
    normalized = re.sub(r"[^a-z0-9-]", "", normalized)
    # Collapse multiple hyphens
    normalized = re.sub(r"-+", "-", normalized)
    # Strip leading/trailing hyphens
    normalized = normalized.strip("-")
    return normalized


class DatasetService:
    """Service for managing datasets and documents.

    Provides a high-level API using Pydantic models for input/output,
    with transactions managed at the service method level.

    Example:
        service = DatasetService()

        # Create a dataset
        dataset = service.create_dataset(
            DatasetCreate(
                name="my-vault",
                source_type="obsidian",
                source_path="/path/to/vault",
            )
        )

        # Add a document
        doc = service.create_document(
            dataset.id,
            DocumentCreate(
                path="notes/test.md",
                content_hash="abc123",
                body="# Test",
            )
        )
    """

    def __init__(self, session: Session | None = None) -> None:
        """Initialize the service.

        Args:
            session: Optional SQLAlchemy session. If not provided,
                    uses the default session from get_session().
        """
        self._external_session = session

    @contextmanager
    def _get_session(self) -> Generator[Session, None, None]:
        """Get a session for operations.

        If an external session was provided, use it directly.
        Otherwise, create a new session with get_session().
        """
        if self._external_session is not None:
            yield self._external_session
        else:
            with get_session() as session:
                yield session

    # Dataset operations

    def create_dataset(self, data: DatasetCreate) -> DatasetInfo:
        """Create a new dataset.

        Args:
            data: Dataset creation data.

        Returns:
            The created dataset info.

        Raises:
            DatasetExistsError: If a dataset with this name already exists.
        """
        normalized_name = normalize_dataset_name(data.name)
        uri = f"dataset:{normalized_name}"

        with self._get_session() as session:
            repo = DatasetRepository(session)

            if repo.exists_by_name(normalized_name):
                raise DatasetExistsError(normalized_name)

            dataset = repo.create(
                name=normalized_name,
                uri=uri,
                source_type=data.source_type,
                source_path=data.source_path,
            )
            session.flush()

            logger.info(f"Created dataset: {normalized_name}")
            return self._dataset_to_info(dataset, session)

    def get_dataset(self, dataset_id: int) -> DatasetInfo:
        """Get a dataset by ID.

        Args:
            dataset_id: The dataset's ID.

        Returns:
            The dataset info.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
        """
        with self._get_session() as session:
            repo = DatasetRepository(session)
            dataset = repo.get_by_id(dataset_id)
            if dataset is None:
                raise DatasetNotFoundError(dataset_id)
            return self._dataset_to_info(dataset, session)

    def get_dataset_by_name(self, name: str) -> DatasetInfo:
        """Get a dataset by name.

        Args:
            name: The dataset's name (will be normalized).

        Returns:
            The dataset info.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
        """
        normalized_name = normalize_dataset_name(name)
        with self._get_session() as session:
            repo = DatasetRepository(session)
            dataset = repo.get_by_name(normalized_name)
            if dataset is None:
                raise DatasetNotFoundError(normalized_name)
            return self._dataset_to_info(dataset, session)

    def list_datasets(self) -> list[DatasetInfo]:
        """List all datasets.

        Returns:
            List of dataset info objects.
        """
        with self._get_session() as session:
            repo = DatasetRepository(session)
            datasets = repo.list_all()
            return [self._dataset_to_info(ds, session) for ds in datasets]

    def delete_dataset(self, dataset_id: int) -> None:
        """Delete a dataset and all its documents.

        Args:
            dataset_id: The dataset's ID.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
        """
        with self._get_session() as session:
            repo = DatasetRepository(session)
            dataset = repo.get_by_id(dataset_id)
            if dataset is None:
                raise DatasetNotFoundError(dataset_id)

            name = dataset.name
            repo.delete(dataset)
            logger.info(f"Deleted dataset: {name}")

    def _dataset_to_info(self, dataset: Dataset, session: Session) -> DatasetInfo:
        """Convert Dataset model to DatasetInfo."""
        doc_repo = DocumentRepository(session)
        doc_count = doc_repo.count_by_dataset(dataset.id, active_only=True)
        return DatasetInfo(
            id=dataset.id,
            name=dataset.name,
            uri=dataset.uri,
            source_type=dataset.source_type,
            source_path=dataset.source_path,
            created_at=dataset.created_at,
            updated_at=dataset.updated_at,
            document_count=doc_count,
        )
    
    @staticmethod
    def create_or_update(
        session: Session,
        name: str,
        source_type: str,
        source_path: str,
    ) -> int:
        """Ensure dataset exists, creating if necessary.

        Args:
            session: SQLAlchemy session.
            name: Dataset name.
            source_type: Type of source (e.g., "directory", "obsidian").
            source_path: Path to the source.

        Returns:
            Dataset ID.
        """
        normalized_name = normalize_dataset_name(name)
        repo = DatasetRepository(session)

        # Check if dataset exists
        dataset = repo.get_by_name(normalized_name)
        if dataset is not None:
            logger.debug(f"Using existing dataset: {normalized_name}")
            return dataset.id

        # Create new dataset
        dataset = repo.create(
            name=normalized_name,
            uri=f"dataset:{normalized_name}",
            source_type=source_type,
            source_path=source_path,
        )
        session.flush()
        logger.info(f"Created dataset: {normalized_name}")
        return dataset.id

    # Document operations

    def create_document(
        self,
        dataset_id: int,
        data: DocumentCreate,
    ) -> DocumentInfo:
        """Create a new document in a dataset.

        Args:
            dataset_id: The parent dataset's ID.
            data: Document creation data.

        Returns:
            The created document info.

        Raises:
            DatasetNotFoundError: If the dataset is not found.
        """
        with self._get_session() as session:
            ds_repo = DatasetRepository(session)
            if ds_repo.get_by_id(dataset_id) is None:
                raise DatasetNotFoundError(dataset_id)

            doc_repo = DocumentRepository(session)
            metadata_json = json.dumps(data.metadata) if data.metadata else None

            doc = doc_repo.create(
                dataset_id=dataset_id,
                path=data.path,
                content_hash=data.content_hash,
                body=data.body,
                etag=data.etag,
                last_modified=data.last_modified,
                metadata_json=metadata_json,
            )
            session.flush()

            logger.debug(f"Created document: {data.path} in dataset {dataset_id}")
            return DocumentInfo.from_orm_model(doc)

    def get_document(self, doc_id: int) -> DocumentInfo:
        """Get a document by ID.

        Args:
            doc_id: The document's ID.

        Returns:
            The document info.

        Raises:
            DocumentNotFoundError: If the document is not found.
        """
        with self._get_session() as session:
            repo = DocumentRepository(session)
            doc = repo.get_by_id(doc_id)
            if doc is None:
                raise DocumentNotFoundError(doc_id)
            return DocumentInfo.from_orm_model(doc)

    def get_document_by_path(
        self,
        dataset_id: int,
        path: str,
    ) -> DocumentInfo:
        """Get a document by dataset and path.

        Args:
            dataset_id: The parent dataset's ID.
            path: The document's path within the dataset.

        Returns:
            The document info.

        Raises:
            DocumentNotFoundError: If the document is not found.
        """
        with self._get_session() as session:
            repo = DocumentRepository(session)
            doc = repo.get_by_path(dataset_id, path)
            if doc is None:
                raise DocumentNotFoundError(path, dataset_id)
            return DocumentInfo.from_orm_model(doc)

    def update_document(
        self,
        doc_id: int,
        data: DocumentUpdate,
    ) -> DocumentInfo:
        """Update a document.

        Args:
            doc_id: The document's ID.
            data: Document update data.

        Returns:
            The updated document info.

        Raises:
            DocumentNotFoundError: If the document is not found.
        """
        with self._get_session() as session:
            repo = DocumentRepository(session)
            doc = repo.get_by_id(doc_id)
            if doc is None:
                raise DocumentNotFoundError(doc_id)

            metadata_json = None
            if data.metadata is not None:
                metadata_json = json.dumps(data.metadata)

            repo.update(
                doc,
                content_hash=data.content_hash,
                body=data.body,
                etag=data.etag,
                last_modified=data.last_modified,
                metadata_json=metadata_json,
                active=data.active,
            )

            logger.debug(f"Updated document: {doc.path}")
            return DocumentInfo.from_orm_model(doc)

    def list_documents(
        self,
        dataset_id: int,
        *,
        active_only: bool = False,
    ) -> list[DocumentInfo]:
        """List documents in a dataset.

        Args:
            dataset_id: The parent dataset's ID.
            active_only: If True, only return active documents.

        Returns:
            List of document info objects.
        """
        with self._get_session() as session:
            repo = DocumentRepository(session)
            docs = repo.list_by_dataset(dataset_id, active_only=active_only)
            return [DocumentInfo.from_orm_model(doc) for doc in docs]

    def list_document_paths(
        self,
        dataset_id: int,
        *,
        active_only: bool = False,
    ) -> set[str]:
        """List document paths in a dataset.

        Args:
            dataset_id: The parent dataset's ID.
            active_only: If True, only return active document paths.

        Returns:
            Set of document paths.
        """
        with self._get_session() as session:
            repo = DocumentRepository(session)
            return repo.list_paths_by_dataset(dataset_id, active_only=active_only)

    def soft_delete_document(self, doc_id: int) -> DocumentInfo:
        """Soft-delete a document by setting active=False.

        Args:
            doc_id: The document's ID.

        Returns:
            The updated document info.

        Raises:
            DocumentNotFoundError: If the document is not found.
        """
        with self._get_session() as session:
            repo = DocumentRepository(session)
            doc = repo.get_by_id(doc_id)
            if doc is None:
                raise DocumentNotFoundError(doc_id)

            repo.soft_delete(doc)
            logger.debug(f"Soft-deleted document: {doc.path}")
            return DocumentInfo.from_orm_model(doc)

    def soft_delete_documents_by_paths(
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
        with self._get_session() as session:
            repo = DocumentRepository(session)
            count = repo.soft_delete_by_paths(dataset_id, paths)
            if count > 0:
                logger.info(f"Soft-deleted {count} documents in dataset {dataset_id}")
            return count

    def delete_document(self, doc_id: int) -> None:
        """Hard-delete a document.

        Args:
            doc_id: The document's ID.

        Raises:
            DocumentNotFoundError: If the document is not found.
        """
        with self._get_session() as session:
            repo = DocumentRepository(session)
            doc = repo.get_by_id(doc_id)
            if doc is None:
                raise DocumentNotFoundError(doc_id)

            path = doc.path
            repo.delete(doc)
            logger.debug(f"Deleted document: {path}")
