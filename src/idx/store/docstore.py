"""idx.store.llama - LlamaIndex document store backed by SQLite.

Provides a LlamaIndex-compatible BaseDocumentStore implementation that uses
the existing idx SQLite storage layer for persistence.

Example usage:
    from idx.store.llama import SQLDocStore
    from idx.store import get_session_factory

    # Create docstore scoped to a dataset
    docstore = SQLDocStore(
        dataset_id=1,
        session_factory=get_session_factory()
    )

    # Use with LlamaIndex
    from llama_index.core import StorageContext, VectorStoreIndex
    storage_context = StorageContext.from_defaults(docstore=docstore)
    index = VectorStoreIndex(nodes, storage_context=storage_context)
"""
#UNUSED
from __future__ import annotations

import json
from typing import Callable, Dict, Optional, Sequence

import fsspec
from sqlalchemy.orm import Session

from llama_index.core.schema import BaseNode
from llama_index.core.storage.docstore.types import (
    BaseDocumentStore,
    RefDocInfo,
    DEFAULT_PERSIST_PATH,
)
from llama_index.core.storage.docstore.utils import doc_to_json, json_to_doc
from llama_index.core.storage.kvstore.types import DEFAULT_BATCH_SIZE

from idx.store.models import Document as SQLDocument
from idx.store.repositories import DocumentRepository

__all__ = ["SQLiteDocumentStore"]


class SQLiteDocumentStore(BaseDocumentStore):
    """LlamaIndex document store backed by idx's SQLite storage.

    This class implements LlamaIndex's BaseDocumentStore interface using the
    existing idx SQLite database for persistence. Documents are scoped to a
    specific dataset, allowing multiple document stores to coexist in the
    same database.

    The store maps LlamaIndex nodes to idx Document records:
    - node.node_id -> Document.path (used as unique identifier within dataset)
    - node content (JSON) -> Document.body
    - node.hash -> Document.content_hash
    - node.metadata -> Document.metadata_json

    Supports upsert semantics: documents are updated if they exist, created if not.

    Attributes:
        dataset_id: The dataset ID to scope all operations to.
        session_factory: Callable that returns a SQLAlchemy Session.

    Example:
        >>> from idx.store.llama import SQLDocStore
        >>> docstore = SQLDocStore(dataset_id=1, session_factory=get_session_factory())
        >>> docstore.add_documents(nodes)
        >>> node = docstore.get_document("node-id-123")
    """

    def __init__(
        self,
        dataset_id: int,
        #database_path: Optional[Path],
        #session_factory: Callable[[], Session],
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize the SQL document store.

        Args:
            dataset_id: The dataset ID to scope all operations to.
            session_factory: Callable that returns a SQLAlchemy Session.
                The caller is responsible for session lifecycle management.
            batch_size: Default batch size for bulk operations.
        """
        self._dataset_id = dataset_id
        #self._database_path = database_path
        #self._session_factory = session_factory
        self._batch_size = batch_size
        # In-memory cache for document hashes (doc_id -> hash)
        self._doc_hashes: Dict[str, str] = {}
        # In-memory cache for ref_doc_info (ref_doc_id -> RefDocInfo)
        self._ref_doc_infos: Dict[str, RefDocInfo] = {}

    @property
    def dataset_id(self) -> int:
        """Get the dataset ID this store is scoped to."""
        return self._dataset_id

    def _get_session(self) -> Session:
        """Get a new database session."""
        from idx.store import get_session_factory
        session_factory = get_session_factory()
        return session_factory()
        #return self._session_factory()

    def _node_to_sql_doc(
        self,
        node: BaseNode,
        *,
        existing_doc: Optional[SQLDocument] = None,
    ) -> SQLDocument:
        """Convert a LlamaIndex node to a SQL Document model.

        Args:
            node: The LlamaIndex node to convert.
            existing_doc: If provided, update this document instead of creating new.

        Returns:
            A SQLDocument instance with the node's data.
        """
        # Serialize the node to JSON for storage
        node_json = doc_to_json(node)
        body = json.dumps(node_json)

        # Extract metadata as JSON string
        metadata_json = json.dumps(node.metadata) if node.metadata else None

        if existing_doc is not None:
            existing_doc.body = body
            existing_doc.content_hash = node.hash
            existing_doc.metadata_json = metadata_json
            existing_doc.active = True
            return existing_doc

        return SQLDocument(
            dataset_id=self._dataset_id,
            path=node.node_id,  # Use node_id as the unique path
            body=body,
            content_hash=node.hash,
            metadata_json=metadata_json,
            active=True,
        )

    def _sql_doc_to_node(self, doc: SQLDocument) -> BaseNode:
        """Convert a SQL Document model to a LlamaIndex node.

        Args:
            doc: The SQLDocument to convert.

        Returns:
            The deserialized LlamaIndex BaseNode.
        """
        node_json = json.loads(doc.body)
        return json_to_doc(node_json)

    # ===== Save/load =====

    def persist(
        self,
        persist_path: str = DEFAULT_PERSIST_PATH,
        fs: Optional[fsspec.AbstractFileSystem] = None,
    ) -> None:
        """Persist the docstore to a file.

        Note: This is a no-op for SQLDocStore since data is already persisted
        to SQLite. Provided for interface compatibility.

        Args:
            persist_path: Path to persist to (ignored).
            fs: Filesystem to use (ignored).
        """
        # Data is already persisted to SQLite, no action needed
        pass

    # ===== Main interface =====

    @property
    def docs(self) -> Dict[str, BaseNode]:
        """Get all documents as a dictionary.

        Returns:
            Dictionary mapping node_id to BaseNode for all active documents.
        """
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            sql_docs = repo.list_by_dataset(self._dataset_id, active_only=True)
            return {doc.path: self._sql_doc_to_node(doc) for doc in sql_docs}
        finally:
            session.close()

    def add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        store_text: bool = True,
    ) -> None:
        """Add documents to the store.

        Implements upsert semantics: updates existing documents, creates new ones.

        Args:
            docs: Sequence of LlamaIndex nodes to add.
            allow_update: If True, allow updating existing documents.
                If False, raise ValueError if document exists.
            batch_size: Batch size for bulk operations (currently unused).
            store_text: If True, store the document text. If False, only store
                metadata and hash (currently always stores text).
        """
        if not docs:
            return

        session = self._get_session()
        try:
            repo = DocumentRepository(session)

            for node in docs:
                existing = repo.get_by_path(self._dataset_id, node.node_id)

                if existing is not None:
                    if not allow_update:
                        raise ValueError(
                            f"node_id {node.node_id} already exists. "
                            "Set allow_update to True to overwrite."
                        )
                    # Update existing document
                    self._node_to_sql_doc(node, existing_doc=existing)
                else:
                    # Create new document
                    sql_doc = self._node_to_sql_doc(node)
                    session.add(sql_doc)

                # Update hash cache
                self._doc_hashes[node.node_id] = node.hash

                # Update ref_doc tracking
                if node.ref_doc_id:
                    ref_doc_id = node.ref_doc_id
                    if ref_doc_id not in self._ref_doc_infos:
                        self._ref_doc_infos[ref_doc_id] = RefDocInfo(
                            node_ids=[], metadata={}
                        )
                    ref_info = self._ref_doc_infos[ref_doc_id]
                    if node.node_id not in ref_info.node_ids:
                        ref_info.node_ids.append(node.node_id)
                    if not ref_info.metadata and node.metadata:
                        ref_info.metadata = node.metadata

            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def async_add_documents(
        self,
        docs: Sequence[BaseNode],
        allow_update: bool = True,
        batch_size: int = DEFAULT_BATCH_SIZE,
        store_text: bool = True,
    ) -> None:
        """Async version of add_documents.

        Note: Currently delegates to synchronous implementation.
        """
        self.add_documents(docs, allow_update, batch_size, store_text)

    def get_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseNode]:
        """Get a document by ID.

        Args:
            doc_id: The node ID to retrieve.
            raise_error: If True, raise ValueError if not found.

        Returns:
            The BaseNode if found, None otherwise (when raise_error=False).

        Raises:
            ValueError: If document not found and raise_error=True.
        """
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            sql_doc = repo.get_by_path(self._dataset_id, doc_id)

            if sql_doc is None or not sql_doc.active:
                if raise_error:
                    raise ValueError(f"doc_id {doc_id} not found.")
                return None

            return self._sql_doc_to_node(sql_doc)
        finally:
            session.close()

    async def aget_document(
        self, doc_id: str, raise_error: bool = True
    ) -> Optional[BaseNode]:
        """Async version of get_document.

        Note: Currently delegates to synchronous implementation.
        """
        return self.get_document(doc_id, raise_error)

    def delete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Delete a document from the store.

        Uses soft-delete by setting active=False.

        Args:
            doc_id: The node ID to delete.
            raise_error: If True, raise ValueError if not found.

        Raises:
            ValueError: If document not found and raise_error=True.
        """
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            sql_doc = repo.get_by_path(self._dataset_id, doc_id)

            if sql_doc is None:
                if raise_error:
                    raise ValueError(f"doc_id {doc_id} not found.")
                return

            repo.soft_delete(sql_doc)
            session.commit()

            # Clean up caches
            self._doc_hashes.pop(doc_id, None)
            # Remove from ref_doc tracking
            for ref_info in self._ref_doc_infos.values():
                if doc_id in ref_info.node_ids:
                    ref_info.node_ids.remove(doc_id)

        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def adelete_document(self, doc_id: str, raise_error: bool = True) -> None:
        """Async version of delete_document.

        Note: Currently delegates to synchronous implementation.
        """
        self.delete_document(doc_id, raise_error)

    def document_exists(self, doc_id: str) -> bool:
        """Check if a document exists.

        Args:
            doc_id: The node ID to check.

        Returns:
            True if the document exists and is active.
        """
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            sql_doc = repo.get_by_path(self._dataset_id, doc_id)
            return sql_doc is not None and sql_doc.active
        finally:
            session.close()

    async def adocument_exists(self, doc_id: str) -> bool:
        """Async version of document_exists.

        Note: Currently delegates to synchronous implementation.
        """
        return self.document_exists(doc_id)

    # ===== Hash =====

    def set_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Set the hash for a document.

        Args:
            doc_id: The node ID.
            doc_hash: The hash value to store.
        """
        self._doc_hashes[doc_id] = doc_hash

        # Also update in database if document exists
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            sql_doc = repo.get_by_path(self._dataset_id, doc_id)
            if sql_doc is not None:
                sql_doc.content_hash = doc_hash
                session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def aset_document_hash(self, doc_id: str, doc_hash: str) -> None:
        """Async version of set_document_hash.

        Note: Currently delegates to synchronous implementation.
        """
        self.set_document_hash(doc_id, doc_hash)

    def set_document_hashes(self, doc_hashes: Dict[str, str]) -> None:
        """Set hashes for multiple documents.

        Args:
            doc_hashes: Dictionary mapping doc_id to hash.
        """
        self._doc_hashes.update(doc_hashes)

        # Also update in database
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            for doc_id, doc_hash in doc_hashes.items():
                sql_doc = repo.get_by_path(self._dataset_id, doc_id)
                if sql_doc is not None:
                    sql_doc.content_hash = doc_hash
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    async def aset_document_hashes(self, doc_hashes: Dict[str, str]) -> None:
        """Async version of set_document_hashes.

        Note: Currently delegates to synchronous implementation.
        """
        self.set_document_hashes(doc_hashes)

    def get_document_hash(self, doc_id: str) -> Optional[str]:
        """Get the hash for a document.

        Args:
            doc_id: The node ID.

        Returns:
            The hash value if found, None otherwise.
        """
        # Check cache first
        if doc_id in self._doc_hashes:
            return self._doc_hashes[doc_id]

        # Fall back to database
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            sql_doc = repo.get_by_path(self._dataset_id, doc_id)
            if sql_doc is not None:
                self._doc_hashes[doc_id] = sql_doc.content_hash
                return sql_doc.content_hash
            return None
        finally:
            session.close()

    async def aget_document_hash(self, doc_id: str) -> Optional[str]:
        """Async version of get_document_hash.

        Note: Currently delegates to synchronous implementation.
        """
        return self.get_document_hash(doc_id)

    def get_all_document_hashes(self) -> Dict[str, str]:
        """Get all document hashes.

        Returns:
            Dictionary mapping doc_hash to doc_id for all active documents.

        Note: The return format matches LlamaIndex convention (hash -> doc_id).
        """
        session = self._get_session()
        try:
            repo = DocumentRepository(session)
            sql_docs = repo.list_by_dataset(self._dataset_id, active_only=True)

            # Update cache and build result
            result: Dict[str, str] = {}
            for doc in sql_docs:
                self._doc_hashes[doc.path] = doc.content_hash
                result[doc.content_hash] = doc.path

            return result
        finally:
            session.close()

    async def aget_all_document_hashes(self) -> Dict[str, str]:
        """Async version of get_all_document_hashes.

        Note: Currently delegates to synchronous implementation.
        """
        return self.get_all_document_hashes()

    # ==== Ref Docs =====

    def get_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """Get all reference document info.

        Returns:
            Dictionary mapping ref_doc_id to RefDocInfo, or None if empty.
        """
        if not self._ref_doc_infos:
            # Rebuild from stored documents
            session = self._get_session()
            try:
                repo = DocumentRepository(session)
                sql_docs = repo.list_by_dataset(self._dataset_id, active_only=True)

                for sql_doc in sql_docs:
                    node = self._sql_doc_to_node(sql_doc)
                    if node.ref_doc_id:
                        ref_doc_id = node.ref_doc_id
                        if ref_doc_id not in self._ref_doc_infos:
                            self._ref_doc_infos[ref_doc_id] = RefDocInfo(
                                node_ids=[], metadata={}
                            )
                        ref_info = self._ref_doc_infos[ref_doc_id]
                        if node.node_id not in ref_info.node_ids:
                            ref_info.node_ids.append(node.node_id)
                        if not ref_info.metadata and node.metadata:
                            ref_info.metadata = node.metadata
            finally:
                session.close()

        return self._ref_doc_infos if self._ref_doc_infos else None

    async def aget_all_ref_doc_info(self) -> Optional[Dict[str, RefDocInfo]]:
        """Async version of get_all_ref_doc_info.

        Note: Currently delegates to synchronous implementation.
        """
        return self.get_all_ref_doc_info()

    def get_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Get reference document info for a specific ref_doc_id.

        Args:
            ref_doc_id: The reference document ID.

        Returns:
            RefDocInfo if found, None otherwise.
        """
        # Ensure cache is populated
        self.get_all_ref_doc_info()
        return self._ref_doc_infos.get(ref_doc_id)

    async def aget_ref_doc_info(self, ref_doc_id: str) -> Optional[RefDocInfo]:
        """Async version of get_ref_doc_info.

        Note: Currently delegates to synchronous implementation.
        """
        return self.get_ref_doc_info(ref_doc_id)

    def delete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """Delete a reference document and all its associated nodes.

        Args:
            ref_doc_id: The reference document ID.
            raise_error: If True, raise ValueError if not found.

        Raises:
            ValueError: If ref_doc not found and raise_error=True.
        """
        ref_doc_info = self.get_ref_doc_info(ref_doc_id)

        if ref_doc_info is None:
            if raise_error:
                raise ValueError(f"ref_doc_id {ref_doc_id} not found.")
            return

        # Delete all associated nodes
        node_ids = ref_doc_info.node_ids.copy()
        for node_id in node_ids:
            self.delete_document(node_id, raise_error=False)

        # Remove from cache
        self._ref_doc_infos.pop(ref_doc_id, None)

    async def adelete_ref_doc(self, ref_doc_id: str, raise_error: bool = True) -> None:
        """Async version of delete_ref_doc.

        Note: Currently delegates to synchronous implementation.
        """
        self.delete_ref_doc(ref_doc_id, raise_error)
