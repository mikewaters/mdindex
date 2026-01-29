"""idx.store - Persistence layer.

SQLAlchemy-based storage with SQLite backend.
Exposes DatasetService for validated Pydantic model persistence.
All persistence including FTS and vector index wiring resides here.
"""

from idx.store.database import (
    Base,
    create_engine_for_path,
    get_engine,
    get_session,
    get_session_factory,
)
from idx.store.models import Dataset, Document
from idx.store.repositories import DatasetRepository, DocumentRepository
from idx.store.schemas import (
    DatasetCreate,
    DatasetInfo,
    DocumentCreate,
    DocumentInfo,
    DocumentUpdate,
)
from idx.store.cleanup import (
    IndexCleanup,
    cleanup_fts_for_document,
    cleanup_fts_for_inactive_documents,
    cleanup_stale_documents,
)
from idx.store.fts import (
    FTSManager,
    FTSResult,
    create_fts_table,
    drop_fts_table,
)
from idx.store.fts_chunk import (
    FTSChunkManager,
    FTSChunkResult,
    create_chunks_fts_table,
    drop_chunks_fts_table,
)
from idx.store.docstore import SQLiteDocumentStore
from idx.store.vector import VectorStoreManager
from idx.store.session_context import (
    SessionNotSetError,
    clear_session,
    current_session,
    use_session,
)
from idx.store.dataset import (
    DatasetExistsError,
    DatasetNotFoundError,
    DatasetService,
    DocumentNotFoundError,
    normalize_dataset_name,
)

__all__ = [
    # Database
    "Base",
    "create_engine_for_path",
    "get_engine",
    "get_session",
    "get_session_factory",
    # Models
    "Dataset",
    "Document",
    # Repositories
    "DatasetRepository",
    "DocumentRepository",
    # Schemas
    "DatasetCreate",
    "DatasetInfo",
    "DocumentCreate",
    "DocumentInfo",
    "DocumentUpdate",
    # Service
    "DatasetService",
    "DatasetExistsError",
    "DatasetNotFoundError",
    "DocumentNotFoundError",
    "normalize_dataset_name",
    # FTS (document-level)
    "FTSManager",
    "FTSResult",
    "create_fts_table",
    "drop_fts_table",
    # FTS (chunk-level)
    "FTSChunkManager",
    "FTSChunkResult",
    "create_chunks_fts_table",
    "drop_chunks_fts_table",
    # Cleanup
    "IndexCleanup",
    "cleanup_fts_for_document",
    "cleanup_fts_for_inactive_documents",
    "cleanup_stale_documents",
    # LlamaIndex integration
    "SQLiteDocumentStore",
    "VectorStoreManager",
    # Session context
    "SessionNotSetError",
    "clear_session",
    "current_session",
    "use_session",
]
