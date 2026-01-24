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
from idx.store.docstore import SQLDocStore
from idx.store.session_context import (
    SessionNotSetError,
    clear_session,
    current_session,
    use_session,
)
from idx.store.service import (
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
    # FTS
    "FTSManager",
    "FTSResult",
    "create_fts_table",
    "drop_fts_table",
    # Cleanup
    "IndexCleanup",
    "cleanup_fts_for_document",
    "cleanup_fts_for_inactive_documents",
    "cleanup_stale_documents",
    # LlamaIndex integration
    "SQLDocStore",
    # Session context
    "SessionNotSetError",
    "clear_session",
    "current_session",
    "use_session",
]
