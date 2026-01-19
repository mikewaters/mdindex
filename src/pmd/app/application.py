"""Application container class.

This module provides the Application class for managing wired services
and their lifecycle.

Use create_application() from pmd.app to create a properly configured instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import Config
    from ..services.indexing import IndexingService
    from ..services.loading import LoadingService
    from search.service import SearchService
    from ..services.status import StatusService
    from ..store.database import Database
    from ..llm.base import LLMProvider


class Application:
    """Application container with wired services and lifecycle management.

    This class holds all wired services and manages their lifecycle.
    Use create_application() to create a properly configured instance.

    Attributes:
        loading: LoadingService for document loading operations.
        indexing: IndexingService for document indexing operations.
        search: SearchService for search operations.
        status: StatusService for status reporting.

    Example:
        async with create_application(config) as app:
            result = await app.indexing.index_collection("docs")
            results = await app.search.hybrid_search("query")
    """

    def __init__(
        self,
        db: "Database",
        llm_provider: "LLMProvider",
        loading: "LoadingService",
        indexing: "IndexingService",
        search: "SearchService",
        status: "StatusService",
        config: "Config",
    ):
        """Initialize Application with wired services.

        This constructor is for internal use. Use create_application() instead.

        Args:
            db: Database instance.
            llm_provider: LLM provider instance.
            loading: LoadingService instance.
            indexing: IndexingService instance.
            search: SearchService instance.
            status: StatusService instance.
            config: Application configuration.
        """
        self._db = db
        self._llm_provider = llm_provider
        self._config = config

        # Create repositories internally (for test compatibility)
        # These are lazily imported to avoid circular dependencies
        from ..store.repositories.collections import SourceCollectionRepository
        from ..store.repositories.documents import DocumentRepository
        from ..store.repositories.embeddings import EmbeddingRepository

        self._source_collection_repo = SourceCollectionRepository(db)
        self._document_repo = DocumentRepository(db)
        self._embedding_repo = EmbeddingRepository(db)

        # Public service accessors
        self.loading = loading
        self.indexing = indexing
        self.search = search
        self.status = status

    @property
    def source_collection_repo(self):
        """Get source collection repository."""
        return self._source_collection_repo

    @property
    def document_repo(self):
        """Get document repository."""
        return self._document_repo

    @property
    def embedding_repo(self):
        """Get embedding repository."""
        return self._embedding_repo

    @property
    def db(self) -> "Database":
        """Get database instance."""
        return self._db

    @property
    def config(self) -> "Config":
        """Get application configuration."""
        return self._config

    @property
    def vec_available(self) -> bool:
        """Check if vector storage is available."""
        return self._db.vec_available

    async def close(self) -> None:
        """Clean shutdown of all resources."""
        await self._llm_provider.close()
        self._db.close()

    async def __aenter__(self) -> "Application":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and clean up resources."""
        await self.close()
