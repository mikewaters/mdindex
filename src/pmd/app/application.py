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
    from ..services.search import SearchService
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
        llm_provider: "LLMProvider | None",
        loading: "LoadingService",
        indexing: "IndexingService",
        search: "SearchService",
        status: "StatusService",
        config: "Config",
        source_collection_repo=None,
        document_repo=None,
        embedding_repo=None,
    ):
        """Initialize Application with wired services.

        This constructor is for internal use. Use create_application() instead.

        Args:
            db: Database instance.
            llm_provider: LLM provider instance (may be None).
            loading: LoadingService instance.
            indexing: IndexingService instance.
            search: SearchService instance.
            status: StatusService instance.
            config: Application configuration.
            source_collection_repo: Repository for source collections.
            document_repo: Repository for documents.
            embedding_repo: Repository for embeddings.
        """
        self._db = db
        self._llm_provider = llm_provider
        self._config = config
        self._source_collection_repo = source_collection_repo
        self._document_repo = document_repo
        self._embedding_repo = embedding_repo

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

    async def is_llm_available(self) -> bool:
        """Check if LLM provider is available.

        Returns:
            True if LLM provider can be reached.
        """
        if self._llm_provider:
            return await self._llm_provider.is_available()
        return False

    async def close(self) -> None:
        """Clean shutdown of all resources."""
        if self._llm_provider:
            await self._llm_provider.close()
        self._db.close()

    async def __aenter__(self) -> "Application":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and clean up resources."""
        await self.close()
