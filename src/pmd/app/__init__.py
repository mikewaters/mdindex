"""Application composition root and dependency injection.

This module provides:
- Protocol definitions for injectable dependencies (types.py)
- The Application class for lifecycle management
- Factory functions for creating configured applications

Example:
    from pmd.app import create_application
    from pmd.core.config import Config

    async with create_application(Config()) as app:
        results = await app.search.hybrid_search("query")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.config import Config
    from ..services.indexing import IndexingService
    from ..services.search import SearchService
    from ..services.status import StatusService
    from ..store.database import Database
    from ..llm.base import LLMProvider

from .types import (
    # Database protocol
    DatabaseProtocol,
    # Repository protocols
    CollectionRepositoryProtocol,
    DocumentRepositoryProtocol,
    FTSRepositoryProtocol,
    EmbeddingRepositoryProtocol,
    # LLM protocols
    LLMProviderProtocol,
    EmbeddingGeneratorProtocol,
    QueryExpanderProtocol,
    DocumentRerankerProtocol,
    # Metadata protocols
    TagMatcherProtocol,
    OntologyProtocol,
    TagRetrieverProtocol,
    DocumentMetadataRepositoryProtocol,
    # Config protocols
    ConfigProtocol,
    SearchConfigProtocol,
    # Type aliases
    CollectionRepo,
    DocumentRepo,
    FTSRepo,
    EmbeddingRepo,
)

__all__ = [
    # Application
    "Application",
    "create_application",
    # Database
    "DatabaseProtocol",
    # Repositories
    "CollectionRepositoryProtocol",
    "DocumentRepositoryProtocol",
    "FTSRepositoryProtocol",
    "EmbeddingRepositoryProtocol",
    # LLM
    "LLMProviderProtocol",
    "EmbeddingGeneratorProtocol",
    "QueryExpanderProtocol",
    "DocumentRerankerProtocol",
    # Metadata
    "TagMatcherProtocol",
    "OntologyProtocol",
    "TagRetrieverProtocol",
    "DocumentMetadataRepositoryProtocol",
    # Config
    "ConfigProtocol",
    "SearchConfigProtocol",
    # Aliases
    "CollectionRepo",
    "DocumentRepo",
    "FTSRepo",
    "EmbeddingRepo",
]


class Application:
    """Application container with wired services and lifecycle management.

    This class holds all wired services and manages their lifecycle.
    Use create_application() to create a properly configured instance.

    Attributes:
        indexing: IndexingService for document indexing operations.
        search: SearchService for search operations.
        status: StatusService for status reporting.

    Example:
        async with create_application(config) as app:
            result = await app.indexing.index_collection("docs", source)
            results = await app.search.hybrid_search("query")
    """

    def __init__(
        self,
        db: "Database",
        llm_provider: "LLMProvider | None",
        indexing: "IndexingService",
        search: "SearchService",
        status: "StatusService",
        config: "Config",
    ):
        """Initialize Application with wired services.

        This constructor is for internal use. Use create_application() instead.

        Args:
            db: Database instance.
            llm_provider: LLM provider instance (may be None).
            indexing: IndexingService instance.
            search: SearchService instance.
            status: StatusService instance.
            config: Application configuration.
        """
        self._db = db
        self._llm_provider = llm_provider
        self._config = config

        # Public service accessors
        self.indexing = indexing
        self.search = search
        self.status = status

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
        if self._llm_provider:
            await self._llm_provider.close()
        self._db.close()

    async def __aenter__(self) -> "Application":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and clean up resources."""
        await self.close()


async def create_application(config: "Config") -> Application:
    """Create and wire a fully configured Application.

    This is the main composition root that wires all dependencies together.

    Args:
        config: Application configuration.

    Returns:
        Configured Application instance ready for use.

    Example:
        from pmd.core.config import Config
        from pmd.app import create_application

        async with create_application(Config()) as app:
            await app.indexing.index_collection("docs", source)
            results = await app.search.hybrid_search("query")
    """
    # Lazy imports to avoid circular dependencies
    from ..core.config import Config as ConfigClass
    from ..store.database import Database
    from ..store.collections import CollectionRepository
    from ..store.documents import DocumentRepository
    from ..store.search import FTS5SearchRepository
    from ..store.embeddings import EmbeddingRepository
    from ..services.indexing import IndexingService
    from ..services.search import SearchService
    from ..services.status import StatusService
    from ..llm import get_llm_provider
    from ..llm.components import EmbeddingGenerator, QueryExpander, DocumentReranker
    from ..metadata import (
        LexicalTagMatcher,
        Ontology,
        TagRetriever,
        DocumentMetadataRepository,
    )

    # Create and connect database
    db = Database(config.db_path)
    db.connect()

    # Create repositories
    collection_repo = CollectionRepository(db)
    document_repo = DocumentRepository(db)
    fts_repo = FTS5SearchRepository(db)
    embedding_repo = EmbeddingRepository(db)

    # Create LLM provider (may be None if provider unavailable)
    llm_provider = await get_llm_provider(config)

    # Create async factories for LLM components
    async def get_embedding_generator():
        if llm_provider:
            return EmbeddingGenerator(llm_provider, embedding_repo, config)
        return None

    async def get_query_expander():
        if llm_provider:
            return QueryExpander(llm_provider)
        return None

    async def get_reranker():
        if llm_provider:
            return DocumentReranker(llm_provider)
        return None

    # Create sync factories for metadata components
    def get_tag_matcher():
        return LexicalTagMatcher()

    def get_ontology():
        return Ontology()

    def get_tag_retriever():
        return TagRetriever(db)

    def get_metadata_repo():
        return DocumentMetadataRepository(db)

    # Check LLM availability
    async def is_llm_available():
        if llm_provider:
            return await llm_provider.is_available()
        return False

    # Create services with explicit dependencies
    indexing = IndexingService(
        db=db,
        collection_repo=collection_repo,
        document_repo=document_repo,
        fts_repo=fts_repo,
        embedding_repo=embedding_repo,
        embedding_generator_factory=get_embedding_generator,
        llm_available_check=is_llm_available,
    )

    search = SearchService(
        db=db,
        fts_repo=fts_repo,
        collection_repo=collection_repo,
        embedding_repo=embedding_repo,
        embedding_generator_factory=get_embedding_generator,
        query_expander_factory=get_query_expander,
        reranker_factory=get_reranker,
        tag_matcher_factory=get_tag_matcher,
        ontology_factory=get_ontology,
        tag_retriever_factory=get_tag_retriever,
        metadata_repo_factory=get_metadata_repo,
        fts_weight=config.search.fts_weight,
        vec_weight=config.search.vec_weight,
        rrf_k=config.search.rrf_k,
        rerank_candidates=config.search.rerank_candidates,
    )

    status = StatusService(
        db=db,
        collection_repo=collection_repo,
        db_path=config.db_path,
        llm_provider=config.llm_provider,
        llm_available_check=is_llm_available,
    )

    return Application(
        db=db,
        llm_provider=llm_provider,
        indexing=indexing,
        search=search,
        status=status,
        config=config,
    )
