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
    from ..services.loading import LoadingService
    from ..services.search import SearchService
    from ..services.status import StatusService
    from ..store.database import Database
    from ..llm.base import LLMProvider

from .types import (
    # Database protocol
    DatabaseProtocol,
    # Repository protocols
    SourceCollectionRepositoryProtocol,
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
    # Service protocols
    LoadingServiceProtocol,
    # Config protocols
    ConfigProtocol,
    SearchConfigProtocol,
    # Type aliases
    SourceCollectionRepo,
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
    "SourceCollectionRepositoryProtocol",
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
    # Services
    "LoadingServiceProtocol",
    # Config
    "ConfigProtocol",
    "SearchConfigProtocol",
    # Aliases
    "SourceCollectionRepo",
    "DocumentRepo",
    "FTSRepo",
    "EmbeddingRepo",
]


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
    from pmd.store.database import Database
    from pmd.store.collections import SourceCollectionRepository
    from pmd.store.documents import DocumentRepository
    from pmd.store.content import ContentRepository
    from pmd.store.search import FTS5SearchRepository
    from pmd.store.embeddings import EmbeddingRepository
    from pmd.store.source_metadata import SourceMetadataRepository
    from pmd.services.indexing import IndexingService
    from pmd.services.loading import LoadingService
    from pmd.services.search import SearchService
    from pmd.services.status import StatusService
    from pmd.sources import get_default_registry
    from pmd.llm import create_llm_provider, EmbeddingGenerator, QueryExpander, DocumentReranker
    from pmd.metadata import (
        LexicalTagMatcher,
        Ontology,
        TagRetriever,
        DocumentMetadataRepository,
    )
    from pmd.services.caching import DocumentCacher

    # Create and connect database
    db = Database(config.db_path)
    db.connect()

    # Create repositories
    source_collection_repo = SourceCollectionRepository(db)
    document_repo = DocumentRepository(db)
    content_repo = ContentRepository(db)
    fts_repo = FTS5SearchRepository(db)
    embedding_repo = EmbeddingRepository(db)

    # Create LLM provider (may be None if provider unavailable)
    try:
        llm_provider = create_llm_provider(config)
    except (ValueError, RuntimeError):
        llm_provider = None

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
        return Ontology()  #type: ignore

    def get_tag_retriever():
        return TagRetriever(db) # type: ignore

    def get_metadata_repo():
        return DocumentMetadataRepository(db)

    # Check LLM availability
    async def is_llm_available():
        if llm_provider:
            return await llm_provider.is_available()
        return False

    # Create source metadata repository
    source_metadata_repo = SourceMetadataRepository(db)

    # Create source registry
    source_registry = get_default_registry()

    # Create document cacher (if enabled in config)
    cacher = DocumentCacher(config.cache) if config.cache.enabled else None

    # Create loading service
    loading = LoadingService(
        db=db,
        source_collection_repo=source_collection_repo,
        document_repo=document_repo,
        source_metadata_repo=source_metadata_repo,
        source_registry=source_registry,
    )

    # Create services with explicit dependencies
    indexing = IndexingService(
        db=db,
        source_collection_repo=source_collection_repo,
        document_repo=document_repo,
        fts_repo=fts_repo,
        loader=loading,
        content_repo=content_repo,
        embedding_repo=embedding_repo,
        embedding_generator_factory=get_embedding_generator, # type: ignore
        llm_available_check=is_llm_available,
        source_registry=source_registry,
        cacher=cacher,
    )

    search = SearchService(
        db=db,
        fts_repo=fts_repo,
        source_collection_repo=source_collection_repo,
        embedding_repo=embedding_repo,
        embedding_generator_factory=get_embedding_generator,  # type: ignore
        query_expander_factory=get_query_expander,  # type: ignore
        reranker_factory=get_reranker,  # type: ignore
        tag_matcher_factory=get_tag_matcher, # type: ignore 
        ontology_factory=get_ontology, # type: ignore
        tag_retriever_factory=get_tag_retriever, # type: ignore
        metadata_repo_factory=get_metadata_repo, # type: ignore
        fts_weight=config.search.fts_weight,
        vec_weight=config.search.vec_weight,
        rrf_k=config.search.rrf_k,
        rerank_candidates=config.search.rerank_candidates,
    )

    status = StatusService(
        document_repo=document_repo,
        embedding_repo=embedding_repo,
        fts_repo=fts_repo,
        source_collection_repo=source_collection_repo,
        db_path=config.db_path,
        llm_provider=config.llm_provider,
        llm_available_check=is_llm_available,
        vec_available=db.vec_available,
    )

    return Application(
        db=db,
        llm_provider=llm_provider,
        loading=loading,
        indexing=indexing,
        search=search,
        status=status,
        config=config,
        source_collection_repo=source_collection_repo,
        document_repo=document_repo,
        embedding_repo=embedding_repo,
    )
