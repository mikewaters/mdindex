"""Service container for dependency injection and lifecycle management."""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from ..core.config import Config
from ..llm import create_llm_provider
from ..llm.embeddings import EmbeddingGenerator
from ..llm.query_expansion import QueryExpander
from ..llm.reranker import DocumentReranker
from ..store.collections import CollectionRepository
from ..store.database import Database
from ..store.documents import DocumentRepository
from ..store.embeddings import EmbeddingRepository
from ..store.search import FTS5SearchRepository

if TYPE_CHECKING:
    from ..llm.base import LLMProvider
    from ..metadata.query.inference import LexicalTagMatcher
    from ..metadata.query.retrieval import TagRetriever
    from ..metadata.model.ontology import Ontology
    from ..metadata.store import DocumentMetadataRepository
    from .indexing import IndexingService
    from .search import SearchService
    from .status import StatusService


class ServiceContainer:
    """Manages service lifecycle and shared resources.

    The ServiceContainer is the main entry point for using PMD services.
    It manages the database connection, LLM provider, and provides access
    to all services through a unified interface.

    Services are created lazily on first access to avoid unnecessary
    initialization. The LLM provider is also lazy-loaded since not all
    operations require it.

    Usage as context manager (recommended):

        async with ServiceContainer(config) as services:
            from pmd.sources import get_default_registry

            collection = services.collection_repo.get_by_name("docs")
            registry = get_default_registry()
            source = registry.create_source(collection)
            result = await services.indexing.index_collection("docs", source=source)
            results = await services.search.hybrid_search("query")

    Usage with manual lifecycle:

        services = ServiceContainer(config)
        services.connect()
        try:
            # use services
        finally:
            await services.close()

    Attributes:
        config: Application configuration.
        db: Database instance (connected after connect() or __aenter__).
        indexing: IndexingService instance.
        search: SearchService instance.
        status: StatusService instance.
    """

    def __init__(self, config: Config):
        """Initialize container with configuration.

        Args:
            config: Application configuration.
        """
        self.config = config
        # Pass embedding dimension from MLX config to database for vector schema
        self.db = Database(config.db_path, embedding_dimension=config.mlx.embedding_dimension)
        self._llm_provider: LLMProvider | None = None
        self._connected = False

        # Lazy-initialized services
        self._indexing: IndexingService | None = None
        self._search: SearchService | None = None
        self._status: StatusService | None = None

        # Lazy-initialized repositories (shared across services)
        self._collection_repo: CollectionRepository | None = None
        self._document_repo: DocumentRepository | None = None
        self._embedding_repo: EmbeddingRepository | None = None
        self._fts_repo: FTS5SearchRepository | None = None

        # Lazy-initialized LLM components
        self._embedding_generator: EmbeddingGenerator | None = None
        self._query_expander: QueryExpander | None = None
        self._reranker: DocumentReranker | None = None

        # Lazy-initialized metadata components
        self._tag_matcher: "LexicalTagMatcher | None" = None
        self._ontology: "Ontology | None" = None
        self._tag_retriever: "TagRetriever | None" = None
        self._metadata_repo: "DocumentMetadataRepository | None" = None

    def connect(self) -> None:
        """Connect to database.

        Must be called before accessing services unless using
        the async context manager.
        """
        if not self._connected:
            self.db.connect()
            self._connected = True
            logger.debug("ServiceContainer connected to database")

    async def close(self) -> None:
        """Close all connections and release resources."""
        if self._llm_provider:
            await self._llm_provider.close()
            self._llm_provider = None
            logger.debug("LLM provider closed")

        if self._connected:
            self.db.close()
            self._connected = False
            logger.debug("ServiceContainer closed")

    async def __aenter__(self) -> "ServiceContainer":
        """Async context manager entry."""
        self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # --- Repository accessors (shared across services) ---

    @property
    def collection_repo(self) -> CollectionRepository:
        """Get or create CollectionRepository."""
        if self._collection_repo is None:
            self._collection_repo = CollectionRepository(self.db)
        return self._collection_repo

    @property
    def document_repo(self) -> DocumentRepository:
        """Get or create DocumentRepository."""
        if self._document_repo is None:
            self._document_repo = DocumentRepository(self.db)
        return self._document_repo

    @property
    def embedding_repo(self) -> EmbeddingRepository:
        """Get or create EmbeddingRepository."""
        if self._embedding_repo is None:
            self._embedding_repo = EmbeddingRepository(self.db)
        return self._embedding_repo

    @property
    def fts_repo(self) -> FTS5SearchRepository:
        """Get or create FTS5SearchRepository."""
        if self._fts_repo is None:
            self._fts_repo = FTS5SearchRepository(self.db)
        return self._fts_repo

    # --- LLM component accessors ---

    async def get_llm_provider(self) -> LLMProvider:
        """Get or create LLM provider (lazy initialization).

        Returns:
            Configured LLM provider.

        Raises:
            RuntimeError: If LLM provider is not available.
        """
        if self._llm_provider is None:
            self._llm_provider = create_llm_provider(self.config)
            logger.debug(f"LLM provider created: {self.config.llm_provider}")
        return self._llm_provider

    async def get_embedding_generator(self) -> EmbeddingGenerator:
        """Get or create EmbeddingGenerator.

        Returns:
            Configured embedding generator.
        """
        if self._embedding_generator is None:
            provider = await self.get_llm_provider()
            self._embedding_generator = EmbeddingGenerator(
                provider, self.embedding_repo, self.config
            )
        return self._embedding_generator

    async def get_query_expander(self) -> QueryExpander:
        """Get or create QueryExpander.

        Returns:
            Configured query expander.
        """
        if self._query_expander is None:
            provider = await self.get_llm_provider()
            self._query_expander = QueryExpander(provider)
        return self._query_expander

    async def get_reranker(self) -> DocumentReranker:
        """Get or create DocumentReranker.

        Returns:
            Configured document reranker.
        """
        if self._reranker is None:
            provider = await self.get_llm_provider()
            self._reranker = DocumentReranker(provider)
        return self._reranker

    # --- Metadata component accessors ---

    def get_tag_matcher(self) -> "LexicalTagMatcher | None":
        """Get or create LexicalTagMatcher.

        Returns:
            Configured tag matcher, or None if not available.
        """
        if self._tag_matcher is None:
            try:
                from ..metadata import create_default_matcher
                self._tag_matcher = create_default_matcher()
            except Exception as e:
                logger.debug(f"Failed to create tag matcher: {e}")
                return None
        return self._tag_matcher

    def get_ontology(self) -> "Ontology | None":
        """Get or create Ontology.

        Returns:
            Configured ontology, or None if not available.
        """
        if self._ontology is None:
            try:
                from ..metadata import load_default_ontology
                self._ontology = load_default_ontology()
            except Exception as e:
                logger.debug(f"Failed to load ontology: {e}")
                return None
        return self._ontology

    def get_tag_retriever(self) -> "TagRetriever | None":
        """Get or create TagRetriever.

        Returns:
            Configured tag retriever, or None if metadata repo not available.
        """
        if self._tag_retriever is None:
            metadata_repo = self.get_metadata_repo()
            if metadata_repo:
                from ..metadata import create_tag_retriever
                self._tag_retriever = create_tag_retriever(self.db, metadata_repo)
        return self._tag_retriever

    def get_metadata_repo(self) -> "DocumentMetadataRepository | None":
        """Get or create DocumentMetadataRepository.

        Returns:
            Document metadata repository.
        """
        if self._metadata_repo is None:
            from ..metadata import DocumentMetadataRepository
            self._metadata_repo = DocumentMetadataRepository(self.db)
        return self._metadata_repo

    # --- Service accessors ---

    @property
    def indexing(self) -> "IndexingService":
        """Get IndexingService instance.

        Returns:
            IndexingService for document indexing operations.
        """
        if self._indexing is None:
            from .indexing import IndexingService

            self._indexing = IndexingService.from_container(self)
        return self._indexing

    @property
    def search(self) -> "SearchService":
        """Get SearchService instance.

        Returns:
            SearchService for search operations.
        """
        if self._search is None:
            from .search import SearchService

            self._search = SearchService.from_container(self)
        return self._search

    @property
    def status(self) -> "StatusService":
        """Get StatusService instance.

        Returns:
            StatusService for index status operations.
        """
        if self._status is None:
            from .status import StatusService

            self._status = StatusService.from_container(self)
        return self._status

    # --- Utility methods ---

    @property
    def vec_available(self) -> bool:
        """Check if vector search is available.

        Returns:
            True if sqlite-vec extension is loaded.
        """
        return self.db.vec_available

    async def is_llm_available(self) -> bool:
        """Check if LLM provider is available.

        Returns:
            True if LLM provider can be reached.
        """
        try:
            provider = await self.get_llm_provider()
            return await provider.is_available()
        except Exception:
            return False
