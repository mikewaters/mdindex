"""Application composition root and dependency injection.

This module provides:
- Protocol definitions for injectable dependencies (types.py)
- Application factory functions (factories.py, future)
- The Application class for lifecycle management (future)

Example:
    from pmd.app import create_application
    from pmd.core.config import Config

    async with create_application(Config()) as app:
        results = await app.search.hybrid_search("query")
"""

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
