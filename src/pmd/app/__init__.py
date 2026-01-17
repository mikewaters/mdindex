"""Application composition root and dependency injection.

This module provides:
- Protocol definitions for injectable dependencies (protocols.py)
- The Application class for lifecycle management (application.py)
- Factory functions for creating configured applications (factory.py)

Example:
    from pmd.app import create_application
    from pmd.core.config import Config

    async with create_application(Config()) as app:
        results = await app.search.hybrid_search("query")
"""

from __future__ import annotations

# Application class
from .application import Application

# Factory function
from .factory import create_application

# Protocol definitions
from .protocols import (
    # Data types
    BoostInfo,
    RerankScore,
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
    # Search pipeline protocols
    TextSearcher,
    VectorSearcher,
    TagSearcher,
    QueryExpander,
    Reranker,
    MetadataBooster,
    TagInferencer,
    # Type aliases
    SourceCollectionRepo,
    DocumentRepo,
    FTSRepo,
    EmbeddingRepo,
    LoadingService,
)

__all__ = [
    # Application
    "Application",
    "create_application",
    # Data types
    "BoostInfo",
    "RerankScore",
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
    # Search pipeline protocols
    "TextSearcher",
    "VectorSearcher",
    "TagSearcher",
    "QueryExpander",
    "Reranker",
    "MetadataBooster",
    "TagInferencer",
    # Aliases
    "SourceCollectionRepo",
    "DocumentRepo",
    "FTSRepo",
    "EmbeddingRepo",
    "LoadingService",
]
