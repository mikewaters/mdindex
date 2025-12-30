"""Document source abstraction for PMD.

This module provides a protocol-based abstraction for sourcing documents
from various backends: filesystem, HTTP, custom entity URIs, etc.

Quick Start
-----------
Use the registry to resolve URIs to document sources:

    from pmd.sources import get_default_registry

    registry = get_default_registry()
    source = registry.resolve("https://docs.example.com")

    for ref in source.list_documents():
        result = await source.fetch_content(ref)
        print(ref.path, len(result.content))

Supported URI Schemes
---------------------
- ``file://`` or bare paths: FileSystemSource (local filesystem)
- ``http://`` and ``https://``: HTTPSource (web pages, APIs)
- ``entity://``: EntitySource (custom backends via resolvers)

FileSystemSource
----------------
Sources documents from the local filesystem using glob patterns:

    config = SourceConfig(
        uri="/path/to/docs",
        extra={"glob_pattern": "**/*.md"}
    )
    source = FileSystemSource(config)

HTTPSource
----------
Sources documents from HTTP/HTTPS URLs with support for:
- Sitemap-based document discovery
- ETag and Last-Modified caching
- Bearer, Basic, and API key authentication
- Automatic retry with exponential backoff

    config = SourceConfig(
        uri="https://docs.example.com",
        extra={
            "sitemap_url": "https://docs.example.com/sitemap.xml",
            "auth_type": "bearer",
            "auth_token": "$ENV:API_TOKEN",
        }
    )
    source = HTTPSource(config)

EntitySource
------------
Sources documents from custom backends via pluggable resolvers.
URI format: ``entity://<resolver>/<resource-type>[/<id>][?query]``

    # Register a custom resolver
    from pmd.sources import get_default_entity_registry, EntityResolver

    class MyDatabaseResolver:
        def list_entities(self, resource_type, query):
            # Return EntityInfo objects
            ...

        async def fetch_entity(self, resource_type, entity_id):
            # Return EntityContent
            ...

    registry = get_default_entity_registry()
    registry.register("mydb", MyDatabaseResolver())

    # Use it
    config = SourceConfig(uri="entity://mydb/articles")
    source = EntitySource(config)

Credential Resolution
---------------------
Credentials can be specified using references:

- ``$ENV:VAR_NAME``: Read from environment variable
- ``$KEYRING:key``: Read from system keyring
- ``$STATIC:key``: Read from static store (testing)
- Literal values: Used directly (not recommended for secrets)

    auth = AuthConfig(
        auth_type="bearer",
        token="$ENV:GITHUB_TOKEN"
    )
    headers = auth.get_headers()  # {"Authorization": "Bearer <token>"}

Custom Sources
--------------
Implement the DocumentSource protocol to create custom sources:

    from pmd.sources import DocumentSource, DocumentReference, FetchResult

    class MySource:
        def __init__(self, config: SourceConfig):
            self.config = config

        def list_documents(self) -> Iterator[DocumentReference]:
            yield DocumentReference(uri="...", path="...")

        async def fetch_content(self, ref: DocumentReference) -> FetchResult:
            return FetchResult(content="...", content_type="text/plain")

        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities()

        async def check_modified(self, ref, stored_metadata) -> bool:
            return True

    # Register it
    registry = get_default_registry()
    registry.register("myscheme", MySource)
"""

from .base import (
    BaseDocumentSource,
    DocumentReference,
    DocumentSource,
    FetchResult,
    SourceCapabilities,
    SourceConfig,
    SourceError,
    SourceFetchError,
    SourceListError,
)
from .auth import (
    AuthConfig,
    CredentialError,
    CredentialNotFoundError,
    CredentialProvider,
    CredentialResolver,
    EnvironmentCredentials,
    KeyringCredentials,
    StaticCredentials,
    get_default_resolver,
)
from .filesystem import FileSystemConfig, FileSystemSource
from .entity import (
    EntityContent,
    EntityInfo,
    EntityResolver,
    EntityResolverError,
    EntityResolverRegistry,
    EntitySource,
    ParsedEntityURI,
    ResolverNotFoundError,
    get_default_entity_registry,
    parse_entity_uri,
)
from .http import HTTPConfig, HTTPSource
from .registry import (
    RegistryError,
    SourceCreationError,
    SourceRegistry,
    UnknownSchemeError,
    get_default_registry,
)

__all__ = [
    # Base types
    "BaseDocumentSource",
    "DocumentReference",
    "DocumentSource",
    "FetchResult",
    "SourceCapabilities",
    "SourceConfig",
    # Base exceptions
    "SourceError",
    "SourceFetchError",
    "SourceListError",
    # Registry
    "RegistryError",
    "SourceCreationError",
    "SourceRegistry",
    "UnknownSchemeError",
    "get_default_registry",
    # Filesystem source
    "FileSystemConfig",
    "FileSystemSource",
    # HTTP source
    "HTTPConfig",
    "HTTPSource",
    # Entity source
    "EntityContent",
    "EntityInfo",
    "EntityResolver",
    "EntityResolverError",
    "EntityResolverRegistry",
    "EntitySource",
    "ParsedEntityURI",
    "ResolverNotFoundError",
    "get_default_entity_registry",
    "parse_entity_uri",
    # Auth
    "AuthConfig",
    "CredentialError",
    "CredentialNotFoundError",
    "CredentialProvider",
    "CredentialResolver",
    "EnvironmentCredentials",
    "KeyringCredentials",
    "StaticCredentials",
    "get_default_resolver",
]
