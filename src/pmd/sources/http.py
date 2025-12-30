"""HTTP/HTTPS document source.

This module provides a document source that fetches documents from HTTP/HTTPS
URLs, with support for sitemaps, authentication, and caching.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any, Iterator
from urllib.parse import urljoin, urlparse

import httpx
from loguru import logger

from .auth import AuthConfig, CredentialResolver, get_default_resolver
from .base import (
    BaseDocumentSource,
    DocumentReference,
    FetchResult,
    SourceCapabilities,
    SourceConfig,
    SourceFetchError,
    SourceListError,
)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class HTTPConfig:
    """Configuration for HTTP source.

    Attributes:
        base_url: Base URL for the source.
        urls: Explicit list of URLs to fetch (optional).
        sitemap_url: URL to sitemap.xml for document discovery (optional).
        auth: Authentication configuration.
        timeout_seconds: Request timeout in seconds.
        max_retries: Maximum number of retries on failure.
        follow_redirects: Whether to follow HTTP redirects.
        allowed_content_types: List of allowed MIME types.
        user_agent: User-Agent header value.
    """

    base_url: str
    urls: list[str] | None = None
    sitemap_url: str | None = None
    auth: AuthConfig = field(default_factory=AuthConfig)
    timeout_seconds: int = 30
    max_retries: int = 3
    follow_redirects: bool = True
    allowed_content_types: list[str] = field(
        default_factory=lambda: [
            "text/html",
            "text/markdown",
            "text/plain",
            "application/json",
        ]
    )
    user_agent: str = "pmd/1.0 (Document Indexer)"

    @classmethod
    def from_source_config(cls, config: SourceConfig) -> "HTTPConfig":
        """Create HTTPConfig from generic SourceConfig.

        Args:
            config: Generic source configuration.

        Returns:
            HTTPConfig instance.
        """
        auth_data = {
            "auth_type": config.get("auth_type", "none"),
            "token": config.get("auth_token") or config.get("token"),
            "username": config.get("username"),
            "api_key_header": config.get("api_key_header", "X-API-Key"),
        }

        return cls(
            base_url=config.uri,
            urls=config.get("urls"),
            sitemap_url=config.get("sitemap_url"),
            auth=AuthConfig.from_dict(auth_data),
            timeout_seconds=config.get("timeout_seconds", 30),
            max_retries=config.get("max_retries", 3),
            follow_redirects=config.get("follow_redirects", True),
            allowed_content_types=config.get(
                "allowed_content_types",
                ["text/html", "text/markdown", "text/plain", "application/json"],
            ),
            user_agent=config.get("user_agent", "pmd/1.0 (Document Indexer)"),
        )


# =============================================================================
# Source Implementation
# =============================================================================


class HTTPSource(BaseDocumentSource):
    """Document source for HTTP/HTTPS URLs.

    Fetches documents from web servers with support for:
    - Sitemap-based document discovery
    - Explicit URL lists
    - ETag/Last-Modified caching
    - Various authentication methods

    Example:
        config = SourceConfig(
            uri="https://docs.example.com",
            extra={
                "sitemap_url": "https://docs.example.com/sitemap.xml",
                "auth_type": "bearer",
                "auth_token": "$ENV:API_TOKEN",
            }
        )
        source = HTTPSource(config)

        for ref in source.list_documents():
            result = await source.fetch_content(ref)
            print(f"{ref.path}: {len(result.content)} chars")
    """

    def __init__(self, config: SourceConfig) -> None:
        """Initialize HTTP source.

        Args:
            config: Source configuration with URI and options.
        """
        self._config = HTTPConfig.from_source_config(config)
        self._resolver = get_default_resolver()
        self._client: httpx.AsyncClient | None = None

    @property
    def base_url(self) -> str:
        """Get the base URL."""
        return self._config.base_url

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers including auth."""
        headers = {
            "User-Agent": self._config.user_agent,
            "Accept": ", ".join(self._config.allowed_content_types),
        }
        auth_headers = self._config.auth.get_headers(self._resolver)
        headers.update(auth_headers)
        return headers

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self._config.timeout_seconds,
                follow_redirects=self._config.follow_redirects,
                headers=self._get_headers(),
            )
        return self._client

    async def _close_client(self) -> None:
        """Close HTTP client if open."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def list_documents(self) -> Iterator[DocumentReference]:
        """Enumerate documents from the HTTP source.

        Discovery order:
        1. Explicit URLs if provided
        2. Sitemap if configured
        3. Single base URL as fallback

        Yields:
            DocumentReference for each discovered URL.

        Raises:
            SourceListError: If document discovery fails.
        """
        # Use explicit URLs if provided
        if self._config.urls:
            for url in self._config.urls:
                full_url = urljoin(self._config.base_url, url)
                yield self._url_to_reference(full_url)
            return

        # Try sitemap if configured
        if self._config.sitemap_url:
            try:
                yield from self._parse_sitemap_sync(self._config.sitemap_url)
                return
            except Exception as e:
                logger.warning(f"Failed to parse sitemap: {e}, falling back to base URL")

        # Fallback: treat base URL as single document
        yield self._url_to_reference(self._config.base_url)

    def _parse_sitemap_sync(self, sitemap_url: str) -> Iterator[DocumentReference]:
        """Parse sitemap synchronously for list_documents.

        Args:
            sitemap_url: URL to sitemap.xml.

        Yields:
            DocumentReference for each URL in sitemap.
        """
        # Use synchronous request for list_documents
        try:
            response = httpx.get(
                sitemap_url,
                timeout=self._config.timeout_seconds,
                follow_redirects=True,
                headers=self._get_headers(),
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise SourceListError(sitemap_url, f"Failed to fetch sitemap: {e}")

        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as e:
            raise SourceListError(sitemap_url, f"Invalid sitemap XML: {e}")

        # Handle both sitemap and sitemapindex
        namespace = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

        # Check for sitemapindex
        sitemap_refs = root.findall(".//sm:sitemap/sm:loc", namespace)
        if sitemap_refs:
            # Recursively parse sub-sitemaps
            for loc in sitemap_refs:
                if loc.text:
                    yield from self._parse_sitemap_sync(loc.text)
            return

        # Parse regular sitemap
        url_elements = root.findall(".//sm:url/sm:loc", namespace)
        if not url_elements:
            # Try without namespace (some sitemaps don't use it)
            url_elements = root.findall(".//url/loc")

        for loc in url_elements:
            if loc.text:
                yield self._url_to_reference(loc.text)

    def _url_to_reference(self, url: str) -> DocumentReference:
        """Convert URL to DocumentReference.

        Args:
            url: Full URL to the document.

        Returns:
            DocumentReference for the URL.
        """
        parsed = urlparse(url)
        # Use URL path as document path, strip leading slash
        path = parsed.path.lstrip("/") or "index"

        # Clean up path for storage
        if path.endswith("/"):
            path = path.rstrip("/") + "/index"

        return DocumentReference(
            uri=url,
            path=path,
            title=None,
            metadata={},
        )

    async def fetch_content(self, ref: DocumentReference) -> FetchResult:
        """Fetch document content from URL.

        Args:
            ref: Reference to the document to fetch.

        Returns:
            FetchResult with document content and metadata.

        Raises:
            SourceFetchError: If fetching fails.
        """
        client = await self._get_client()

        # Add conditional headers if we have cached metadata
        headers = {}
        if ref.metadata.get("etag"):
            headers["If-None-Match"] = ref.metadata["etag"]
        if ref.metadata.get("last_modified"):
            headers["If-Modified-Since"] = ref.metadata["last_modified"]

        for attempt in range(self._config.max_retries):
            try:
                response = await client.get(ref.uri, headers=headers)

                # Handle 304 Not Modified
                if response.status_code == 304:
                    raise SourceFetchError(
                        ref.uri,
                        "Document not modified (304)",
                        retryable=False,
                    )

                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    raise SourceFetchError(
                        ref.uri,
                        f"Rate limited (429), retry after {retry_after}s",
                        retryable=True,
                    )

                response.raise_for_status()

                # Check content type
                content_type = response.headers.get("Content-Type", "text/plain")
                base_content_type = content_type.split(";")[0].strip()

                # Get content
                content = response.text

                # Convert HTML to plain text if needed
                if base_content_type == "text/html":
                    content = self._extract_text_from_html(content)
                    base_content_type = "text/plain"

                # Build metadata
                metadata: dict[str, Any] = {
                    "http_status": response.status_code,
                }
                if "ETag" in response.headers:
                    metadata["etag"] = response.headers["ETag"]
                if "Last-Modified" in response.headers:
                    metadata["last_modified"] = response.headers["Last-Modified"]

                return FetchResult(
                    content=content,
                    content_type=base_content_type,
                    encoding=response.encoding or "utf-8",
                    metadata=metadata,
                )

            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:
                    # Server error, maybe retry
                    if attempt < self._config.max_retries - 1:
                        logger.warning(
                            f"HTTP {e.response.status_code} for {ref.uri}, "
                            f"retrying ({attempt + 1}/{self._config.max_retries})"
                        )
                        continue
                raise SourceFetchError(
                    ref.uri,
                    f"HTTP {e.response.status_code}: {e.response.reason_phrase}",
                    retryable=e.response.status_code >= 500,
                )

            except httpx.TimeoutException:
                if attempt < self._config.max_retries - 1:
                    logger.warning(
                        f"Timeout for {ref.uri}, "
                        f"retrying ({attempt + 1}/{self._config.max_retries})"
                    )
                    continue
                raise SourceFetchError(ref.uri, "Request timed out", retryable=True)

            except httpx.RequestError as e:
                raise SourceFetchError(ref.uri, str(e), retryable=True)

        raise SourceFetchError(ref.uri, "Max retries exceeded", retryable=True)

    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML.

        Simple extraction without external dependencies.
        For better results, consider using beautifulsoup4 or similar.

        Args:
            html: HTML content.

        Returns:
            Extracted text content.
        """
        # Remove script and style elements
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML comments
        html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)

        # Replace block elements with newlines
        html = re.sub(r"<(p|div|br|h[1-6]|li|tr)[^>]*>", "\n", html, flags=re.IGNORECASE)

        # Remove remaining tags
        text = re.sub(r"<[^>]+>", "", html)

        # Decode HTML entities
        import html as html_module
        text = html_module.unescape(text)

        # Clean up whitespace
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        return text

    def capabilities(self) -> SourceCapabilities:
        """Return HTTP source capabilities."""
        return SourceCapabilities(
            supports_incremental=True,
            supports_etag=True,
            supports_last_modified=True,
            supports_streaming=False,
            is_readonly=True,
        )

    async def check_modified(
        self,
        ref: DocumentReference,
        stored_metadata: dict[str, Any],
    ) -> bool:
        """Check if document has been modified using HEAD request.

        Args:
            ref: Reference to the document.
            stored_metadata: Metadata from previous fetch.

        Returns:
            True if document may have changed, False if definitely unchanged.
        """
        # If no stored ETag or Last-Modified, assume modified
        stored_etag = stored_metadata.get("etag")
        stored_last_modified = stored_metadata.get("last_modified")

        if not stored_etag and not stored_last_modified:
            return True

        try:
            client = await self._get_client()
            response = await client.head(ref.uri)
            response.raise_for_status()

            # Check ETag
            current_etag = response.headers.get("ETag")
            if stored_etag and current_etag:
                if stored_etag == current_etag:
                    return False

            # Check Last-Modified
            current_last_modified = response.headers.get("Last-Modified")
            if stored_last_modified and current_last_modified:
                if stored_last_modified == current_last_modified:
                    return False

            return True

        except Exception as e:
            logger.debug(f"HEAD request failed for {ref.uri}: {e}, assuming modified")
            return True
