"""Tests for HTTPSource."""

import pytest
import httpx
import respx

from pmd.sources import (
    HTTPSource,
    SourceConfig,
    SourceFetchError,
    SourceListError,
    DocumentReference,
)


class TestHTTPSource:
    """Tests for HTTPSource."""

    def test_create_from_config(self):
        """Can create HTTPSource from SourceConfig."""
        config = SourceConfig(
            uri="https://docs.example.com",
            extra={"timeout_seconds": 60},
        )
        source = HTTPSource(config)

        assert source.base_url == "https://docs.example.com"

    def test_list_documents_explicit_urls(self):
        """list_documents uses explicit URLs if provided."""
        config = SourceConfig(
            uri="https://example.com",
            extra={"urls": ["/doc1.html", "/doc2.html"]},
        )
        source = HTTPSource(config)

        docs = list(source.list_documents())
        assert len(docs) == 2
        assert docs[0].uri == "https://example.com/doc1.html"
        assert docs[1].uri == "https://example.com/doc2.html"

    def test_list_documents_fallback_to_base_url(self):
        """list_documents yields base URL as single doc if no sitemap/urls."""
        config = SourceConfig(uri="https://example.com/page")
        source = HTTPSource(config)

        docs = list(source.list_documents())
        assert len(docs) == 1
        assert docs[0].uri == "https://example.com/page"

    @respx.mock
    def test_list_documents_parses_sitemap(self):
        """list_documents parses sitemap.xml."""
        sitemap_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
            <url><loc>https://example.com/page1</loc></url>
            <url><loc>https://example.com/page2</loc></url>
        </urlset>
        """

        respx.get("https://example.com/sitemap.xml").respond(
            200, text=sitemap_xml, headers={"Content-Type": "application/xml"}
        )

        config = SourceConfig(
            uri="https://example.com",
            extra={"sitemap_url": "https://example.com/sitemap.xml"},
        )
        source = HTTPSource(config)

        docs = list(source.list_documents())
        uris = {d.uri for d in docs}

        assert "https://example.com/page1" in uris
        assert "https://example.com/page2" in uris

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_content_success(self):
        """fetch_content retrieves document content."""
        respx.get("https://example.com/doc").respond(
            200,
            text="# Hello World",
            headers={"Content-Type": "text/markdown"},
        )

        config = SourceConfig(uri="https://example.com")
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/doc", path="doc")
        result = await source.fetch_content(ref)

        assert "Hello World" in result.content
        assert result.metadata["http_status"] == 200

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_content_extracts_text_from_html(self):
        """fetch_content extracts text from HTML."""
        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <script>console.log('ignore');</script>
            <p>Hello World</p>
        </body>
        </html>
        """
        respx.get("https://example.com/page").respond(
            200, text=html, headers={"Content-Type": "text/html"}
        )

        config = SourceConfig(uri="https://example.com")
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/page", path="page")
        result = await source.fetch_content(ref)

        assert "Hello World" in result.content
        assert "console.log" not in result.content

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_content_includes_etag(self):
        """fetch_content captures ETag header."""
        respx.get("https://example.com/doc").respond(
            200,
            text="content",
            headers={"ETag": '"abc123"', "Content-Type": "text/plain"},
        )

        config = SourceConfig(uri="https://example.com")
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/doc", path="doc")
        result = await source.fetch_content(ref)

        assert result.metadata["etag"] == '"abc123"'

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_content_404_raises(self):
        """fetch_content raises for 404 response."""
        respx.get("https://example.com/missing").respond(404)

        config = SourceConfig(uri="https://example.com")
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/missing", path="missing")

        with pytest.raises(SourceFetchError, match="404"):
            await source.fetch_content(ref)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_content_rate_limit_raises_retryable(self):
        """fetch_content raises retryable error for 429."""
        respx.get("https://example.com/doc").respond(
            429, headers={"Retry-After": "60"}
        )

        config = SourceConfig(uri="https://example.com")
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/doc", path="doc")

        with pytest.raises(SourceFetchError) as exc_info:
            await source.fetch_content(ref)

        assert exc_info.value.retryable is True
        assert "429" in str(exc_info.value)

    @pytest.mark.asyncio
    @respx.mock
    async def test_fetch_content_retries_on_server_error(self):
        """fetch_content retries on 5xx errors."""
        route = respx.get("https://example.com/doc")
        route.side_effect = [
            httpx.Response(500),
            httpx.Response(500),
            httpx.Response(200, text="success"),
        ]

        config = SourceConfig(
            uri="https://example.com",
            extra={"max_retries": 3},
        )
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/doc", path="doc")
        result = await source.fetch_content(ref)

        assert result.content == "success"
        assert route.call_count == 3

    def test_capabilities(self):
        """HTTPSource reports correct capabilities."""
        config = SourceConfig(uri="https://example.com")
        source = HTTPSource(config)

        caps = source.capabilities()
        assert caps.supports_incremental is True
        assert caps.supports_etag is True
        assert caps.supports_last_modified is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_check_modified_uses_head_request(self):
        """check_modified uses HEAD request for efficiency."""
        respx.head("https://example.com/doc").respond(
            200, headers={"ETag": '"abc123"'}
        )

        config = SourceConfig(uri="https://example.com")
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/doc", path="doc")

        # Same ETag = not modified
        is_modified = await source.check_modified(ref, {"etag": '"abc123"'})
        assert is_modified is False

        # Different ETag = modified
        is_modified = await source.check_modified(ref, {"etag": '"old"'})
        assert is_modified is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_auth_bearer_token(self):
        """HTTPSource includes bearer token in requests."""
        import os
        os.environ["TEST_TOKEN"] = "secret123"

        respx.get("https://example.com/doc").respond(200, text="content")

        config = SourceConfig(
            uri="https://example.com",
            extra={
                "auth_type": "bearer",
                "auth_token": "$ENV:TEST_TOKEN",
            },
        )
        source = HTTPSource(config)

        ref = DocumentReference(uri="https://example.com/doc", path="doc")
        await source.fetch_content(ref)

        # Check that Authorization header was sent
        request = respx.calls.last.request
        assert "Authorization" in request.headers
        assert request.headers["Authorization"] == "Bearer secret123"

        del os.environ["TEST_TOKEN"]
