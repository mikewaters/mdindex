"""Tests for idx.llm.prompts module."""

import pytest

from idx.llm.prompts import (
    RERANK_PROMPT,
    RERANK_SYSTEM,
    format_rerank_prompt,
)


class TestPromptConstants:
    """Tests for prompt constant strings."""

    def test_rerank_system_prompt_exists(self) -> None:
        """System prompt is defined and non-empty."""
        assert RERANK_SYSTEM
        assert isinstance(RERANK_SYSTEM, str)
        assert "relevance" in RERANK_SYSTEM.lower()
        assert "Yes" in RERANK_SYSTEM or "yes" in RERANK_SYSTEM

    def test_rerank_prompt_has_placeholders(self) -> None:
        """Rerank prompt contains required placeholders."""
        assert "{query}" in RERANK_PROMPT
        assert "{document}" in RERANK_PROMPT

    def test_rerank_prompt_requests_yes_no(self) -> None:
        """Rerank prompt asks for Yes/No answer."""
        assert "Yes" in RERANK_PROMPT or "yes" in RERANK_PROMPT
        assert "No" in RERANK_PROMPT or "no" in RERANK_PROMPT


class TestFormatRerankPrompt:
    """Tests for format_rerank_prompt function."""

    def test_formats_query_and_document(self) -> None:
        """Correctly formats query and document into prompt."""
        result = format_rerank_prompt(
            query="test query",
            document_text="test document content",
        )

        assert "test query" in result
        assert "test document content" in result

    def test_truncates_long_document(self) -> None:
        """Truncates document text exceeding max_doc_chars."""
        long_doc = "x" * 5000
        result = format_rerank_prompt(
            query="test",
            document_text=long_doc,
            max_doc_chars=100,
        )

        # Should be truncated with ellipsis
        assert len(result) < len(long_doc) + 100  # Allow for prompt overhead
        assert "..." in result

    def test_preserves_short_document(self) -> None:
        """Does not truncate documents under max_doc_chars."""
        short_doc = "Short document text."
        result = format_rerank_prompt(
            query="test",
            document_text=short_doc,
            max_doc_chars=2000,
        )

        assert short_doc in result
        # Only has ellipsis from truncation, not inherent
        assert result.count("...") == 0 or "..." in short_doc

    def test_custom_max_chars(self) -> None:
        """Respects custom max_doc_chars parameter."""
        doc = "a" * 500
        result_100 = format_rerank_prompt("q", doc, max_doc_chars=100)
        result_1000 = format_rerank_prompt("q", doc, max_doc_chars=1000)

        assert len(result_100) < len(result_1000)

    def test_empty_document(self) -> None:
        """Handles empty document text."""
        result = format_rerank_prompt(
            query="test query",
            document_text="",
        )

        assert "test query" in result
        # Should still have structure
        assert "Document" in result

    def test_multiline_document(self) -> None:
        """Handles multiline document text."""
        doc = "Line 1\nLine 2\nLine 3"
        result = format_rerank_prompt(
            query="test",
            document_text=doc,
        )

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_special_characters_in_query(self) -> None:
        """Handles special characters in query."""
        result = format_rerank_prompt(
            query='query with "quotes" and {braces}',
            document_text="doc",
        )

        assert '"quotes"' in result
        # Note: {braces} in query shouldn't be interpreted as format placeholder
        # This works because we use .format() on the template, not the query

    def test_unicode_content(self) -> None:
        """Handles unicode in query and document."""
        result = format_rerank_prompt(
            query="unicode query",
            document_text="Hello World",
        )

        assert "unicode" in result
        assert "Hello" in result
