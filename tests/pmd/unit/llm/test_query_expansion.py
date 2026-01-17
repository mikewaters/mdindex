"""Tests for QueryExpander."""

import pytest

from pmd.llm.query_expansion import QueryExpander

from .conftest import MockLLMProvider


@pytest.fixture
def query_expander(mock_llm_provider: MockLLMProvider) -> QueryExpander:
    """Create query expander for testing."""
    return QueryExpander(mock_llm_provider)


class TestQueryExpanderInit:
    """Tests for QueryExpander initialization."""

    def test_stores_llm_provider(self, query_expander, mock_llm_provider):
        """Should store LLM provider."""
        assert query_expander.llm == mock_llm_provider

    def test_gets_default_model(self, query_expander):
        """Should get default expansion model from provider."""
        assert query_expander.model == "mock-expansion-model"


class TestQueryExpanderExpand:
    """Tests for QueryExpander.expand method."""

    @pytest.mark.asyncio
    async def test_expand_returns_original_query(self, query_expander):
        """expand should always include original query."""
        result = await query_expander.expand("test query")

        assert "test query" in result
        assert result[0] == "test query"

    @pytest.mark.asyncio
    async def test_expand_returns_variations(self):
        """expand should include LLM-generated variations."""
        mock_provider = MockLLMProvider(
            generate_result="alternative phrasing\ndifferent wording"
        )
        expander = QueryExpander(mock_provider)

        result = await expander.expand("test query", num_variations=2)

        assert len(result) >= 1
        assert result[0] == "test query"

    @pytest.mark.asyncio
    async def test_expand_respects_num_variations(self):
        """expand should return correct number of variations."""
        mock_provider = MockLLMProvider(
            generate_result="var1\nvar2\nvar3\nvar4\nvar5"
        )
        expander = QueryExpander(mock_provider)

        result = await expander.expand("query", num_variations=2)

        # Original + 2 variations = 3
        assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_expand_handles_llm_failure(self, query_expander):
        """expand should return original on LLM failure."""
        # Default mock returns None for generate

        result = await query_expander.expand("test query")

        # Should still have original
        assert result == ["test query"]

    @pytest.mark.asyncio
    async def test_expand_calls_llm_with_prompt(self, mock_llm_provider):
        """expand should call LLM with expansion prompt."""
        mock_provider = MockLLMProvider(generate_result="variation")
        expander = QueryExpander(mock_provider)

        await expander.expand("find python code")

        assert len(mock_provider.generate_calls) == 1
        prompt = mock_provider.generate_calls[0][0]
        assert "find python code" in prompt
        assert "Alternative" in prompt or "alternative" in prompt


class TestQueryExpanderExpandWithSemantics:
    """Tests for QueryExpander.expand_with_semantics method."""

    @pytest.mark.asyncio
    async def test_expand_with_semantics_returns_original(self, query_expander):
        """expand_with_semantics should include original query."""
        result = await query_expander.expand_with_semantics("test query")

        assert result[0] == "test query"

    @pytest.mark.asyncio
    async def test_expand_with_semantics_uses_different_prompt(self):
        """expand_with_semantics should use semantic prompt."""
        mock_provider = MockLLMProvider(generate_result="semantic variation")
        expander = QueryExpander(mock_provider)

        await expander.expand_with_semantics("test query")

        prompt = mock_provider.generate_calls[0][0]
        assert "semantic" in prompt.lower()


class TestQueryExpanderBuildPrompts:
    """Tests for prompt building methods."""

    def test_build_expansion_prompt(self, query_expander):
        """_build_expansion_prompt should create proper prompt."""
        prompt = query_expander._build_expansion_prompt("test query", 3)

        assert "test query" in prompt
        assert "3" in prompt
        assert "Alternative" in prompt

    def test_build_semantic_prompt(self, query_expander):
        """_build_semantic_prompt should create semantic prompt."""
        prompt = query_expander._build_semantic_prompt("test query", 2)

        assert "test query" in prompt
        assert "2" in prompt
        assert "semantic" in prompt.lower()


class TestQueryExpanderParseVariations:
    """Tests for _parse_variations static method."""

    def test_parse_simple_lines(self):
        """Should parse simple newline-separated lines."""
        response = "variation one\nvariation two\nvariation three"

        result = QueryExpander._parse_variations(response, 3)

        assert len(result) == 3
        assert result[0] == "variation one"
        assert result[1] == "variation two"

    def test_parse_numbered_lines(self):
        """Should remove numbering from lines."""
        response = "1. first variation\n2. second variation\n3. third"

        result = QueryExpander._parse_variations(response, 3)

        assert "1." not in result[0]
        assert result[0] == "first variation"

    def test_parse_bulleted_lines(self):
        """Should remove bullet points from lines."""
        response = "- variation one\n* variation two\n• variation three"

        result = QueryExpander._parse_variations(response, 3)

        assert not result[0].startswith("-")
        assert not result[1].startswith("*")

    def test_parse_removes_trailing_punctuation(self):
        """Should remove trailing punctuation."""
        response = "variation one.\nvariation two?\nvariation three!"

        result = QueryExpander._parse_variations(response, 3)

        assert not result[0].endswith(".")
        assert not result[1].endswith("?")
        assert not result[2].endswith("!")

    def test_parse_respects_limit(self):
        """Should respect num_variations limit."""
        response = "one\ntwo\nthree\nfour\nfive"

        result = QueryExpander._parse_variations(response, 2)

        assert len(result) == 2

    def test_parse_handles_empty_lines(self):
        """Should skip empty lines."""
        response = "variation one\n\n\nvariation two\n"

        result = QueryExpander._parse_variations(response, 3)

        assert len(result) == 2
        assert "" not in result

    def test_parse_strips_whitespace(self):
        """Should strip whitespace from lines."""
        response = "  variation one  \n  variation two  "

        result = QueryExpander._parse_variations(response, 2)

        assert result[0] == "variation one"
        assert result[1] == "variation two"


class TestQueryExpanderIntegration:
    """Integration tests for QueryExpander."""

    @pytest.mark.asyncio
    async def test_full_expansion_flow(self):
        """Test complete expansion with realistic response."""
        mock_provider = MockLLMProvider(
            generate_result="""1. how to search for Python code
2. finding Python snippets
3. locate Python programming examples"""
        )
        expander = QueryExpander(mock_provider)

        result = await expander.expand("python code search", num_variations=2)

        assert result[0] == "python code search"
        assert len(result) == 3  # original + 2 variations
        assert "how to search for Python code" in result

    @pytest.mark.asyncio
    async def test_handles_exception_in_generate(self):
        """Should handle exceptions gracefully."""

        class FailingProvider(MockLLMProvider):
            async def generate(self, *args, **kwargs):
                raise Exception("API Error")

        expander = QueryExpander(FailingProvider())

        result = await expander.expand("test query")

        assert result == ["test query"]
