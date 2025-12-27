"""Query expansion for improving search recall."""

from .base import LLMProvider


class QueryExpander:
    """Expands queries into semantic variations."""

    def __init__(self, llm_provider: LLMProvider):
        """Initialize query expander.

        Args:
            llm_provider: LLM provider for generating variations.
        """
        self.llm = llm_provider
        self.model = llm_provider.get_default_expansion_model()

    async def expand(self, query: str, num_variations: int = 2) -> list[str]:
        """Generate query variations for better recall.

        Returns the original query plus variations.

        Args:
            query: Original search query.
            num_variations: Number of variations to generate (default: 2).

        Returns:
            List of query strings: [original, variation1, variation2, ...].
        """
        variations = [query]  # Always include original

        prompt = self._build_expansion_prompt(query, num_variations)

        try:
            response = await self.llm.generate(
                prompt,
                model=self.model,
                max_tokens=100,
                temperature=0.7,
            )

            if response:
                # Parse variations from response
                expanded = self._parse_variations(response, num_variations)
                variations.extend(expanded)
        except Exception:
            # Fail gracefully, return original query only
            pass

        return variations[:num_variations + 1]  # Include original + N variations

    async def expand_with_semantics(self, query: str, num_variations: int = 2) -> list[str]:
        """Generate query variations with semantic understanding.

        Uses a more sophisticated prompt to generate semantically similar queries.

        Args:
            query: Original search query.
            num_variations: Number of variations to generate.

        Returns:
            List of query strings.
        """
        variations = [query]

        prompt = self._build_semantic_prompt(query, num_variations)

        try:
            response = await self.llm.generate(
                prompt,
                model=self.model,
                max_tokens=200,
                temperature=0.8,
            )

            if response:
                expanded = self._parse_variations(response, num_variations)
                variations.extend(expanded)
        except Exception:
            pass

        return variations[:num_variations + 1]

    def _build_expansion_prompt(self, query: str, num_variations: int) -> str:
        """Build prompt for query expansion.

        Args:
            query: Original query.
            num_variations: Number of variations requested.

        Returns:
            Prompt string.
        """
        return f"""Generate {num_variations} alternative phrasings for this search query that capture the same intent but use different wording. Return ONLY the variations, one per line, with no numbering, bullets, or explanations.

Original Query: {query}

Alternative Phrasings:"""

    def _build_semantic_prompt(self, query: str, num_variations: int) -> str:
        """Build prompt for semantic query expansion.

        Args:
            query: Original query.
            num_variations: Number of variations requested.

        Returns:
            Prompt string.
        """
        return f"""Generate {num_variations} semantically similar but differently phrased search queries that would retrieve similar documents. Consider synonyms, related concepts, and alternative ways to express the same information need.

Original Query: {query}

Semantically Similar Queries:"""

    @staticmethod
    def _parse_variations(response: str, num_variations: int) -> list[str]:
        """Parse variations from LLM response.

        Args:
            response: Raw response from LLM.
            num_variations: Expected number of variations.

        Returns:
            List of parsed query variations.
        """
        # Split by newlines and filter empty lines
        lines = [line.strip() for line in response.split("\n") if line.strip()]

        # Remove common prefixes (numbering, bullets, etc.)
        cleaned = []
        for line in lines:
            # Remove leading numbering (1., 2., etc.)
            if line[0].isdigit() and "." in line[:3]:
                line = line[3:].strip()

            # Remove leading bullets or dashes
            for prefix in ["- ", "* ", "• "]:
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()

            # Remove trailing punctuation if it's just a sentence ender
            if line.endswith((".", "?", "!", ":")):
                line = line[:-1].strip()

            if line:
                cleaned.append(line)

        return cleaned[:num_variations]
