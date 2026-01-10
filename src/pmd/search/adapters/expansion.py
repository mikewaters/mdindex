"""Query expansion adapter.

Wraps QueryExpander to implement the QueryExpander protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pmd.llm.query_expansion import QueryExpander


class LLMQueryExpanderAdapter:
    """Adapter that wraps QueryExpander for the QueryExpander protocol.

    This adapter provides a thin wrapper around the existing QueryExpander,
    which already matches the protocol signature closely.

    Example:
        >>> from pmd.llm.query_expansion import QueryExpander
        >>> llm_expander = QueryExpander(llm_provider)
        >>> adapter = LLMQueryExpanderAdapter(llm_expander)
        >>> variations = await adapter.expand("python tutorial", num_variations=2)
    """

    def __init__(self, query_expander: "QueryExpander"):
        """Initialize with query expander.

        Args:
            query_expander: QueryExpander instance.
        """
        self._expander = query_expander

    async def expand(
        self,
        query: str,
        num_variations: int = 2,
    ) -> list[str]:
        """Expand query into variations.

        Args:
            query: Original search query.
            num_variations: Number of variations to generate.

        Returns:
            List of query variations (includes original query first).
        """
        return await self._expander.expand(query, num_variations=num_variations)
