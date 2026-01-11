"""In-memory test fakes for search ports.

These fakes implement the search port protocols using in-memory data
structures, enabling fast unit testing without database or LLM dependencies.

Usage:
    from tests.fakes import InMemoryTextSearcher, StubReranker

    # Create fake with pre-configured results
    text_searcher = InMemoryTextSearcher()
    text_searcher.add_result(SearchResult(...))

    # Use in pipeline
    pipeline = HybridSearchPipeline(text_searcher=text_searcher)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pmd.core.types import SearchResult, SearchSource, RankedResult
from pmd.search.ports import BoostInfo, RerankScore


class InMemoryTextSearcher:
    """In-memory implementation of TextSearcher for testing.

    Stores results in memory and returns them based on query matching.
    Supports both exact query matching and a default result list.

    Example:
        >>> searcher = InMemoryTextSearcher()
        >>> searcher.add_result(make_search_result("doc1.md", score=0.9))
        >>> results = searcher.search("python", limit=10)
    """

    def __init__(self):
        """Initialize with empty result store."""
        self._results: list[SearchResult] = []
        self._query_results: dict[str, list[SearchResult]] = {}

    def add_result(self, result: SearchResult) -> None:
        """Add a result to the default result list.

        Args:
            result: SearchResult to add.
        """
        self._results.append(result)

    def set_results_for_query(self, query: str, results: list[SearchResult]) -> None:
        """Set specific results for a query.

        Args:
            query: Query string to match.
            results: Results to return for this query.
        """
        self._query_results[query] = results

    def search(
        self,
        query: str,
        limit: int,
        source_collection_id: int | None = None,
    ) -> list[SearchResult]:
        """Search for results matching the query.

        Args:
            query: Search query string.
            limit: Maximum number of results.
            source_collection_id: Optional collection filter.

        Returns:
            List of SearchResult objects.
        """
        # Check for query-specific results first
        if query in self._query_results:
            results = self._query_results[query]
        else:
            results = self._results

        # Apply collection filter if specified
        if source_collection_id is not None:
            results = [r for r in results if r.source_collection_id == source_collection_id]

        # Sort by score descending and limit
        results = sorted(results, key=lambda r: r.score, reverse=True)
        return results[:limit]


class InMemoryVectorSearcher:
    """In-memory implementation of VectorSearcher for testing.

    Similar to InMemoryTextSearcher but async to match the protocol.

    Example:
        >>> searcher = InMemoryVectorSearcher()
        >>> searcher.add_result(make_search_result("doc1.md", score=0.85))
        >>> results = await searcher.search("machine learning", limit=10)
    """

    def __init__(self):
        """Initialize with empty result store."""
        self._results: list[SearchResult] = []
        self._query_results: dict[str, list[SearchResult]] = {}

    def add_result(self, result: SearchResult) -> None:
        """Add a result to the default result list."""
        self._results.append(result)

    def set_results_for_query(self, query: str, results: list[SearchResult]) -> None:
        """Set specific results for a query."""
        self._query_results[query] = results

    async def search(
        self,
        query: str,
        limit: int,
        source_collection_id: int | None = None,
    ) -> list[SearchResult]:
        """Search for results matching the query (async).

        Args:
            query: Search query string.
            limit: Maximum number of results.
            source_collection_id: Optional collection filter.

        Returns:
            List of SearchResult objects.
        """
        if query in self._query_results:
            results = self._query_results[query]
        else:
            results = self._results

        if source_collection_id is not None:
            results = [r for r in results if r.source_collection_id == source_collection_id]

        results = sorted(results, key=lambda r: r.score, reverse=True)
        return results[:limit]


class InMemoryTagSearcher:
    """In-memory implementation of TagSearcher for testing.

    Stores documents with their tags and returns matches based on tag overlap.

    Example:
        >>> searcher = InMemoryTagSearcher()
        >>> searcher.add_document("doc1.md", {"python", "web"}, score=0.8)
        >>> results = searcher.search({"python": 1.0}, limit=10)
    """

    def __init__(self):
        """Initialize with empty document store."""
        self._documents: list[tuple[SearchResult, set[str]]] = []

    def add_document(
        self,
        filepath: str,
        tags: set[str],
        score: float = 1.0,
        source_collection_id: int = 1,
        title: str = "",
        body: str = "",
    ) -> None:
        """Add a document with its tags.

        Args:
            filepath: Document path.
            tags: Set of tags for this document.
            score: Base score for the document.
            source_collection_id: Collection ID.
            title: Document title.
            body: Document body.
        """
        result = SearchResult(
            filepath=filepath,
            display_path=filepath,
            title=title or filepath,
            context=None,
            hash=f"hash_{filepath}",
            source_collection_id=source_collection_id,
            modified_at="2024-01-01T00:00:00",
            body_length=len(body),
            body=body,
            score=score,
            source=SearchSource.TAG,
        )
        self._documents.append((result, tags))

    def search(
        self,
        tags: dict[str, float] | set[str],
        limit: int,
        source_collection_id: int | None = None,
    ) -> list[SearchResult]:
        """Search documents by tag matches.

        Args:
            tags: Tags to search for (dict with weights or set).
            limit: Maximum number of results.
            source_collection_id: Optional collection filter.

        Returns:
            List of SearchResult objects sorted by score.
        """
        if isinstance(tags, set):
            tags_dict = {t: 1.0 for t in tags}
        else:
            tags_dict = tags

        results: list[SearchResult] = []

        for result, doc_tags in self._documents:
            if source_collection_id is not None and result.source_collection_id != source_collection_id:
                continue

            # Calculate score based on tag overlap
            score = 0.0
            for tag, weight in tags_dict.items():
                if tag in doc_tags:
                    score += weight

            if score > 0:
                # Create new result with calculated score
                scored_result = SearchResult(
                    filepath=result.filepath,
                    display_path=result.display_path,
                    title=result.title,
                    context=result.context,
                    hash=result.hash,
                    source_collection_id=result.source_collection_id,
                    modified_at=result.modified_at,
                    body_length=result.body_length,
                    body=result.body,
                    score=score,
                    source=SearchSource.TAG,
                )
                results.append(scored_result)

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]


class StubQueryExpander:
    """Stub implementation of QueryExpander for testing.

    Returns pre-configured variations or echoes the query.

    Example:
        >>> expander = StubQueryExpander(["variation1", "variation2"])
        >>> variations = await expander.expand("python", num_variations=2)
        >>> # Returns ["python", "variation1", "variation2"]
    """

    def __init__(self, variations: list[str] | None = None):
        """Initialize with optional variations.

        Args:
            variations: List of variations to return (after original).
        """
        self._variations = variations or []

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
            List with original query plus configured variations.
        """
        result = [query]
        result.extend(self._variations[:num_variations])
        return result


class StubReranker:
    """Stub implementation of Reranker for testing.

    Returns pre-configured scores or calculates scores based on a function.

    Example:
        >>> reranker = StubReranker(default_score=0.8)
        >>> scores = await reranker.rerank("query", candidates)
    """

    def __init__(
        self,
        default_score: float = 0.5,
        score_func: Callable[[str, RankedResult], float] | None = None,
        scores: dict[str, float] | None = None,
    ):
        """Initialize with scoring configuration.

        Args:
            default_score: Default score for all documents.
            score_func: Optional function(query, candidate) -> score.
            scores: Optional dict mapping file paths to scores.
        """
        self._default_score = default_score
        self._score_func = score_func
        self._scores = scores or {}

    async def rerank(
        self,
        query: str,
        candidates: list[RankedResult],
    ) -> list[RerankScore]:
        """Rerank candidate documents.

        Args:
            query: Search query.
            candidates: Candidate documents.

        Returns:
            List of RerankScore objects.
        """
        results: list[RerankScore] = []

        for candidate in candidates:
            if self._score_func:
                score = self._score_func(query, candidate)
            elif candidate.file in self._scores:
                score = self._scores[candidate.file]
            else:
                score = self._default_score

            results.append(
                RerankScore(
                    file=candidate.file,
                    score=score,
                    relevant=score > 0.5,
                    confidence=score,
                )
            )

        return results


class InMemoryMetadataBooster:
    """In-memory implementation of MetadataBooster for testing.

    Stores document tags and applies boosting based on tag matches.

    Example:
        >>> booster = InMemoryMetadataBooster()
        >>> booster.add_document_tags("doc1.md", {"python", "web"})
        >>> boosted = booster.boost(results, {"python": 1.0})
    """

    def __init__(self, boost_factor: float = 1.15, max_boost: float = 2.0):
        """Initialize with boost configuration.

        Args:
            boost_factor: Base boost factor.
            max_boost: Maximum allowed boost.
        """
        self._doc_tags: dict[str, set[str]] = {}
        self._boost_factor = boost_factor
        self._max_boost = max_boost

    def add_document_tags(self, filepath: str, tags: set[str]) -> None:
        """Add tags for a document.

        Args:
            filepath: Document path.
            tags: Set of tags for this document.
        """
        self._doc_tags[filepath] = tags

    def boost(
        self,
        results: list[RankedResult],
        query_tags: dict[str, float],
    ) -> list[tuple[RankedResult, BoostInfo]]:
        """Apply metadata-based score boosting.

        Args:
            results: Ranked results to boost.
            query_tags: Tags inferred from query with weights.

        Returns:
            List of (result, boost_info) tuples.
        """
        if not results or not query_tags:
            return [(r, BoostInfo(r.score, r.score, {}, 1.0)) for r in results]

        boosted_results: list[tuple[RankedResult, BoostInfo]] = []

        for result in results:
            doc_tags = self._doc_tags.get(result.file, set())

            # Calculate weighted matches
            matching_tags: dict[str, float] = {}
            for tag, weight in query_tags.items():
                if tag in doc_tags:
                    matching_tags[tag] = weight

            total_match_weight = sum(matching_tags.values())

            if total_match_weight > 0:
                boost = self._boost_factor ** total_match_weight
                boost = min(boost, self._max_boost)
                boosted_score = result.score * boost
            else:
                boost = 1.0
                boosted_score = result.score

            # Create updated result
            from dataclasses import replace
            updated_result = replace(result, score=boosted_score)

            boosted_results.append((
                updated_result,
                BoostInfo(
                    original_score=result.score,
                    boosted_score=boosted_score,
                    matching_tags=matching_tags,
                    boost_applied=boost,
                ),
            ))

        boosted_results.sort(key=lambda x: x[0].score, reverse=True)
        return boosted_results


class InMemoryTagInferencer:
    """In-memory implementation of TagInferencer for testing.

    Returns pre-configured tags for queries.

    Example:
        >>> inferencer = InMemoryTagInferencer()
        >>> inferencer.set_tags_for_query("python ml", {"python", "ml"})
        >>> tags = inferencer.infer_tags("python ml")
    """

    def __init__(self):
        """Initialize with empty tag mappings."""
        self._query_tags: dict[str, set[str]] = {}
        self._expansion: dict[str, dict[str, float]] = {}

    def set_tags_for_query(self, query: str, tags: set[str]) -> None:
        """Set tags that will be inferred for a query.

        Args:
            query: Query string.
            tags: Tags to return for this query.
        """
        self._query_tags[query] = tags

    def set_expansion(self, tag: str, expanded: dict[str, float]) -> None:
        """Set expansion for a tag.

        Args:
            tag: Base tag.
            expanded: Expanded tags with weights.
        """
        self._expansion[tag] = expanded

    def infer_tags(self, query: str) -> set[str]:
        """Infer tags from query text.

        Args:
            query: Search query string.

        Returns:
            Set of inferred tags.
        """
        return self._query_tags.get(query, set())

    def expand_tags(self, tags: set[str]) -> dict[str, float]:
        """Expand tags using configured expansions.

        Args:
            tags: Base tags to expand.

        Returns:
            Dictionary mapping expanded tags to weights.
        """
        result: dict[str, float] = {}

        for tag in tags:
            if tag in self._expansion:
                for expanded_tag, weight in self._expansion[tag].items():
                    # Keep highest weight if duplicate
                    if expanded_tag not in result or result[expanded_tag] < weight:
                        result[expanded_tag] = weight
            else:
                result[tag] = 1.0

        return result


# Helper function to create test SearchResults
def make_search_result(
    filepath: str,
    score: float = 0.5,
    source: SearchSource = SearchSource.FTS,
    source_collection_id: int = 1,
    title: str = "",
    body: str = "",
) -> SearchResult:
    """Create a SearchResult for testing.

    Args:
        filepath: Document path.
        score: Result score.
        source: Search source type.
        source_collection_id: Collection ID.
        title: Document title.
        body: Document body.

    Returns:
        SearchResult instance.
    """
    return SearchResult(
        filepath=filepath,
        display_path=filepath,
        title=title or filepath,
        context=None,
        hash=f"hash_{filepath}",
        source_collection_id=source_collection_id,
        modified_at="2024-01-01T00:00:00",
        body_length=len(body),
        body=body,
        score=score,
        source=source,
    )


def make_ranked_result(
    filepath: str,
    score: float = 0.5,
    title: str = "",
    body: str = "",
) -> RankedResult:
    """Create a RankedResult for testing.

    Args:
        filepath: Document path.
        score: Result score.
        title: Document title.
        body: Document body.

    Returns:
        RankedResult instance.
    """
    return RankedResult(
        file=filepath,
        display_path=filepath,
        title=title or filepath,
        body=body,
        score=score,
    )
