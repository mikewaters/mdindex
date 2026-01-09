"""Query-time tag inference for metadata-boosted search.

Provides dictionary-based inference of tags from search queries,
allowing the search system to boost documents with matching tags.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class TagMatch:
    """A matched tag from query inference.

    Attributes:
        tag: The normalized tag that was matched.
        confidence: Confidence score (0.0 to 1.0).
        match_type: How the match was found (exact, alias, prefix).
        matched_term: The query term that triggered the match.
    """

    tag: str
    confidence: float
    match_type: str
    matched_term: str


class LexicalTagMatcher:
    """Dictionary-based query-time tag inference.

    Matches query terms against known tags using:
    - Exact matching (case-insensitive)
    - Alias matching (configurable synonyms)
    - Prefix matching (optional)

    Example:
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust", "javascript"])
        matcher.register_alias("py", "python")
        matcher.register_alias("js", "javascript")

        matches = matcher.infer_tags("python tutorial for beginners")
        # [TagMatch(tag="python", confidence=1.0, ...)]

        matches = matcher.infer_tags("quick py script")
        # [TagMatch(tag="python", confidence=0.9, match_type="alias", ...)]
    """

    def __init__(
        self,
        *,
        enable_prefix_matching: bool = False,
        min_prefix_length: int = 3,
        alias_confidence: float = 0.9,
        prefix_confidence: float = 0.7,
    ):
        """Initialize the tag matcher.

        Args:
            enable_prefix_matching: Whether to match query terms as tag prefixes.
            min_prefix_length: Minimum length for prefix matching.
            alias_confidence: Confidence score for alias matches.
            prefix_confidence: Confidence score for prefix matches.
        """
        self.enable_prefix_matching = enable_prefix_matching
        self.min_prefix_length = min_prefix_length
        self.alias_confidence = alias_confidence
        self.prefix_confidence = prefix_confidence

        # Known tags (normalized to lowercase)
        self._tags: set[str] = set()
        # Alias -> tag mapping
        self._aliases: dict[str, str] = {}

    def register_tags(self, tags: set[str] | list[str]) -> None:
        """Register known tags for matching.

        Args:
            tags: Collection of tags to register.
        """
        for tag in tags:
            self._tags.add(tag.lower())

    def register_alias(self, alias: str, tag: str) -> None:
        """Register an alias for a tag.

        Args:
            alias: The alias term (e.g., "py").
            tag: The canonical tag it maps to (e.g., "python").
        """
        self._aliases[alias.lower()] = tag.lower()

    def register_aliases(self, aliases: dict[str, str]) -> None:
        """Register multiple aliases at once.

        Args:
            aliases: Dictionary mapping alias -> tag.
        """
        for alias, tag in aliases.items():
            self.register_alias(alias, tag)

    def clear(self) -> None:
        """Clear all registered tags and aliases."""
        self._tags.clear()
        self._aliases.clear()

    def infer_tags(self, query: str) -> list[TagMatch]:
        """Infer tags from a search query.

        Tokenizes the query and matches each term against:
        1. Exact tag matches
        2. Alias matches
        3. Prefix matches (if enabled)

        Args:
            query: Search query string.

        Returns:
            List of TagMatch objects, sorted by confidence (highest first).
        """
        matches: list[TagMatch] = []
        seen_tags: set[str] = set()

        terms = self._tokenize(query)

        for term in terms:
            term_lower = term.lower()

            # 1. Exact tag match
            if term_lower in self._tags:
                if term_lower not in seen_tags:
                    matches.append(TagMatch(
                        tag=term_lower,
                        confidence=1.0,
                        match_type="exact",
                        matched_term=term,
                    ))
                    seen_tags.add(term_lower)
                continue

            # 2. Alias match
            if term_lower in self._aliases:
                tag = self._aliases[term_lower]
                if tag not in seen_tags:
                    matches.append(TagMatch(
                        tag=tag,
                        confidence=self.alias_confidence,
                        match_type="alias",
                        matched_term=term,
                    ))
                    seen_tags.add(tag)
                continue

            # 3. Prefix match (if enabled)
            if self.enable_prefix_matching and len(term_lower) >= self.min_prefix_length:
                prefix_matches = self._find_prefix_matches(term_lower)
                for tag in prefix_matches:
                    if tag not in seen_tags:
                        matches.append(TagMatch(
                            tag=tag,
                            confidence=self.prefix_confidence,
                            match_type="prefix",
                            matched_term=term,
                        ))
                        seen_tags.add(tag)

        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        return matches

    def get_matching_tags(self, query: str) -> set[str]:
        """Get just the tag names that match a query.

        Convenience method that returns only the tag names.

        Args:
            query: Search query string.

        Returns:
            Set of matching tag names.
        """
        matches = self.infer_tags(query)
        return {m.tag for m in matches}

    def _tokenize(self, query: str) -> list[str]:
        """Tokenize a query into terms.

        Splits on whitespace and punctuation, preserving meaningful tokens.

        Args:
            query: Query string to tokenize.

        Returns:
            List of token strings.
        """
        # Split on whitespace and common punctuation
        tokens = re.findall(r"[a-zA-Z0-9_/-]+", query)
        return [t for t in tokens if t]

    def _find_prefix_matches(self, prefix: str) -> list[str]:
        """Find tags that start with the given prefix.

        Args:
            prefix: Lowercase prefix to match.

        Returns:
            List of matching tags.
        """
        return [tag for tag in self._tags if tag.startswith(prefix)]


def load_default_aliases() -> dict[str, str]:
    """Load default tag aliases from the package data file.

    Returns:
        Dictionary mapping alias -> canonical tag.

    Note:
        This function is kept for backward compatibility.
        Prefer using pmd.metadata.load_default_aliases() directly.
    """
    from pmd.metadata import load_default_aliases as _load_aliases

    aliases = _load_aliases()
    return aliases.all_aliases()


def create_default_matcher(known_tags: set[str] | None = None) -> LexicalTagMatcher:
    """Create a matcher with common default aliases.

    Loads aliases from the package data file (data/tag_aliases.json).

    Args:
        known_tags: Optional set of known tags to register.

    Returns:
        Configured LexicalTagMatcher instance.
    """
    matcher = LexicalTagMatcher(
        enable_prefix_matching=False,
        alias_confidence=0.9,
    )

    # Load aliases from package data
    default_aliases = load_default_aliases()
    matcher.register_aliases(default_aliases)

    if known_tags:
        matcher.register_tags(known_tags)

    return matcher
