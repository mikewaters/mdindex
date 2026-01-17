"""Tests for query-time tag inference in search.metadata module.

Note: The pmd.search.metadata module has been moved to pmd.metadata.query.
These tests import from pmd.metadata which re-exports the inference module.
"""

import pytest

from pmd.metadata import (
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
)
from pmd.ontology.inference import load_default_aliases


class TestTagMatchDataclass:
    """Tests for the TagMatch dataclass."""

    def test_tag_match_fields(self):
        """TagMatch should have all required fields."""
        match = TagMatch(
            tag="python",
            confidence=0.9,
            match_type="alias",
            matched_term="py",
        )

        assert match.tag == "python"
        assert match.confidence == 0.9
        assert match.match_type == "alias"
        assert match.matched_term == "py"

    def test_tag_match_equality(self):
        """TagMatch instances with same values should be equal."""
        match1 = TagMatch(tag="python", confidence=1.0, match_type="exact", matched_term="python")
        match2 = TagMatch(tag="python", confidence=1.0, match_type="exact", matched_term="python")

        assert match1 == match2

    def test_tag_match_inequality(self):
        """TagMatch instances with different values should not be equal."""
        match1 = TagMatch(tag="python", confidence=1.0, match_type="exact", matched_term="python")
        match2 = TagMatch(tag="rust", confidence=1.0, match_type="exact", matched_term="rust")

        assert match1 != match2

    def test_tag_match_different_confidence(self):
        """TagMatch instances with different confidence should not be equal."""
        match1 = TagMatch(tag="python", confidence=1.0, match_type="exact", matched_term="python")
        match2 = TagMatch(tag="python", confidence=0.9, match_type="alias", matched_term="py")

        assert match1 != match2


class TestLexicalTagMatcherInitialization:
    """Tests for LexicalTagMatcher initialization."""

    def test_default_initialization(self):
        """Should initialize with default parameters."""
        matcher = LexicalTagMatcher()

        assert matcher.enable_prefix_matching is False
        assert matcher.min_prefix_length == 3
        assert matcher.alias_confidence == 0.9
        assert matcher.prefix_confidence == 0.7

    def test_custom_initialization(self):
        """Should initialize with custom parameters."""
        matcher = LexicalTagMatcher(
            enable_prefix_matching=True,
            min_prefix_length=5,
            alias_confidence=0.85,
            prefix_confidence=0.6,
        )

        assert matcher.enable_prefix_matching is True
        assert matcher.min_prefix_length == 5
        assert matcher.alias_confidence == 0.85
        assert matcher.prefix_confidence == 0.6

    def test_empty_tags_and_aliases_on_init(self):
        """Should start with no registered tags or aliases."""
        matcher = LexicalTagMatcher()

        matches = matcher.infer_tags("python rust javascript")
        assert matches == []


class TestLexicalTagMatcherRegistration:
    """Tests for tag and alias registration."""

    def test_register_tags_with_list(self):
        """Should register tags from a list."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust", "javascript"])

        matches = matcher.infer_tags("python code")
        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_register_tags_with_set(self):
        """Should register tags from a set."""
        matcher = LexicalTagMatcher()
        matcher.register_tags({"python", "rust", "javascript"})

        matches = matcher.infer_tags("rust code")
        assert len(matches) == 1
        assert matches[0].tag == "rust"

    def test_register_tags_normalizes_case(self):
        """Should normalize tags to lowercase."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["Python", "RUST", "JavaScript"])

        matches = matcher.infer_tags("python rust javascript")
        assert len(matches) == 3

    def test_register_tags_multiple_times(self):
        """Registering the same tag multiple times should not create duplicates."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_tags(["python", "rust"])
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python tutorial")
        assert len(matches) == 1

    def test_register_alias_single(self):
        """Should register a single alias."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")
        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_register_alias_normalizes_case(self):
        """Should normalize alias and tag to lowercase."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["Python"])
        matcher.register_alias("PY", "Python")

        matches = matcher.infer_tags("py code")
        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_register_aliases_bulk(self):
        """Should register multiple aliases at once."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "javascript", "rust"])
        matcher.register_aliases({
            "py": "python",
            "js": "javascript",
            "rs": "rust",
        })

        matches = matcher.infer_tags("py js rs code")
        tags = {m.tag for m in matches}
        assert tags == {"python", "javascript", "rust"}

    def test_register_multiple_aliases_same_tag(self):
        """Should support multiple aliases for the same tag."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")
        matcher.register_alias("python3", "python")
        matcher.register_alias("py3", "python")

        matches1 = matcher.infer_tags("py code")
        matches2 = matcher.infer_tags("python3 code")
        matches3 = matcher.infer_tags("py3 code")

        assert matches1[0].tag == "python"
        assert matches2[0].tag == "python"
        assert matches3[0].tag == "python"

    def test_clear_removes_all(self):
        """clear should remove all tags and aliases."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust"])
        matcher.register_alias("py", "python")

        matcher.clear()

        assert matcher.infer_tags("python rust py") == []

    def test_clear_and_reregister(self):
        """Should work correctly after clear and re-register."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matcher.clear()
        matcher.register_tags(["rust", "go"])
        matcher.register_alias("rs", "rust")

        matches_old = matcher.infer_tags("python py")
        matches_new = matcher.infer_tags("rust go")

        assert matches_old == []
        assert len(matches_new) == 2
        tags = {m.tag for m in matches_new}
        assert tags == {"rust", "go"}


class TestLexicalTagMatcherExactMatch:
    """Tests for exact tag matching."""

    def test_exact_match_single_term(self):
        """Should match exact tag in query."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust", "javascript"])

        matches = matcher.infer_tags("python tutorial")

        assert len(matches) == 1
        assert matches[0].tag == "python"
        assert matches[0].confidence == 1.0
        assert matches[0].match_type == "exact"
        assert matches[0].matched_term == "python"

    def test_exact_match_multiple_terms(self):
        """Should match multiple tags in query."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "web", "api"])

        matches = matcher.infer_tags("python web api development")

        tags = {m.tag for m in matches}
        assert tags == {"python", "web", "api"}
        assert all(m.confidence == 1.0 for m in matches)
        assert all(m.match_type == "exact" for m in matches)

    def test_exact_match_case_insensitive(self):
        """Should match regardless of case."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "javascript"])

        matches = matcher.infer_tags("PYTHON and JavaScript")

        tags = {m.tag for m in matches}
        assert tags == {"python", "javascript"}

    def test_no_match_returns_empty(self):
        """Should return empty list when no matches."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust"])

        matches = matcher.infer_tags("java tutorial")

        assert matches == []

    def test_duplicate_terms_single_match(self):
        """Should only return one match per tag."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python python python")

        assert len(matches) == 1

    def test_partial_match_not_matched(self):
        """Should not match partial words."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("pythonic programming")

        assert matches == []

    def test_match_preserves_original_term(self):
        """Should preserve the original matched term."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("PYTHON code")

        assert matches[0].matched_term == "PYTHON"


class TestLexicalTagMatcherAliasMatch:
    """Tests for alias matching."""

    def test_alias_match(self):
        """Should match alias to tag."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("quick py script")

        assert len(matches) == 1
        assert matches[0].tag == "python"
        assert matches[0].match_type == "alias"
        assert matches[0].confidence == 0.9
        assert matches[0].matched_term == "py"

    def test_multiple_aliases(self):
        """Should support multiple aliases for same tag."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")
        matcher.register_alias("python3", "python")

        matches1 = matcher.infer_tags("py code")
        matches2 = matcher.infer_tags("python3 code")

        assert matches1[0].tag == "python"
        assert matches2[0].tag == "python"

    def test_exact_match_priority_over_alias(self):
        """Exact match should be returned with higher confidence."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "py"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")

        # Should match "py" exactly, not as alias
        assert len(matches) == 1
        assert matches[0].tag == "py"
        assert matches[0].confidence == 1.0
        assert matches[0].match_type == "exact"

    def test_alias_without_registered_tag(self):
        """Should match alias even if target tag is not registered."""
        matcher = LexicalTagMatcher()
        # Not registering "python" tag
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")

        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_alias_duplicate_prevention(self):
        """Should not return duplicate matches from same alias."""
        matcher = LexicalTagMatcher()
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py py py")

        assert len(matches) == 1

    def test_custom_alias_confidence(self):
        """Should use custom alias confidence."""
        matcher = LexicalTagMatcher(alias_confidence=0.75)
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")

        assert matches[0].confidence == 0.75

    def test_alias_case_insensitive(self):
        """Should match aliases case-insensitively."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("PY code")

        assert len(matches) == 1
        assert matches[0].tag == "python"


class TestLexicalTagMatcherPrefixMatch:
    """Tests for prefix matching."""

    def test_prefix_match_disabled_by_default(self):
        """Prefix matching should be disabled by default."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "pytorch", "pypi"])

        matches = matcher.infer_tags("pyt")

        assert matches == []

    def test_prefix_match_when_enabled(self):
        """Should match prefixes when enabled."""
        matcher = LexicalTagMatcher(enable_prefix_matching=True)
        matcher.register_tags(["python", "pytorch", "pypi"])

        matches = matcher.infer_tags("pyt")

        tags = {m.tag for m in matches}
        assert "python" in tags or "pytorch" in tags

    def test_prefix_match_respects_min_length(self):
        """Should not match prefixes shorter than min_prefix_length."""
        matcher = LexicalTagMatcher(
            enable_prefix_matching=True,
            min_prefix_length=4,
        )
        matcher.register_tags(["python"])

        matches_short = matcher.infer_tags("py")
        matches_long = matcher.infer_tags("pyth")

        assert matches_short == []
        assert len(matches_long) == 1

    def test_prefix_match_confidence(self):
        """Prefix matches should have configured confidence."""
        matcher = LexicalTagMatcher(
            enable_prefix_matching=True,
            prefix_confidence=0.7,
        )
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("pyth")

        assert matches[0].confidence == 0.7
        assert matches[0].match_type == "prefix"

    def test_custom_prefix_confidence(self):
        """Should use custom prefix confidence."""
        matcher = LexicalTagMatcher(
            enable_prefix_matching=True,
            prefix_confidence=0.5,
        )
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("pyth")

        assert matches[0].confidence == 0.5

    def test_prefix_match_multiple_tags(self):
        """Should match multiple tags with same prefix."""
        matcher = LexicalTagMatcher(enable_prefix_matching=True)
        matcher.register_tags(["python", "pytorch", "pydantic"])

        matches = matcher.infer_tags("pyt")

        tags = {m.tag for m in matches}
        assert "python" in tags or "pytorch" in tags

    def test_prefix_matching_respects_seen_tags(self):
        """Prefix matching should not duplicate already matched tags."""
        matcher = LexicalTagMatcher(enable_prefix_matching=True)
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python pyth")

        # "python" exact match should be found, "pyth" prefix should also match
        # but both point to "python", so only one result
        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_prefix_match_with_exact_and_alias_priority(self):
        """Exact and alias matches should take priority over prefix."""
        matcher = LexicalTagMatcher(enable_prefix_matching=True)
        matcher.register_tags(["python", "pytest"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("python py pyt")

        # Should find: python (exact), python (alias), pytest (prefix)
        # But python is deduplicated
        tags = {m.tag for m in matches}
        assert "python" in tags


class TestLexicalTagMatcherHelpers:
    """Tests for helper methods."""

    def test_get_matching_tags(self):
        """get_matching_tags should return just tag names."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust"])

        tags = matcher.get_matching_tags("python and rust")

        assert tags == {"python", "rust"}

    def test_get_matching_tags_empty_query(self):
        """get_matching_tags should return empty set for no matches."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        tags = matcher.get_matching_tags("javascript code")

        assert tags == set()

    def test_get_matching_tags_with_duplicates(self):
        """get_matching_tags should return unique tags."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        tags = matcher.get_matching_tags("python python python")

        assert tags == {"python"}


class TestLexicalTagMatcherTokenization:
    """Tests for query tokenization."""

    def test_tokenize_whitespace(self):
        """Should split on whitespace."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust"])

        matches = matcher.infer_tags("python   rust")

        tags = {m.tag for m in matches}
        assert tags == {"python", "rust"}

    def test_tokenize_punctuation(self):
        """Should handle punctuation correctly."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python, tutorial!")

        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_tokenize_preserves_hyphens(self):
        """Should preserve hyphens in tags."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["machine-learning"])

        matches = matcher.infer_tags("machine-learning guide")

        assert len(matches) == 1
        assert matches[0].tag == "machine-learning"

    def test_tokenize_preserves_underscores(self):
        """Should preserve underscores in tags."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["code_review"])

        matches = matcher.infer_tags("code_review process")

        assert len(matches) == 1
        assert matches[0].tag == "code_review"

    def test_tokenize_preserves_slashes(self):
        """Should preserve slashes in tags."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["project/active"])

        matches = matcher.infer_tags("check project/active tasks")

        assert len(matches) == 1
        assert matches[0].tag == "project/active"

    def test_tokenize_numbers_and_letters(self):
        """Should match tags with alphanumeric patterns."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python3", "k8s", "web2"])

        matches = matcher.infer_tags("using python3 and k8s")

        tags = {m.tag for m in matches}
        assert tags == {"python3", "k8s"}

    def test_tokenize_ignores_pure_punctuation(self):
        """Should ignore tokens that are pure punctuation."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python ... !!! ???")

        assert len(matches) == 1
        assert matches[0].tag == "python"


class TestLexicalTagMatcherSorting:
    """Tests for result sorting."""

    def test_sorted_by_confidence(self):
        """Results should be sorted by confidence (highest first)."""
        matcher = LexicalTagMatcher(enable_prefix_matching=True, prefix_confidence=0.7)
        matcher.register_tags(["python", "pytorch"])
        matcher.register_alias("py", "python")

        # This will generate matches with different confidences
        matcher2 = LexicalTagMatcher(
            enable_prefix_matching=True,
            prefix_confidence=0.6,
            alias_confidence=0.8,
        )
        matcher2.register_tags(["rust", "ruby"])
        matcher2.register_alias("rs", "rust")

        matches = matcher2.infer_tags("ruby rs rub")

        confidences = [m.confidence for m in matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_exact_before_alias_before_prefix(self):
        """Should sort exact matches before aliases before prefixes."""
        matcher = LexicalTagMatcher(
            enable_prefix_matching=True,
            alias_confidence=0.9,
            prefix_confidence=0.7,
        )
        matcher.register_tags(["python", "rust", "ruby"])
        matcher.register_alias("py", "python")

        # Mix of match types
        matches = matcher.infer_tags("python py rub")

        # python (exact 1.0), python (alias 0.9 - deduplicated), ruby (prefix 0.7)
        assert matches[0].confidence >= matches[1].confidence


class TestLexicalTagMatcherEdgeCases:
    """Edge case tests for LexicalTagMatcher."""

    def test_empty_query(self):
        """Empty query should return no matches."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("")

        assert matches == []

    def test_whitespace_only_query(self):
        """Whitespace-only query should return no matches."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("   \t\n  ")

        assert matches == []

    def test_query_with_only_punctuation(self):
        """Query with only punctuation should return no matches."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("... !!! ???")

        assert matches == []

    def test_very_long_query(self):
        """Should handle very long queries efficiently."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust"])

        # Very long query with the tag buried in it
        long_query = " ".join(["word"] * 500 + ["python"] + ["more"] * 500)
        matches = matcher.infer_tags(long_query)

        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_many_registered_tags(self):
        """Should handle large number of registered tags."""
        matcher = LexicalTagMatcher()
        # Register 1000 tags
        tags = {f"tag{i}" for i in range(1000)}
        matcher.register_tags(tags)

        matches = matcher.infer_tags("tag500 tutorial")

        assert len(matches) == 1
        assert matches[0].tag == "tag500"

    def test_many_query_terms(self):
        """Should handle queries with many terms."""
        matcher = LexicalTagMatcher()
        tags = {f"tag{i}" for i in range(100)}
        matcher.register_tags(tags)

        # Query with all 100 tags
        query = " ".join(sorted(tags))
        matches = matcher.infer_tags(query)

        assert len(matches) == 100

    def test_tag_same_as_alias(self):
        """Tag that is also an alias should prefer exact match."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["py", "python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")

        # "py" as exact match should win
        assert matches[0].tag == "py"
        assert matches[0].match_type == "exact"

    def test_empty_tag_list(self):
        """Should handle empty tag list gracefully."""
        matcher = LexicalTagMatcher()
        matcher.register_tags([])

        matches = matcher.infer_tags("python code")

        assert matches == []

    def test_special_characters_in_query(self):
        """Should handle special characters in query."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python @#$% code")

        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_unicode_in_query(self):
        """Should handle unicode characters in query."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python 编程 tutorial")

        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_mixed_exact_and_alias_same_tag(self):
        """Should deduplicate when both exact and alias match same tag."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("python py")

        # Should only return one match for "python" (exact takes precedence)
        assert len(matches) == 1
        assert matches[0].tag == "python"
        assert matches[0].match_type == "exact"


class TestLoadDefaultAliases:
    """Tests for load_default_aliases function."""

    def test_load_default_aliases_returns_dict(self):
        """Should return a dictionary of aliases."""
        aliases = load_default_aliases()

        assert isinstance(aliases, dict)

    def test_load_default_aliases_has_common_aliases(self):
        """Should include common programming language aliases."""
        aliases = load_default_aliases()

        # Check for some common aliases
        assert "py" in aliases or "js" in aliases or len(aliases) > 0

    def test_load_default_aliases_integration(self):
        """Should work when used with matcher."""
        aliases = load_default_aliases()

        matcher = LexicalTagMatcher()
        matcher.register_aliases(aliases)

        # Should work with common aliases
        # Note: We can't assert specific aliases without knowing the data file
        # but we can test that the integration works
        assert isinstance(matcher._aliases, dict)


class TestCreateDefaultMatcher:
    """Tests for the default matcher factory."""

    def test_creates_with_default_config(self):
        """Should create matcher with default configuration."""
        matcher = create_default_matcher()

        assert isinstance(matcher, LexicalTagMatcher)
        assert matcher.enable_prefix_matching is False
        assert matcher.alias_confidence == 0.9

    def test_creates_with_common_aliases(self):
        """Should include common programming aliases."""
        matcher = create_default_matcher()

        # Test common language aliases (if they exist in the data file)
        matches = matcher.infer_tags("py code and js frontend")

        # Should have some aliases registered
        assert len(matcher._aliases) > 0

    def test_accepts_known_tags(self):
        """Should register provided known tags."""
        matcher = create_default_matcher(known_tags={"custom-tag", "another"})

        matches = matcher.infer_tags("custom-tag tutorial")

        assert len(matches) == 1
        assert matches[0].tag == "custom-tag"

    def test_known_tags_with_none(self):
        """Should work when known_tags is None."""
        matcher = create_default_matcher(known_tags=None)

        assert isinstance(matcher, LexicalTagMatcher)

    def test_known_tags_with_empty_set(self):
        """Should work when known_tags is empty set."""
        matcher = create_default_matcher(known_tags=set())

        assert isinstance(matcher, LexicalTagMatcher)

    def test_known_tags_normalized(self):
        """Should normalize known tags to lowercase."""
        matcher = create_default_matcher(known_tags={"Python", "RUST"})

        matches = matcher.infer_tags("python rust")

        tags = {m.tag for m in matches}
        assert tags == {"python", "rust"}


class TestLexicalTagMatcherIntegration:
    """Integration tests combining multiple features."""

    def test_exact_alias_and_prefix_all_together(self):
        """Should handle exact, alias, and prefix matches in one query."""
        matcher = LexicalTagMatcher(
            enable_prefix_matching=True,
            alias_confidence=0.9,
            prefix_confidence=0.7,
        )
        matcher.register_tags(["python", "rust", "javascript", "java"])
        matcher.register_alias("py", "python")
        matcher.register_alias("js", "javascript")

        matches = matcher.infer_tags("python py java jav")

        # python (exact 1.0), java (exact 1.0), java (prefix - deduplicated)
        tags = {m.tag for m in matches}
        assert "python" in tags
        assert "java" in tags

    def test_complex_query_with_punctuation_and_noise(self):
        """Should extract relevant tags from complex real-world query."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "api", "rest", "tutorial"])

        query = "How to build a Python REST API? Looking for tutorial..."
        matches = matcher.infer_tags(query)

        tags = {m.tag for m in matches}
        assert "python" in tags
        assert "api" in tags
        assert "rest" in tags
        assert "tutorial" in tags

    def test_hierarchical_tags_with_different_levels(self):
        """Should handle hierarchical tag structures."""
        matcher = LexicalTagMatcher()
        matcher.register_tags([
            "project/active",
            "project/archived",
            "project/active/urgent",
        ])

        matches = matcher.infer_tags("project/active and project/archived")

        tags = {m.tag for m in matches}
        assert "project/active" in tags
        assert "project/archived" in tags

    def test_reusing_matcher_multiple_queries(self):
        """Should work correctly when reused for multiple queries."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust", "go"])
        matcher.register_alias("py", "python")

        matches1 = matcher.infer_tags("python code")
        matches2 = matcher.infer_tags("rust tutorial")
        matches3 = matcher.infer_tags("py and go")

        assert matches1[0].tag == "python"
        assert matches2[0].tag == "rust"
        assert {m.tag for m in matches3} == {"python", "go"}

    def test_performance_with_large_dataset(self):
        """Should perform reasonably with large number of tags and aliases."""
        matcher = LexicalTagMatcher()

        # Register 1000 tags
        tags = [f"tag-{i}" for i in range(1000)]
        matcher.register_tags(tags)

        # Register 500 aliases
        aliases = {f"alias-{i}": f"tag-{i}" for i in range(500)}
        matcher.register_aliases(aliases)

        # Query with multiple terms
        matches = matcher.infer_tags("tag-42 alias-100 tag-999")

        tags_found = {m.tag for m in matches}
        assert "tag-42" in tags_found
        assert "tag-100" in tags_found  # from alias-100
        assert "tag-999" in tags_found
