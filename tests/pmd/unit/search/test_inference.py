"""Tests for query-time tag inference."""

import pytest

from pmd.metadata import (
    LexicalTagMatcher,
    TagMatch,
    create_default_matcher,
)


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

    def test_exact_match_multiple_terms(self):
        """Should match multiple tags in query."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "web", "api"])

        matches = matcher.infer_tags("python web api development")

        tags = {m.tag for m in matches}
        assert tags == {"python", "web", "api"}

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

    def test_register_aliases_bulk(self):
        """Should register multiple aliases at once."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "javascript"])
        matcher.register_aliases({
            "py": "python",
            "js": "javascript",
        })

        matches = matcher.infer_tags("py and js")

        tags = {m.tag for m in matches}
        assert tags == {"python", "javascript"}

    def test_exact_match_priority_over_alias(self):
        """Exact match should be returned with higher confidence."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "py"])  # 'py' is also a tag
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")

        # Should match "py" exactly, not as alias
        assert len(matches) == 1
        assert matches[0].tag == "py"
        assert matches[0].confidence == 1.0


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
        """Prefix matches should have lower confidence."""
        matcher = LexicalTagMatcher(
            enable_prefix_matching=True,
            prefix_confidence=0.7,
        )
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("pyth")

        assert matches[0].confidence == 0.7
        assert matches[0].match_type == "prefix"


class TestLexicalTagMatcherHelpers:
    """Tests for helper methods."""

    def test_get_matching_tags(self):
        """get_matching_tags should return just tag names."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python", "rust"])

        tags = matcher.get_matching_tags("python and rust")

        assert tags == {"python", "rust"}

    def test_clear_removes_all(self):
        """clear should remove all tags and aliases."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matcher.clear()

        assert matcher.infer_tags("python py") == []


class TestCreateDefaultMatcher:
    """Tests for the default matcher factory."""

    def test_creates_with_common_aliases(self):
        """Should include common programming aliases."""
        matcher = create_default_matcher()

        # Test common language aliases
        matches = matcher.infer_tags("py code and js frontend")

        tags = {m.tag for m in matches}
        assert "python" in tags
        assert "javascript" in tags

    def test_accepts_known_tags(self):
        """Should register provided known tags."""
        matcher = create_default_matcher(known_tags={"custom-tag", "another"})

        matches = matcher.infer_tags("custom-tag tutorial")

        assert len(matches) == 1
        assert matches[0].tag == "custom-tag"


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

    def test_unicode_in_tags(self):
        """Should handle unicode characters in tags."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])  # Only ASCII tags

        matches = matcher.infer_tags("python tutorial")

        assert len(matches) == 1

    def test_hyphenated_tags(self):
        """Should match hyphenated tags correctly."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["machine-learning", "deep-learning"])

        matches = matcher.infer_tags("machine-learning guide")

        assert len(matches) == 1
        assert matches[0].tag == "machine-learning"

    def test_underscored_tags(self):
        """Should match underscored tags correctly."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["code_review", "pull_request"])

        matches = matcher.infer_tags("code_review process")

        assert len(matches) == 1
        assert matches[0].tag == "code_review"

    def test_hierarchical_tags_with_slash(self):
        """Should match hierarchical tags with slashes."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["project/active", "project/archived"])

        matches = matcher.infer_tags("check project/active tasks")

        assert len(matches) == 1
        assert matches[0].tag == "project/active"

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

    def test_tag_same_as_alias(self):
        """Tag that is also an alias should prefer exact match."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["py", "python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")

        # "py" as exact match should win
        assert matches[0].tag == "py"
        assert matches[0].match_type == "exact"

    def test_custom_alias_confidence(self):
        """Should use custom alias confidence."""
        matcher = LexicalTagMatcher(alias_confidence=0.75)
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("py code")

        assert matches[0].confidence == 0.75

    def test_sorted_by_confidence(self):
        """Results should be sorted by confidence (highest first)."""
        matcher = LexicalTagMatcher(enable_prefix_matching=True, prefix_confidence=0.7)
        matcher.register_tags(["python", "pytorch"])
        matcher.register_alias("py", "python")

        matches = matcher.infer_tags("python py pytorch")

        confidences = [m.confidence for m in matches]
        assert confidences == sorted(confidences, reverse=True)

    def test_register_tags_multiple_times(self):
        """Registering the same tag multiple times should not create duplicates."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_tags(["python", "rust"])
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python tutorial")

        assert len(matches) == 1

    def test_query_with_numbers_and_letters(self):
        """Should match tags with alphanumeric patterns."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python3", "k8s", "web2"])

        matches = matcher.infer_tags("using python3 and k8s")

        tags = {m.tag for m in matches}
        assert tags == {"python3", "k8s"}

    def test_prefix_matching_respects_seen_tags(self):
        """Prefix matching should not duplicate already matched tags."""
        matcher = LexicalTagMatcher(enable_prefix_matching=True)
        matcher.register_tags(["python"])

        matches = matcher.infer_tags("python pyth")

        # "python" exact match should be found, "pyth" prefix should also match
        # but both point to "python", so only one result
        assert len(matches) == 1
        assert matches[0].tag == "python"

    def test_clear_and_reregister(self):
        """Should work correctly after clear and re-register."""
        matcher = LexicalTagMatcher()
        matcher.register_tags(["python"])
        matcher.register_alias("py", "python")

        matcher.clear()
        matcher.register_tags(["rust", "go"])  # Two tags
        matcher.register_alias("rs", "rust")

        matches_old = matcher.infer_tags("python py")
        matches_new = matcher.infer_tags("rust go")  # Two different tags

        assert matches_old == []
        assert len(matches_new) == 2
        tags = {m.tag for m in matches_new}
        assert tags == {"rust", "go"}


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
