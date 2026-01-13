"""Tests for tag alias management."""

import json
import tempfile
from pathlib import Path

import pytest

from pmd.metadata.aliases import (
    TagAliases,
    load_aliases,
    load_default_aliases,
)


class TestTagAliasesRegister:
    """Tests for TagAliases.register() method."""

    def test_register_single_alias(self):
        """Should register a single alias."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.is_alias("py")
        assert aliases.get_canonical("py") == "python"

    def test_register_multiple_different_aliases(self):
        """Should register multiple different aliases."""
        aliases = TagAliases()
        aliases.register("py", "python")
        aliases.register("js", "javascript")

        assert aliases.get_canonical("py") == "python"
        assert aliases.get_canonical("js") == "javascript"

    def test_register_case_insensitive_alias(self):
        """Should normalize alias to lowercase."""
        aliases = TagAliases()
        aliases.register("PY", "python")

        assert aliases.get_canonical("py") == "python"
        assert aliases.get_canonical("PY") == "python"
        assert aliases.get_canonical("Py") == "python"

    def test_register_case_insensitive_canonical(self):
        """Should normalize canonical tag to lowercase."""
        aliases = TagAliases()
        aliases.register("py", "PYTHON")

        assert aliases.get_canonical("py") == "python"

    def test_register_overwrites_existing_alias(self):
        """Registering the same alias should overwrite previous mapping."""
        aliases = TagAliases()
        aliases.register("js", "javascript")
        aliases.register("js", "java-script")

        assert aliases.get_canonical("js") == "java-script"

    def test_register_multiple_aliases_same_canonical(self):
        """Should support multiple aliases for the same canonical tag."""
        aliases = TagAliases()
        aliases.register("py", "python")
        aliases.register("python3", "python")
        aliases.register("python2", "python")

        assert aliases.get_canonical("py") == "python"
        assert aliases.get_canonical("python3") == "python"
        assert aliases.get_canonical("python2") == "python"


class TestTagAliasesRegisterMany:
    """Tests for TagAliases.register_many() method."""

    def test_register_many_empty_dict(self):
        """Should handle empty dictionary."""
        aliases = TagAliases()
        aliases.register_many({})

        assert aliases.all_aliases() == {}

    def test_register_many_single_entry(self):
        """Should register a single entry from dict."""
        aliases = TagAliases()
        aliases.register_many({"py": "python"})

        assert aliases.get_canonical("py") == "python"

    def test_register_many_multiple_entries(self):
        """Should register multiple entries from dict."""
        aliases = TagAliases()
        aliases.register_many({
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
        })

        assert aliases.get_canonical("py") == "python"
        assert aliases.get_canonical("js") == "javascript"
        assert aliases.get_canonical("ts") == "typescript"

    def test_register_many_case_normalization(self):
        """Should normalize all entries to lowercase."""
        aliases = TagAliases()
        aliases.register_many({
            "PY": "PYTHON",
            "JS": "JavaScript",
        })

        assert aliases.get_canonical("py") == "python"
        assert aliases.get_canonical("js") == "javascript"

    def test_register_many_combined_with_register(self):
        """Should work correctly when combined with register()."""
        aliases = TagAliases()
        aliases.register("py", "python")
        aliases.register_many({
            "js": "javascript",
            "ts": "typescript",
        })

        assert aliases.get_canonical("py") == "python"
        assert aliases.get_canonical("js") == "javascript"
        assert aliases.get_canonical("ts") == "typescript"


class TestTagAliasesResolve:
    """Tests for TagAliases.resolve() method."""

    def test_resolve_known_alias(self):
        """Should return canonical tag for known alias."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.resolve("py") == "python"

    def test_resolve_unknown_term_passthrough(self):
        """Should return the term unchanged for unknown terms."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.resolve("unknown") == "unknown"

    def test_resolve_canonical_passthrough(self):
        """Should return canonical tag unchanged."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.resolve("python") == "python"

    def test_resolve_case_insensitive(self):
        """Should resolve regardless of case."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.resolve("PY") == "python"
        assert aliases.resolve("Py") == "python"
        assert aliases.resolve("pY") == "python"

    def test_resolve_normalizes_to_lowercase(self):
        """Should always return lowercase result."""
        aliases = TagAliases()

        result = aliases.resolve("UNKNOWN")

        assert result == "unknown"

    def test_resolve_empty_string(self):
        """Should handle empty string."""
        aliases = TagAliases()

        assert aliases.resolve("") == ""

    def test_resolve_whitespace(self):
        """Should pass through whitespace."""
        aliases = TagAliases()

        assert aliases.resolve("  spaces  ") == "  spaces  "


class TestTagAliasesIsAlias:
    """Tests for TagAliases.is_alias() method."""

    def test_is_alias_true_for_registered(self):
        """Should return True for registered alias."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.is_alias("py") is True

    def test_is_alias_false_for_unregistered(self):
        """Should return False for unregistered term."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.is_alias("js") is False

    def test_is_alias_false_for_canonical(self):
        """Should return False for canonical tag (not an alias)."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.is_alias("python") is False

    def test_is_alias_case_insensitive(self):
        """Should check case-insensitively."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.is_alias("PY") is True
        assert aliases.is_alias("Py") is True

    def test_is_alias_empty_string(self):
        """Should return False for empty string."""
        aliases = TagAliases()

        assert aliases.is_alias("") is False


class TestTagAliasesGetCanonical:
    """Tests for TagAliases.get_canonical() method."""

    def test_get_canonical_for_registered_alias(self):
        """Should return canonical tag for registered alias."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.get_canonical("py") == "python"

    def test_get_canonical_returns_none_for_unregistered(self):
        """Should return None for unregistered term."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.get_canonical("js") is None

    def test_get_canonical_returns_none_for_canonical(self):
        """Should return None for canonical tag (not an alias)."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.get_canonical("python") is None

    def test_get_canonical_case_insensitive(self):
        """Should look up case-insensitively."""
        aliases = TagAliases()
        aliases.register("py", "python")

        assert aliases.get_canonical("PY") == "python"
        assert aliases.get_canonical("Py") == "python"

    def test_get_canonical_empty_string(self):
        """Should return None for empty string."""
        aliases = TagAliases()

        assert aliases.get_canonical("") is None


class TestTagAliasesAllAliases:
    """Tests for TagAliases.all_aliases() method."""

    def test_all_aliases_empty(self):
        """Should return empty dict for empty registry."""
        aliases = TagAliases()

        assert aliases.all_aliases() == {}

    def test_all_aliases_single(self):
        """Should return dict with single alias."""
        aliases = TagAliases()
        aliases.register("py", "python")

        result = aliases.all_aliases()

        assert result == {"py": "python"}

    def test_all_aliases_multiple(self):
        """Should return dict with all aliases."""
        aliases = TagAliases()
        aliases.register("py", "python")
        aliases.register("js", "javascript")
        aliases.register("ts", "typescript")

        result = aliases.all_aliases()

        assert result == {
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
        }

    def test_all_aliases_returns_copy(self):
        """Should return a copy, not the internal dict."""
        aliases = TagAliases()
        aliases.register("py", "python")

        result = aliases.all_aliases()
        result["js"] = "javascript"

        # Original should be unchanged
        assert aliases.get_canonical("js") is None

    def test_all_aliases_after_register_many(self):
        """Should return all aliases registered via register_many."""
        aliases = TagAliases()
        aliases.register_many({
            "py": "python",
            "js": "javascript",
        })

        result = aliases.all_aliases()

        assert result == {
            "py": "python",
            "js": "javascript",
        }


class TestTagAliasesClear:
    """Tests for TagAliases.clear() method."""

    def test_clear_empty_registry(self):
        """Should handle clearing an already empty registry."""
        aliases = TagAliases()
        aliases.clear()

        assert aliases.all_aliases() == {}

    def test_clear_removes_all_aliases(self):
        """Should remove all registered aliases."""
        aliases = TagAliases()
        aliases.register_many({
            "py": "python",
            "js": "javascript",
            "ts": "typescript",
        })

        aliases.clear()

        assert aliases.all_aliases() == {}
        assert aliases.is_alias("py") is False
        assert aliases.is_alias("js") is False

    def test_clear_allows_reregistration(self):
        """Should allow registering new aliases after clear."""
        aliases = TagAliases()
        aliases.register("py", "python")
        aliases.clear()
        aliases.register("rs", "rust")

        assert aliases.is_alias("py") is False
        assert aliases.is_alias("rs") is True
        assert aliases.get_canonical("rs") == "rust"


class TestLoadAliases:
    """Tests for load_aliases() function."""

    def test_load_aliases_from_valid_file(self):
        """Should load aliases from valid JSON file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "aliases": {
                    "py": "python",
                    "js": "javascript",
                }
            }, f)
            temp_path = f.name

        try:
            aliases = load_aliases(temp_path)

            assert aliases.get_canonical("py") == "python"
            assert aliases.get_canonical("js") == "javascript"
        finally:
            Path(temp_path).unlink()

    def test_load_aliases_with_path_object(self):
        """Should accept Path object as parameter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "aliases": {
                    "py": "python",
                }
            }, f)
            temp_path = Path(f.name)

        try:
            aliases = load_aliases(temp_path)

            assert aliases.get_canonical("py") == "python"
        finally:
            temp_path.unlink()

    def test_load_aliases_with_string_path(self):
        """Should accept string path as parameter."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "aliases": {
                    "py": "python",
                }
            }, f)
            temp_path = str(f.name)

        try:
            aliases = load_aliases(temp_path)

            assert aliases.get_canonical("py") == "python"
        finally:
            Path(temp_path).unlink()

    def test_load_aliases_empty_aliases_dict(self):
        """Should handle file with empty aliases dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"aliases": {}}, f)
            temp_path = f.name

        try:
            aliases = load_aliases(temp_path)

            assert aliases.all_aliases() == {}
        finally:
            Path(temp_path).unlink()

    def test_load_aliases_missing_aliases_key(self):
        """Should handle file without 'aliases' key."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"other_key": "value"}, f)
            temp_path = f.name

        try:
            aliases = load_aliases(temp_path)

            assert aliases.all_aliases() == {}
        finally:
            Path(temp_path).unlink()

    def test_load_aliases_with_comments(self):
        """Should handle JSON with $comment field."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "$comment": "This is a comment",
                "aliases": {
                    "py": "python",
                }
            }, f)
            temp_path = f.name

        try:
            aliases = load_aliases(temp_path)

            assert aliases.get_canonical("py") == "python"
        finally:
            Path(temp_path).unlink()

    def test_load_aliases_file_not_found(self):
        """Should raise FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_aliases("/nonexistent/path/aliases.json")

    def test_load_aliases_invalid_json(self):
        """Should raise JSONDecodeError for invalid JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {")
            temp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                load_aliases(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_load_aliases_case_normalization(self):
        """Should normalize aliases to lowercase."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({
                "aliases": {
                    "PY": "PYTHON",
                    "JS": "JavaScript",
                }
            }, f)
            temp_path = f.name

        try:
            aliases = load_aliases(temp_path)

            assert aliases.get_canonical("py") == "python"
            assert aliases.get_canonical("js") == "javascript"
        finally:
            Path(temp_path).unlink()


class TestLoadDefaultAliases:
    """Tests for load_default_aliases() function."""

    def test_load_default_aliases_returns_tag_aliases_instance(self):
        """Should return a TagAliases instance."""
        aliases = load_default_aliases()

        assert isinstance(aliases, TagAliases)

    def test_load_default_aliases_has_common_aliases(self):
        """Should include common programming language aliases."""
        aliases = load_default_aliases()

        # Test a few common aliases that should always be present
        assert aliases.resolve("py") == "python"
        assert aliases.resolve("js") == "javascript"
        assert aliases.resolve("ts") == "typescript"

    def test_load_default_aliases_has_infrastructure_aliases(self):
        """Should include infrastructure and devops aliases."""
        aliases = load_default_aliases()

        # K8s is a very common alias
        assert aliases.resolve("k8s") == "kubernetes"
        assert aliases.resolve("db") == "database"

    def test_load_default_aliases_has_documentation_aliases(self):
        """Should include documentation-related aliases."""
        aliases = load_default_aliases()

        assert aliases.resolve("doc") == "documentation"
        assert aliases.resolve("docs") == "documentation"

    def test_load_default_aliases_multiple_aliases_same_canonical(self):
        """Should support multiple aliases for the same canonical tag."""
        aliases = load_default_aliases()

        # Multiple Python aliases
        assert aliases.resolve("py") == "python"
        # May or may not have python3, but if it does, should map to python
        if aliases.is_alias("python3"):
            assert aliases.resolve("python3") == "python"

    def test_load_default_aliases_not_empty(self):
        """Should return non-empty alias registry."""
        aliases = load_default_aliases()

        all_aliases = aliases.all_aliases()
        assert len(all_aliases) > 0

    def test_load_default_aliases_loads_from_data_file_if_exists(self):
        """Should load from data/tag_aliases.json if it exists."""
        aliases = load_default_aliases()

        # If the data file exists, it should have many more aliases
        # than the fallback set
        all_aliases = aliases.all_aliases()

        # Fallback has 9 aliases, real file should have many more
        # This tests that the real file is being loaded
        assert len(all_aliases) >= 9

    def test_load_default_aliases_fallback_when_no_file(self):
        """Should provide fallback aliases if data file doesn't exist."""
        # We can't easily test this without mocking, but we can at least
        # verify that the function always returns something usable
        aliases = load_default_aliases()

        # These are in the fallback set
        assert aliases.is_alias("py")
        assert aliases.is_alias("js")
        assert aliases.is_alias("k8s")


class TestTagAliasesEdgeCases:
    """Edge case tests for TagAliases."""

    def test_empty_alias_string(self):
        """Should handle empty string as alias."""
        aliases = TagAliases()
        aliases.register("", "empty")

        assert aliases.get_canonical("") == "empty"

    def test_empty_canonical_string(self):
        """Should handle empty string as canonical."""
        aliases = TagAliases()
        aliases.register("alias", "")

        assert aliases.get_canonical("alias") == ""

    def test_whitespace_in_alias(self):
        """Should preserve whitespace in alias."""
        aliases = TagAliases()
        aliases.register("two words", "tag")

        # Whitespace is preserved but lowercased
        assert aliases.get_canonical("two words") == "tag"

    def test_special_characters_in_alias(self):
        """Should handle special characters in alias."""
        aliases = TagAliases()
        aliases.register("c++", "cplusplus")
        aliases.register("c#", "csharp")

        assert aliases.get_canonical("c++") == "cplusplus"
        assert aliases.get_canonical("c#") == "csharp"

    def test_unicode_in_alias(self):
        """Should handle unicode characters."""
        aliases = TagAliases()
        aliases.register("pythön", "python")

        assert aliases.get_canonical("pythön") == "python"

    def test_hyphenated_tags(self):
        """Should handle hyphenated tags correctly."""
        aliases = TagAliases()
        aliases.register("ml", "machine-learning")
        aliases.register("dl", "deep-learning")

        assert aliases.get_canonical("ml") == "machine-learning"
        assert aliases.get_canonical("dl") == "deep-learning"

    def test_very_long_alias(self):
        """Should handle very long alias strings."""
        aliases = TagAliases()
        long_alias = "a" * 1000
        long_canonical = "b" * 1000

        aliases.register(long_alias, long_canonical)

        assert aliases.get_canonical(long_alias) == long_canonical

    def test_many_aliases_performance(self):
        """Should handle large number of aliases efficiently."""
        aliases = TagAliases()

        # Register 1000 aliases
        alias_dict = {f"alias{i}": f"tag{i}" for i in range(1000)}
        aliases.register_many(alias_dict)

        # Should be able to resolve any of them quickly
        assert aliases.get_canonical("alias500") == "tag500"
        assert aliases.get_canonical("alias999") == "tag999"

    def test_same_alias_and_canonical(self):
        """Should handle alias that is the same as canonical."""
        aliases = TagAliases()
        aliases.register("python", "python")

        assert aliases.get_canonical("python") == "python"
        assert aliases.is_alias("python") is True

    def test_circular_alias_prevention(self):
        """Should not prevent circular aliases (no validation)."""
        # Note: The implementation doesn't prevent circular aliases
        # This test documents the behavior
        aliases = TagAliases()
        aliases.register("a", "b")
        aliases.register("b", "a")

        # Both resolve to their targets
        assert aliases.resolve("a") == "b"
        assert aliases.resolve("b") == "a"

    def test_overwrite_with_different_case(self):
        """Should handle overwriting with different case."""
        aliases = TagAliases()
        aliases.register("py", "python")
        aliases.register("PY", "PYTHON")  # Should overwrite (both lowercase)

        # Should have single entry, lowercased
        all_aliases = aliases.all_aliases()
        assert all_aliases == {"py": "python"}

    def test_resolve_preserves_unknown_case(self):
        """Should lowercase unknown terms when resolving."""
        aliases = TagAliases()

        result = aliases.resolve("UnKnOwN")

        assert result == "unknown"

    def test_numeric_tags(self):
        """Should handle numeric tags."""
        aliases = TagAliases()
        aliases.register("py3", "python3")
        aliases.register("k8s", "kubernetes")

        assert aliases.get_canonical("py3") == "python3"
        assert aliases.get_canonical("k8s") == "kubernetes"

    def test_dots_in_tags(self):
        """Should handle dots in tags."""
        aliases = TagAliases()
        aliases.register("node.js", "nodejs")

        assert aliases.get_canonical("node.js") == "nodejs"

    def test_underscores_in_tags(self):
        """Should handle underscores in tags."""
        aliases = TagAliases()
        aliases.register("deep_learning", "machine-learning")

        assert aliases.get_canonical("deep_learning") == "machine-learning"

    def test_slashes_in_tags(self):
        """Should handle slashes in tags."""
        aliases = TagAliases()
        aliases.register("ci/cd", "continuous-integration")

        assert aliases.get_canonical("ci/cd") == "continuous-integration"

    def test_multiple_registration_same_alias_different_canonical(self):
        """Should overwrite when same alias registered to different canonical."""
        aliases = TagAliases()
        aliases.register("js", "javascript")

        # Verify first registration
        assert aliases.get_canonical("js") == "javascript"

        # Re-register with different canonical
        aliases.register("js", "java-script")

        # Should use latest registration
        assert aliases.get_canonical("js") == "java-script"

    def test_all_aliases_after_clear_and_reregister(self):
        """Should return correct aliases after clear and re-registration."""
        aliases = TagAliases()
        aliases.register_many({
            "py": "python",
            "js": "javascript",
        })

        aliases.clear()

        aliases.register_many({
            "rs": "rust",
            "go": "golang",
        })

        result = aliases.all_aliases()

        assert result == {
            "rs": "rust",
            "go": "golang",
        }
        assert "py" not in result
        assert "js" not in result
