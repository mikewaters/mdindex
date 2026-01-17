"""Tests for metadata profile registry."""

import pytest

from pmd.metadata import (
    MetadataProfileRegistry,
    get_default_profile_registry,
    GenericProfile,
    ObsidianProfile,
    DraftsProfile,
)
from pmd.extraction.registry import ProfileRegistration


class MockProfile:
    """Mock profile for testing."""

    def __init__(self, name: str):
        self.name = name

    def extract_metadata(self, content: str, path: str):
        pass

    def normalize_tag(self, tag: str) -> str:
        return tag.lower()


class TestProfileRegistration:
    """Tests for the ProfileRegistration dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        profile = MockProfile("test")
        reg = ProfileRegistration(profile=profile)

        assert reg.profile == profile
        assert reg.path_patterns == []
        assert reg.detectors == []
        assert reg.priority == 0

    def test_with_values(self):
        """Should accept all values."""
        import re

        profile = MockProfile("test")
        pattern = re.compile(r"test")
        detector = lambda c, p: True

        reg = ProfileRegistration(
            profile=profile,
            path_patterns=[pattern],
            detectors=[detector],
            priority=10,
        )

        assert reg.priority == 10
        assert len(reg.path_patterns) == 1
        assert len(reg.detectors) == 1


class TestMetadataProfileRegistry:
    """Tests for the MetadataProfileRegistry class."""

    def test_register_profile(self):
        """Should register a profile by name."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("test")

        registry.register(profile)

        assert registry.is_registered("test")
        assert registry.get("test") == profile

    def test_register_with_path_patterns(self):
        """Should register profile with path patterns."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("vault")

        registry.register(
            profile,
            path_patterns=[r"vault/.*", r"\.obsidian"],
        )

        assert registry.is_registered("vault")

    def test_register_duplicate_raises(self):
        """Should raise if registering duplicate without override."""
        registry = MetadataProfileRegistry()
        profile1 = MockProfile("test")
        profile2 = MockProfile("test")

        registry.register(profile1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(profile2)

    def test_register_duplicate_with_override(self):
        """Should allow override with override=True."""
        registry = MetadataProfileRegistry()
        profile1 = MockProfile("test")
        profile2 = MockProfile("test")

        registry.register(profile1)
        registry.register(profile2, override=True)

        # Should now have the second profile
        assert registry.get("test") == profile2

    def test_unregister_profile(self):
        """Should unregister a profile."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("test")

        registry.register(profile)
        result = registry.unregister("test")

        assert result is True
        assert registry.get("test") is None
        assert not registry.is_registered("test")

    def test_unregister_nonexistent(self):
        """Should return False for non-existent profile."""
        registry = MetadataProfileRegistry()

        result = registry.unregister("nonexistent")

        assert result is False

    def test_get_nonexistent(self):
        """Should return None for non-existent profile."""
        registry = MetadataProfileRegistry()

        result = registry.get("nonexistent")

        assert result is None

    def test_list_profiles(self):
        """Should list all registered profiles."""
        registry = MetadataProfileRegistry()
        registry.register(MockProfile("alpha"))
        registry.register(MockProfile("beta"))
        registry.register(MockProfile("gamma"))

        profiles = registry.list_profiles()

        assert profiles == ["alpha", "beta", "gamma"]

    def test_list_profiles_empty(self):
        """Should return empty list for empty registry."""
        registry = MetadataProfileRegistry()

        assert registry.list_profiles() == []


class TestMetadataProfileRegistryDetection:
    """Tests for profile auto-detection."""

    def test_detect_by_path_pattern(self):
        """Should detect profile by path pattern."""
        registry = MetadataProfileRegistry()
        vault_profile = MockProfile("vault")

        registry.register(vault_profile, path_patterns=[r"vault/"])

        detected = registry.detect("content", "vault/notes/test.md")

        assert detected == vault_profile

    def test_detect_by_path_case_insensitive(self):
        """Path patterns should be case-insensitive."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("test")

        registry.register(profile, path_patterns=[r"vault"])

        detected = registry.detect("content", "VAULT/test.md")

        assert detected == profile

    def test_detect_by_detector_function(self):
        """Should detect profile by detector function."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("special")

        def detector(content: str, path: str) -> bool:
            return "MAGIC_STRING" in content

        registry.register(profile, detectors=[detector])

        detected = registry.detect("Contains MAGIC_STRING here", "test.md")

        assert detected == profile

    def test_detect_no_match(self):
        """Should return None when no profile matches."""
        registry = MetadataProfileRegistry()
        registry.register(MockProfile("test"), path_patterns=[r"specific/"])

        detected = registry.detect("content", "other/path/test.md")

        assert detected is None

    def test_detect_priority_order(self):
        """Higher priority profiles should be checked first."""
        registry = MetadataProfileRegistry()
        low_priority = MockProfile("low")
        high_priority = MockProfile("high")

        # Both match the same pattern
        registry.register(low_priority, path_patterns=[r"test"], priority=0)
        registry.register(high_priority, path_patterns=[r"test"], priority=10)

        detected = registry.detect("content", "test/file.md")

        # High priority should be selected
        assert detected == high_priority

    def test_detect_path_before_detector(self):
        """Path patterns should be checked before detectors."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("test")

        called_detector = []

        def detector(content: str, path: str) -> bool:
            called_detector.append(True)
            return True

        registry.register(
            profile,
            path_patterns=[r"match/"],
            detectors=[detector],
        )

        # Path matches, detector should not be called
        registry.detect("content", "match/test.md")

        assert len(called_detector) == 0

    def test_detect_detector_exception_handled(self):
        """Should handle exceptions in detector functions."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("test")
        fallback = MockProfile("fallback")

        def bad_detector(content: str, path: str) -> bool:
            raise RuntimeError("Detector failed")

        registry.register(profile, detectors=[bad_detector], priority=10)
        registry.register(fallback, path_patterns=[r".*"], priority=0)

        # Should not raise, should fall back
        detected = registry.detect("content", "test.md")

        assert detected == fallback


class TestMetadataProfileRegistryDetectOrDefault:
    """Tests for detect_or_default method."""

    def test_detect_or_default_when_detected(self):
        """Should return detected profile when match found."""
        registry = MetadataProfileRegistry()
        profile = MockProfile("match")
        default = MockProfile("default")

        registry.register(profile, path_patterns=[r"match/"])
        registry.register(default)

        result = registry.detect_or_default("content", "match/test.md", "default")

        assert result == profile

    def test_detect_or_default_fallback(self):
        """Should return default when no match found."""
        registry = MetadataProfileRegistry()
        default = MockProfile("default")

        registry.register(default)

        result = registry.detect_or_default("content", "nomatch.md", "default")

        assert result == default

    def test_detect_or_default_missing_default_raises(self):
        """Should raise when default profile not registered."""
        registry = MetadataProfileRegistry()

        with pytest.raises(ValueError, match="not registered"):
            registry.detect_or_default("content", "test.md", "nonexistent")


class TestGetDefaultProfileRegistry:
    """Tests for the default profile registry."""

    def test_returns_registry(self):
        """Should return a MetadataProfileRegistry instance."""
        registry = get_default_profile_registry()

        assert isinstance(registry, MetadataProfileRegistry)

    def test_has_generic_profile(self):
        """Default registry should have generic profile."""
        registry = get_default_profile_registry()

        assert registry.is_registered("generic")
        assert isinstance(registry.get("generic"), GenericProfile)

    def test_has_obsidian_profile(self):
        """Default registry should have obsidian profile."""
        registry = get_default_profile_registry()

        assert registry.is_registered("obsidian")
        assert isinstance(registry.get("obsidian"), ObsidianProfile)

    def test_has_drafts_profile(self):
        """Default registry should have drafts profile."""
        registry = get_default_profile_registry()

        assert registry.is_registered("drafts")
        assert isinstance(registry.get("drafts"), DraftsProfile)

    def test_detects_obsidian_by_path(self):
        """Should detect Obsidian by vault path."""
        registry = get_default_profile_registry()

        profile = registry.detect("content", "vault/notes/test.md")

        assert profile is not None
        assert profile.name == "obsidian"

    def test_detects_obsidian_by_content(self):
        """Should detect Obsidian by wikilinks in content."""
        registry = get_default_profile_registry()
        content = "This has [[wikilink]] syntax."

        profile = registry.detect(content, "random/path.md")

        assert profile is not None
        assert profile.name == "obsidian"

    def test_detects_drafts_by_path(self):
        """Should detect Drafts by path."""
        registry = get_default_profile_registry()

        profile = registry.detect("content", "drafts/note.md")

        assert profile is not None
        assert profile.name == "drafts"

    def test_detects_drafts_by_content(self):
        """Should detect Drafts by UUID in content."""
        registry = get_default_profile_registry()
        content = """---
uuid: 12345678-1234-1234-1234-123456789abc
---
Content here.
"""
        profile = registry.detect(content, "random/path.md")

        assert profile is not None
        assert profile.name == "drafts"

    def test_generic_as_fallback(self):
        """Generic should be detected for unmatched content."""
        registry = get_default_profile_registry()
        content = "Just plain markdown with no special features."

        # detect returns None, but detect_or_default should fall back
        profile = registry.detect_or_default(content, "plain.md", "generic")

        assert profile.name == "generic"

    def test_singleton_behavior(self):
        """Should return the same registry instance."""
        registry1 = get_default_profile_registry()
        registry2 = get_default_profile_registry()

        assert registry1 is registry2
