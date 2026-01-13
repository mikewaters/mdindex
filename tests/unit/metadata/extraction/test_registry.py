"""Tests for metadata profile registry with auto-detection."""

import pytest
import re
from unittest.mock import MagicMock

from pmd.metadata.extraction.registry import (
    MetadataProfileRegistry,
    ProfileRegistration,
    get_default_profile_registry,
    _register_builtin_profiles,
)
from pmd.metadata import GenericProfile, ExtractedMetadata


@pytest.fixture
def registry():
    """Create a fresh empty registry for testing."""
    return MetadataProfileRegistry()


@pytest.fixture
def mock_profile():
    """Create a mock profile for testing."""
    profile = MagicMock()
    profile.name = "test-profile"
    profile.extract_metadata.return_value = ExtractedMetadata(
        tags={"test"},
        source_tags=["test"],
        attributes={},
        extraction_source="test-profile",
    )
    return profile


@pytest.fixture
def generic_profile():
    """Create a real GenericProfile instance."""
    return GenericProfile()


class TestProfileRegistration:
    """Tests for ProfileRegistration dataclass."""

    def test_profile_registration_creation(self, mock_profile):
        """Should create ProfileRegistration with all fields."""
        pattern = re.compile(r"\.test", re.IGNORECASE)
        detector = lambda c, p: True

        reg = ProfileRegistration(
            profile=mock_profile,
            path_patterns=[pattern],
            detectors=[detector],
            priority=10,
        )

        assert reg.profile is mock_profile
        assert reg.path_patterns == [pattern]
        assert reg.detectors == [detector]
        assert reg.priority == 10

    def test_profile_registration_defaults(self, mock_profile):
        """Should use default values for optional fields."""
        reg = ProfileRegistration(profile=mock_profile)

        assert reg.profile is mock_profile
        assert reg.path_patterns == []
        assert reg.detectors == []
        assert reg.priority == 0


class TestMetadataProfileRegistryRegister:
    """Tests for registering profiles."""

    def test_register_profile_by_name(self, registry, mock_profile):
        """Should register profile by its name attribute."""
        registry.register(mock_profile)

        assert registry.is_registered("test-profile")
        assert registry.get("test-profile") is mock_profile

    def test_register_with_path_patterns(self, registry, mock_profile):
        """Should register profile with path patterns."""
        registry.register(
            mock_profile,
            path_patterns=[r"\.obsidian", r"vault"],
        )

        reg = registry._profiles["test-profile"]
        assert len(reg.path_patterns) == 2
        assert all(isinstance(p, re.Pattern) for p in reg.path_patterns)

    def test_register_with_detectors(self, registry, mock_profile):
        """Should register profile with detector functions."""
        detector = lambda c, p: True
        registry.register(mock_profile, detectors=[detector])

        reg = registry._profiles["test-profile"]
        assert len(reg.detectors) == 1
        assert reg.detectors[0] is detector

    def test_register_with_priority(self, registry, mock_profile):
        """Should register profile with custom priority."""
        registry.register(mock_profile, priority=50)

        reg = registry._profiles["test-profile"]
        assert reg.priority == 50

    def test_register_duplicate_raises_error(self, registry, mock_profile):
        """Should raise ValueError when registering duplicate profile."""
        registry.register(mock_profile)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(mock_profile)

    def test_register_duplicate_with_override(self, registry, mock_profile):
        """Should allow duplicate registration with override=True."""
        new_profile = MagicMock()
        new_profile.name = "test-profile"

        registry.register(mock_profile, priority=10)
        registry.register(new_profile, priority=20, override=True)

        # Should have the new profile
        assert registry.get("test-profile") is new_profile
        assert registry._profiles["test-profile"].priority == 20

    def test_register_compiles_patterns_case_insensitive(self, registry, mock_profile):
        """Should compile path patterns as case insensitive."""
        registry.register(mock_profile, path_patterns=[r"\.OBSIDIAN"])

        reg = registry._profiles["test-profile"]
        pattern = reg.path_patterns[0]

        assert pattern.search("/vault/.obsidian/config")
        assert pattern.search("/vault/.OBSIDIAN/config")
        assert pattern.search("/vault/.Obsidian/config")

    def test_register_with_no_path_patterns(self, registry, mock_profile):
        """Should handle None path_patterns."""
        registry.register(mock_profile, path_patterns=None)

        reg = registry._profiles["test-profile"]
        assert reg.path_patterns == []

    def test_register_with_no_detectors(self, registry, mock_profile):
        """Should handle None detectors."""
        registry.register(mock_profile, detectors=None)

        reg = registry._profiles["test-profile"]
        assert reg.detectors == []


class TestMetadataProfileRegistryUnregister:
    """Tests for unregistering profiles."""

    def test_unregister_existing_profile(self, registry, mock_profile):
        """Should remove registered profile."""
        registry.register(mock_profile)

        result = registry.unregister("test-profile")

        assert result is True
        assert not registry.is_registered("test-profile")

    def test_unregister_nonexistent_profile(self, registry):
        """Should return False for nonexistent profile."""
        result = registry.unregister("nonexistent")

        assert result is False

    def test_unregister_then_register_again(self, registry, mock_profile):
        """Should allow re-registration after unregister."""
        registry.register(mock_profile)
        registry.unregister("test-profile")
        registry.register(mock_profile)

        assert registry.is_registered("test-profile")


class TestMetadataProfileRegistryGet:
    """Tests for getting profiles by name."""

    def test_get_registered_profile(self, registry, mock_profile):
        """Should return registered profile."""
        registry.register(mock_profile)

        result = registry.get("test-profile")

        assert result is mock_profile

    def test_get_nonexistent_profile(self, registry):
        """Should return None for nonexistent profile."""
        result = registry.get("nonexistent")

        assert result is None

    def test_get_returns_profile_not_registration(self, registry, mock_profile):
        """Should return profile instance, not registration object."""
        registry.register(mock_profile, priority=10)

        result = registry.get("test-profile")

        assert result is mock_profile
        assert not isinstance(result, ProfileRegistration)


class TestMetadataProfileRegistryDetect:
    """Tests for auto-detecting profiles."""

    def test_detect_by_path_pattern(self, registry, mock_profile):
        """Should detect profile based on path pattern match."""
        registry.register(mock_profile, path_patterns=[r"\.obsidian"])

        result = registry.detect("content", "/vault/.obsidian/note.md")

        assert result is mock_profile

    def test_detect_by_multiple_path_patterns(self, registry, mock_profile):
        """Should detect when any path pattern matches."""
        registry.register(
            mock_profile,
            path_patterns=[r"\.obsidian", r"vault"],
        )

        result1 = registry.detect("content", "/docs/.obsidian/note.md")
        result2 = registry.detect("content", "/my/vault/note.md")

        assert result1 is mock_profile
        assert result2 is mock_profile

    def test_detect_by_content_detector(self, registry, mock_profile):
        """Should detect profile based on content detector."""
        detector = lambda content, path: "obsidian" in content
        registry.register(mock_profile, detectors=[detector])

        result = registry.detect("obsidian content", "/path/note.md")

        assert result is mock_profile

    def test_detect_with_multiple_detectors(self, registry, mock_profile):
        """Should detect when any detector returns True."""
        detector1 = lambda c, p: "pattern1" in c
        detector2 = lambda c, p: "pattern2" in c
        registry.register(mock_profile, detectors=[detector1, detector2])

        result1 = registry.detect("has pattern1", "/path/note.md")
        result2 = registry.detect("has pattern2", "/path/note.md")

        assert result1 is mock_profile
        assert result2 is mock_profile

    def test_detect_checks_path_before_content(self, registry, mock_profile):
        """Should check path patterns before content detectors."""
        detector_called = []
        detector = lambda c, p: detector_called.append(True) or True
        registry.register(
            mock_profile,
            path_patterns=[r"\.obsidian"],
            detectors=[detector],
        )

        result = registry.detect("content", "/vault/.obsidian/note.md")

        assert result is mock_profile
        assert len(detector_called) == 0  # Detector should not be called

    def test_detect_respects_priority_order(self, registry):
        """Should check higher priority profiles first."""
        high_priority = MagicMock()
        high_priority.name = "high-priority"
        low_priority = MagicMock()
        low_priority.name = "low-priority"

        registry.register(low_priority, path_patterns=[r"\.md"], priority=1)
        registry.register(high_priority, path_patterns=[r"\.md"], priority=10)

        result = registry.detect("content", "/path/note.md")

        assert result is high_priority

    def test_detect_returns_none_when_no_match(self, registry, mock_profile):
        """Should return None when no profile matches."""
        registry.register(mock_profile, path_patterns=[r"\.obsidian"])

        result = registry.detect("content", "/other/path/note.md")

        assert result is None

    def test_detect_handles_detector_exceptions(self, registry, mock_profile):
        """Should skip detectors that raise exceptions."""
        failing_detector = lambda c, p: 1 / 0  # Raises ZeroDivisionError
        passing_detector = lambda c, p: True

        registry.register(
            mock_profile,
            detectors=[failing_detector, passing_detector],
        )

        result = registry.detect("content", "/path/note.md")

        assert result is mock_profile

    def test_detect_continues_after_detector_failure(self, registry):
        """Should continue checking other profiles after detector fails."""
        failing_profile = MagicMock()
        failing_profile.name = "failing"
        passing_profile = MagicMock()
        passing_profile.name = "passing"

        failing_detector = lambda c, p: 1 / 0
        passing_detector = lambda c, p: True

        registry.register(failing_profile, detectors=[failing_detector], priority=10)
        registry.register(passing_profile, detectors=[passing_detector], priority=5)

        result = registry.detect("content", "/path/note.md")

        assert result is passing_profile

    def test_detect_with_empty_registry(self, registry):
        """Should return None with empty registry."""
        result = registry.detect("content", "/path/note.md")

        assert result is None

    def test_detect_case_insensitive_path_matching(self, registry, mock_profile):
        """Should match paths case-insensitively."""
        registry.register(mock_profile, path_patterns=[r"\.OBSIDIAN"])

        result = registry.detect("content", "/vault/.obsidian/note.md")

        assert result is mock_profile


class TestMetadataProfileRegistryDetectOrDefault:
    """Tests for auto-detecting with fallback to default."""

    def test_detect_or_default_returns_detected_profile(self, registry, mock_profile, generic_profile):
        """Should return detected profile when match found."""
        registry.register(generic_profile)
        registry.register(mock_profile, path_patterns=[r"\.obsidian"])

        result = registry.detect_or_default(
            "content",
            "/vault/.obsidian/note.md",
            default_name="generic",
        )

        assert result is mock_profile

    def test_detect_or_default_returns_default_when_no_match(self, registry, generic_profile):
        """Should return default profile when no match found."""
        registry.register(generic_profile)

        result = registry.detect_or_default(
            "content",
            "/path/note.md",
            default_name="generic",
        )

        assert result is generic_profile

    def test_detect_or_default_raises_when_default_not_registered(self, registry):
        """Should raise ValueError when default profile not registered."""
        with pytest.raises(ValueError, match="Default profile 'generic' is not registered"):
            registry.detect_or_default(
                "content",
                "/path/note.md",
                default_name="generic",
            )

    def test_detect_or_default_uses_custom_default_name(self, registry, mock_profile):
        """Should use custom default name."""
        registry.register(mock_profile)

        result = registry.detect_or_default(
            "content",
            "/path/note.md",
            default_name="test-profile",
        )

        assert result is mock_profile

    def test_detect_or_default_default_parameter(self, registry, generic_profile):
        """Should use 'generic' as default when default_name not specified."""
        registry.register(generic_profile)

        result = registry.detect_or_default("content", "/path/note.md")

        assert result is generic_profile


class TestMetadataProfileRegistryHelpers:
    """Tests for helper methods."""

    def test_list_profiles_empty_registry(self, registry):
        """Should return empty list for empty registry."""
        result = registry.list_profiles()

        assert result == []

    def test_list_profiles_returns_sorted_names(self, registry):
        """Should return sorted list of profile names."""
        profiles = []
        for name in ["zebra", "alpha", "beta"]:
            profile = MagicMock()
            profile.name = name
            profiles.append(profile)
            registry.register(profile)

        result = registry.list_profiles()

        assert result == ["alpha", "beta", "zebra"]

    def test_is_registered_returns_true_for_registered(self, registry, mock_profile):
        """Should return True for registered profiles."""
        registry.register(mock_profile)

        assert registry.is_registered("test-profile") is True

    def test_is_registered_returns_false_for_unregistered(self, registry):
        """Should return False for unregistered profiles."""
        assert registry.is_registered("nonexistent") is False

    def test_list_profiles_reflects_registration_changes(self, registry, mock_profile):
        """Should reflect changes after registration/unregistration."""
        assert registry.list_profiles() == []

        registry.register(mock_profile)
        assert registry.list_profiles() == ["test-profile"]

        registry.unregister("test-profile")
        assert registry.list_profiles() == []


class TestMetadataProfileRegistryEdgeCases:
    """Edge case tests for MetadataProfileRegistry."""

    def test_empty_path_patterns_list(self, registry, mock_profile):
        """Should handle empty path patterns list."""
        registry.register(mock_profile, path_patterns=[])

        result = registry.detect("content", "/path/note.md")

        assert result is None

    def test_empty_detectors_list(self, registry, mock_profile):
        """Should handle empty detectors list."""
        registry.register(mock_profile, detectors=[])

        result = registry.detect("content", "/path/note.md")

        assert result is None

    def test_invalid_regex_pattern_raises(self, registry, mock_profile):
        """Should raise error for invalid regex patterns."""
        with pytest.raises(re.error):
            registry.register(mock_profile, path_patterns=[r"[invalid(regex"])

    def test_detector_receives_correct_arguments(self, registry, mock_profile):
        """Should pass content and path to detector."""
        received_args = []

        def capturing_detector(content, path):
            received_args.append((content, path))
            return True

        registry.register(mock_profile, detectors=[capturing_detector])
        registry.detect("test content", "/test/path.md")

        assert received_args == [("test content", "/test/path.md")]

    def test_priority_handles_negative_values(self, registry):
        """Should handle negative priority values."""
        low_priority = MagicMock()
        low_priority.name = "low"
        high_priority = MagicMock()
        high_priority.name = "high"

        registry.register(low_priority, path_patterns=[r"\.md"], priority=-100)
        registry.register(high_priority, path_patterns=[r"\.md"], priority=0)

        result = registry.detect("content", "/path/note.md")

        assert result is high_priority

    def test_priority_with_equal_values(self, registry):
        """Should handle profiles with equal priority consistently."""
        profile1 = MagicMock()
        profile1.name = "profile1"
        profile2 = MagicMock()
        profile2.name = "profile2"

        registry.register(profile1, path_patterns=[r"\.md"], priority=10)
        registry.register(profile2, path_patterns=[r"\.md"], priority=10)

        # Should return one of them consistently (order depends on dict iteration)
        result = registry.detect("content", "/path/note.md")
        assert result in [profile1, profile2]

    def test_multiple_profiles_no_match(self, registry):
        """Should return None when multiple profiles exist but none match."""
        for i in range(5):
            profile = MagicMock()
            profile.name = f"profile{i}"
            registry.register(profile, path_patterns=[f"pattern{i}"])

        result = registry.detect("content", "/nomatch/path.md")

        assert result is None

    def test_path_pattern_with_special_regex_chars(self, registry, mock_profile):
        """Should handle path patterns with special regex characters."""
        registry.register(mock_profile, path_patterns=[r"\.obsidian\.\w+"])

        result = registry.detect("content", "/vault/.obsidian.config/note.md")

        assert result is mock_profile

    def test_detector_returns_falsy_values(self, registry, mock_profile):
        """Should handle detectors returning various falsy values."""
        detector_none = lambda c, p: None
        detector_false = lambda c, p: False
        detector_empty = lambda c, p: ""
        detector_zero = lambda c, p: 0

        for detector in [detector_none, detector_false, detector_empty, detector_zero]:
            registry.register(mock_profile, detectors=[detector])
            result = registry.detect("content", "/path/note.md")
            assert result is None
            registry.unregister("test-profile")

    def test_very_long_path(self, registry, mock_profile):
        """Should handle very long file paths."""
        registry.register(mock_profile, path_patterns=[r"\.obsidian"])

        long_path = "/" + "/".join(["dir"] * 100) + "/.obsidian/note.md"
        result = registry.detect("content", long_path)

        assert result is mock_profile

    def test_very_long_content(self, registry, mock_profile):
        """Should handle very long content strings."""
        detector = lambda c, p: "needle" in c
        registry.register(mock_profile, detectors=[detector])

        long_content = "x" * 100000 + "needle" + "x" * 100000
        result = registry.detect(long_content, "/path/note.md")

        assert result is mock_profile

    def test_unicode_in_path(self, registry, mock_profile):
        """Should handle unicode characters in paths."""
        registry.register(mock_profile, path_patterns=[r"日本語"])

        result = registry.detect("content", "/vault/日本語/note.md")

        assert result is mock_profile

    def test_unicode_in_content(self, registry, mock_profile):
        """Should handle unicode characters in content."""
        detector = lambda c, p: "日本語" in c
        registry.register(mock_profile, detectors=[detector])

        result = registry.detect("content with 日本語", "/path/note.md")

        assert result is mock_profile


class TestDefaultProfileRegistry:
    """Tests for the default global registry singleton."""

    def test_get_default_registry_returns_instance(self):
        """Should return MetadataProfileRegistry instance."""
        registry = get_default_profile_registry()

        assert isinstance(registry, MetadataProfileRegistry)

    def test_get_default_registry_is_singleton(self):
        """Should return same instance on repeated calls."""
        registry1 = get_default_profile_registry()
        registry2 = get_default_profile_registry()

        assert registry1 is registry2

    def test_default_registry_has_builtin_profiles(self):
        """Should have built-in profiles registered."""
        registry = get_default_profile_registry()

        profiles = registry.list_profiles()
        assert "generic" in profiles
        assert "obsidian" in profiles
        assert "drafts" in profiles

    def test_default_registry_can_detect_generic(self):
        """Should detect generic profile as fallback."""
        registry = get_default_profile_registry()

        result = registry.detect_or_default("content", "/path/note.md")

        assert result is not None
        assert result.name == "generic"

    def test_default_registry_can_detect_obsidian_by_path(self):
        """Should detect obsidian profile by path pattern."""
        registry = get_default_profile_registry()

        result = registry.detect("content", "/vault/.obsidian/note.md")

        assert result is not None
        assert result.name == "obsidian"

    def test_default_registry_can_detect_drafts_by_path(self):
        """Should detect drafts profile by path pattern."""
        registry = get_default_profile_registry()

        result = registry.detect("content", "/drafts/note.md")

        assert result is not None
        assert result.name == "drafts"

    def test_builtin_profiles_have_correct_priorities(self):
        """Should have correct priority settings for built-in profiles."""
        registry = get_default_profile_registry()

        generic_reg = registry._profiles["generic"]
        obsidian_reg = registry._profiles["obsidian"]
        drafts_reg = registry._profiles["drafts"]

        # Obsidian and Drafts should have higher priority than generic
        assert obsidian_reg.priority > generic_reg.priority
        assert drafts_reg.priority > generic_reg.priority

    def test_register_builtin_profiles_on_empty_registry(self):
        """Should register all built-in profiles on empty registry."""
        test_registry = MetadataProfileRegistry()
        _register_builtin_profiles(test_registry)

        profiles = test_registry.list_profiles()
        assert "generic" in profiles
        assert "obsidian" in profiles
        assert "drafts" in profiles


class TestMetadataProfileRegistryIntegration:
    """Integration tests with real profiles."""

    def test_register_and_detect_generic_profile(self, registry):
        """Should work with real GenericProfile."""
        generic = GenericProfile()
        registry.register(generic, priority=-100)

        result = registry.detect_or_default("# Test", "/path/note.md")

        assert result is generic

    def test_priority_ordering_with_multiple_profiles(self, registry):
        """Should respect priority ordering with multiple profiles."""
        generic = GenericProfile()
        high_priority = MagicMock()
        high_priority.name = "high-priority"

        registry.register(generic, path_patterns=[r"\.md"], priority=0)
        registry.register(high_priority, path_patterns=[r"\.md"], priority=100)

        result = registry.detect("content", "/path/note.md")

        assert result is high_priority

    def test_path_and_content_detection_combined(self, registry, mock_profile):
        """Should detect using both path patterns and content detectors."""
        detector = lambda c, p: "obsidian" in c
        registry.register(
            mock_profile,
            path_patterns=[r"\.obsidian"],
            detectors=[detector],
        )

        result1 = registry.detect("content", "/vault/.obsidian/note.md")
        result2 = registry.detect("obsidian content", "/other/path/note.md")
        result3 = registry.detect("plain content", "/other/path/note.md")

        assert result1 is mock_profile  # Path match
        assert result2 is mock_profile  # Content match
        assert result3 is None  # No match

    def test_full_workflow_register_detect_unregister(self, registry, mock_profile):
        """Should handle full workflow of register, detect, unregister."""
        # Register
        registry.register(mock_profile, path_patterns=[r"\.test"])
        assert registry.is_registered("test-profile")

        # Detect
        result = registry.detect("content", "/path/.test/note.md")
        assert result is mock_profile

        # Unregister
        registry.unregister("test-profile")
        assert not registry.is_registered("test-profile")

        # Should not detect after unregister
        result = registry.detect("content", "/path/.test/note.md")
        assert result is None

    def test_complex_path_pattern_matching(self, registry, mock_profile):
        """Should handle complex regex patterns."""
        # Match files with .obsidian in path OR ending in .draft
        registry.register(
            mock_profile,
            path_patterns=[r"\.obsidian", r"\.draft$"],
        )

        result1 = registry.detect("content", "/vault/.obsidian/note.md")
        result2 = registry.detect("content", "/docs/myfile.draft")
        result3 = registry.detect("content", "/docs/myfile.md")

        assert result1 is mock_profile
        assert result2 is mock_profile
        assert result3 is None
