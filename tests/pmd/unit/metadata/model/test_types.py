"""Tests for core metadata types."""

import pytest

from pmd.extraction.types import (
    ExtractedMetadata,
    MetadataProfile,
    StoredDocumentMetadata,
)


class TestExtractedMetadataCreation:
    """Tests for ExtractedMetadata creation and initialization."""

    def test_default_initialization(self):
        """Should create ExtractedMetadata with default empty values."""
        metadata = ExtractedMetadata()

        assert metadata.tags == set()
        assert metadata.source_tags == []
        assert metadata.attributes == {}
        assert metadata.extraction_source == ""

    def test_initialization_with_tags(self):
        """Should create ExtractedMetadata with specified tags."""
        metadata = ExtractedMetadata(
            tags={"python", "rust"},
            source_tags=["Python", "Rust"],
        )

        assert metadata.tags == {"python", "rust"}
        assert metadata.source_tags == ["Python", "Rust"]

    def test_initialization_with_attributes(self):
        """Should create ExtractedMetadata with specified attributes."""
        metadata = ExtractedMetadata(
            attributes={"author": "John", "date": "2024-01-01"}
        )

        assert metadata.attributes == {"author": "John", "date": "2024-01-01"}

    def test_initialization_with_extraction_source(self):
        """Should create ExtractedMetadata with specified extraction source."""
        metadata = ExtractedMetadata(extraction_source="obsidian")

        assert metadata.extraction_source == "obsidian"

    def test_initialization_with_all_fields(self):
        """Should create ExtractedMetadata with all fields specified."""
        metadata = ExtractedMetadata(
            tags={"python", "web"},
            source_tags=["Python", "Web"],
            attributes={"author": "Jane", "category": "tutorial"},
            extraction_source="drafts",
        )

        assert metadata.tags == {"python", "web"}
        assert metadata.source_tags == ["Python", "Web"]
        assert metadata.attributes == {"author": "Jane", "category": "tutorial"}
        assert metadata.extraction_source == "drafts"


class TestExtractedMetadataMergeBasic:
    """Tests for basic merge functionality."""

    def test_merge_empty_metadata(self):
        """Merging two empty metadata should return empty metadata."""
        m1 = ExtractedMetadata()
        m2 = ExtractedMetadata()

        result = m1.merge(m2)

        assert result.tags == set()
        assert result.source_tags == []
        assert result.attributes == {}
        assert result.extraction_source == ""

    def test_merge_tags_combines_sets(self):
        """Merging should combine tag sets."""
        m1 = ExtractedMetadata(tags={"python", "web"})
        m2 = ExtractedMetadata(tags={"rust", "web"})

        result = m1.merge(m2)

        assert result.tags == {"python", "web", "rust"}

    def test_merge_source_tags_concatenates(self):
        """Merging should concatenate source_tags lists."""
        m1 = ExtractedMetadata(source_tags=["Python", "Web"])
        m2 = ExtractedMetadata(source_tags=["Rust", "API"])

        result = m1.merge(m2)

        assert result.source_tags == ["Python", "Web", "Rust", "API"]

    def test_merge_disjoint_attributes(self):
        """Merging metadata with different attributes should combine them."""
        m1 = ExtractedMetadata(attributes={"author": "Alice"})
        m2 = ExtractedMetadata(attributes={"date": "2024-01-01"})

        result = m1.merge(m2)

        assert result.attributes == {"author": "Alice", "date": "2024-01-01"}

    def test_merge_extraction_source_keeps_first(self):
        """Merging should keep the first extraction source if set."""
        m1 = ExtractedMetadata(extraction_source="obsidian")
        m2 = ExtractedMetadata(extraction_source="drafts")

        result = m1.merge(m2)

        assert result.extraction_source == "obsidian"

    def test_merge_extraction_source_uses_second_if_first_empty(self):
        """Merging should use second extraction source if first is empty."""
        m1 = ExtractedMetadata(extraction_source="")
        m2 = ExtractedMetadata(extraction_source="drafts")

        result = m1.merge(m2)

        assert result.extraction_source == "drafts"


class TestExtractedMetadataMergeAttributes:
    """Tests for attribute merging behavior."""

    def test_merge_duplicate_string_attributes_creates_list(self):
        """Merging duplicate string attributes should create list."""
        m1 = ExtractedMetadata(attributes={"tag": "python"})
        m2 = ExtractedMetadata(attributes={"tag": "rust"})

        result = m1.merge(m2)

        assert result.attributes["tag"] == ["python", "rust"]

    def test_merge_list_with_string_extends_list(self):
        """Merging list attribute with string should extend list."""
        m1 = ExtractedMetadata(attributes={"tags": ["python", "web"]})
        m2 = ExtractedMetadata(attributes={"tags": "rust"})

        result = m1.merge(m2)

        assert result.attributes["tags"] == ["python", "web", "rust"]

    def test_merge_string_with_list_prepends_to_list(self):
        """Merging string with list should prepend string to list."""
        m1 = ExtractedMetadata(attributes={"tags": "python"})
        m2 = ExtractedMetadata(attributes={"tags": ["rust", "web"]})

        result = m1.merge(m2)

        assert result.attributes["tags"] == ["python", "rust", "web"]

    def test_merge_two_lists_concatenates(self):
        """Merging two list attributes should concatenate them."""
        m1 = ExtractedMetadata(attributes={"tags": ["python", "web"]})
        m2 = ExtractedMetadata(attributes={"tags": ["rust", "api"]})

        result = m1.merge(m2)

        assert result.attributes["tags"] == ["python", "web", "rust", "api"]

    def test_merge_multiple_attributes_with_mixed_types(self):
        """Should handle merging multiple attributes with different types."""
        m1 = ExtractedMetadata(
            attributes={
                "author": "Alice",
                "tags": ["python"],
                "category": "tutorial",
            }
        )
        m2 = ExtractedMetadata(
            attributes={
                "author": "Bob",
                "tags": ["web"],
                "date": "2024-01-01",
            }
        )

        result = m1.merge(m2)

        assert result.attributes["author"] == ["Alice", "Bob"]
        assert result.attributes["tags"] == ["python", "web"]
        assert result.attributes["category"] == "tutorial"
        assert result.attributes["date"] == "2024-01-01"


class TestExtractedMetadataMergeChaining:
    """Tests for chaining multiple merge operations."""

    def test_merge_three_metadata_objects(self):
        """Should correctly merge three metadata objects in sequence."""
        m1 = ExtractedMetadata(tags={"python"}, attributes={"a": "1"})
        m2 = ExtractedMetadata(tags={"rust"}, attributes={"b": "2"})
        m3 = ExtractedMetadata(tags={"go"}, attributes={"c": "3"})

        result = m1.merge(m2).merge(m3)

        assert result.tags == {"python", "rust", "go"}
        assert result.attributes == {"a": "1", "b": "2", "c": "3"}

    def test_merge_accumulates_duplicate_values(self):
        """Merging multiple times should accumulate duplicate attribute values."""
        m1 = ExtractedMetadata(attributes={"tag": "a"})
        m2 = ExtractedMetadata(attributes={"tag": "b"})
        m3 = ExtractedMetadata(attributes={"tag": "c"})

        result = m1.merge(m2).merge(m3)

        assert result.attributes["tag"] == ["a", "b", "c"]

    def test_merge_does_not_modify_original(self):
        """Merge should create new instance without modifying originals."""
        m1 = ExtractedMetadata(tags={"python"}, attributes={"author": "Alice"})
        m2 = ExtractedMetadata(tags={"rust"}, attributes={"author": "Bob"})

        original_m1_tags = m1.tags.copy()
        original_m1_attrs = m1.attributes.copy()
        original_m2_tags = m2.tags.copy()
        original_m2_attrs = m2.attributes.copy()

        result = m1.merge(m2)

        # Verify originals unchanged
        assert m1.tags == original_m1_tags
        assert m1.attributes == original_m1_attrs
        assert m2.tags == original_m2_tags
        assert m2.attributes == original_m2_attrs

        # Verify result is different
        assert result.tags == {"python", "rust"}
        assert result.attributes["author"] == ["Alice", "Bob"]


class TestExtractedMetadataEdgeCases:
    """Edge case tests for ExtractedMetadata."""

    def test_empty_tags_set(self):
        """Should handle empty tag set correctly."""
        metadata = ExtractedMetadata(tags=set())

        assert metadata.tags == set()
        assert len(metadata.tags) == 0

    def test_empty_source_tags_list(self):
        """Should handle empty source tags list correctly."""
        metadata = ExtractedMetadata(source_tags=[])

        assert metadata.source_tags == []
        assert len(metadata.source_tags) == 0

    def test_empty_attributes_dict(self):
        """Should handle empty attributes dict correctly."""
        metadata = ExtractedMetadata(attributes={})

        assert metadata.attributes == {}
        assert len(metadata.attributes) == 0

    def test_tags_with_special_characters(self):
        """Should handle tags with special characters."""
        metadata = ExtractedMetadata(
            tags={"python-3", "c++", "c#", "project/active"}
        )

        assert "python-3" in metadata.tags
        assert "c++" in metadata.tags
        assert "c#" in metadata.tags
        assert "project/active" in metadata.tags

    def test_attributes_with_empty_string_values(self):
        """Should handle attributes with empty string values."""
        metadata = ExtractedMetadata(attributes={"author": "", "date": ""})

        assert metadata.attributes["author"] == ""
        assert metadata.attributes["date"] == ""

    def test_attributes_with_empty_list_values(self):
        """Should handle attributes with empty list values."""
        metadata = ExtractedMetadata(attributes={"tags": []})

        assert metadata.attributes["tags"] == []

    def test_very_long_tag_names(self):
        """Should handle very long tag names."""
        long_tag = "a" * 1000
        metadata = ExtractedMetadata(tags={long_tag})

        assert long_tag in metadata.tags

    def test_many_tags(self):
        """Should handle large number of tags efficiently."""
        many_tags = {f"tag{i}" for i in range(1000)}
        metadata = ExtractedMetadata(tags=many_tags)

        assert len(metadata.tags) == 1000
        assert "tag500" in metadata.tags

    def test_many_attributes(self):
        """Should handle large number of attributes."""
        many_attrs = {f"attr{i}": f"value{i}" for i in range(100)}
        metadata = ExtractedMetadata(attributes=many_attrs)

        assert len(metadata.attributes) == 100
        assert metadata.attributes["attr50"] == "value50"

    def test_unicode_in_tags(self):
        """Should handle unicode characters in tags."""
        metadata = ExtractedMetadata(tags={"python", "日本語", "emoji😀"})

        assert "日本語" in metadata.tags
        assert "emoji😀" in metadata.tags

    def test_unicode_in_attributes(self):
        """Should handle unicode in attribute values."""
        metadata = ExtractedMetadata(
            attributes={"author": "山田太郎", "emoji": "🎉"}
        )

        assert metadata.attributes["author"] == "山田太郎"
        assert metadata.attributes["emoji"] == "🎉"

    def test_merge_with_empty_metadata(self):
        """Merging with empty metadata should preserve original values."""
        m1 = ExtractedMetadata(
            tags={"python"},
            source_tags=["Python"],
            attributes={"author": "Alice"},
            extraction_source="obsidian",
        )
        m2 = ExtractedMetadata()

        result = m1.merge(m2)

        assert result.tags == {"python"}
        assert result.source_tags == ["Python"]
        assert result.attributes == {"author": "Alice"}
        assert result.extraction_source == "obsidian"

    def test_whitespace_in_tags(self):
        """Should handle whitespace in tag names."""
        metadata = ExtractedMetadata(tags={"multi word", " leading", "trailing "})

        assert "multi word" in metadata.tags
        assert " leading" in metadata.tags
        assert "trailing " in metadata.tags


class TestMetadataProfileProtocol:
    """Tests for MetadataProfile protocol."""

    def test_protocol_implementation_with_all_methods(self):
        """Class implementing all protocol methods should be recognized."""

        class TestProfile:
            name = "test"

            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                return ExtractedMetadata()

            def normalize_tag(self, tag: str) -> str:
                return tag.lower()

        profile = TestProfile()

        assert isinstance(profile, MetadataProfile)
        assert profile.name == "test"

    def test_protocol_implementation_extract_metadata(self):
        """Should correctly call extract_metadata method."""

        class TestProfile:
            name = "test"

            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                return ExtractedMetadata(
                    tags={"found"},
                    extraction_source="test",
                )

            def normalize_tag(self, tag: str) -> str:
                return tag.lower()

        profile = TestProfile()
        result = profile.extract_metadata("content", "path.md")

        assert result.tags == {"found"}
        assert result.extraction_source == "test"

    def test_protocol_implementation_normalize_tag(self):
        """Should correctly call normalize_tag method."""

        class TestProfile:
            name = "test"

            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                return ExtractedMetadata()

            def normalize_tag(self, tag: str) -> str:
                return tag.lower().strip()

        profile = TestProfile()
        result = profile.normalize_tag("  PYTHON  ")

        assert result == "python"

    def test_protocol_incomplete_implementation_missing_method(self):
        """Class missing protocol methods should still be created but not match protocol."""

        class IncompleteProfile:
            name = "incomplete"

            def normalize_tag(self, tag: str) -> str:
                return tag.lower()

        profile = IncompleteProfile()

        # Should not be recognized as MetadataProfile
        assert not isinstance(profile, MetadataProfile)

    def test_protocol_incomplete_implementation_missing_name(self):
        """Class missing name attribute should not match protocol."""

        class NoNameProfile:
            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                return ExtractedMetadata()

            def normalize_tag(self, tag: str) -> str:
                return tag.lower()

        profile = NoNameProfile()

        # Should not be recognized due to missing name
        assert not isinstance(profile, MetadataProfile)


class TestMetadataProfileProtocolUsage:
    """Tests for using MetadataProfile protocol in practice."""

    def test_multiple_profiles_with_different_behaviors(self):
        """Different profile implementations should have different behaviors."""

        class ObsidianProfile:
            name = "obsidian"

            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                return ExtractedMetadata(
                    tags={"obsidian"},
                    extraction_source="obsidian",
                )

            def normalize_tag(self, tag: str) -> str:
                return tag.lower().replace("/", "-")

        class DraftsProfile:
            name = "drafts"

            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                return ExtractedMetadata(
                    tags={"drafts"},
                    extraction_source="drafts",
                )

            def normalize_tag(self, tag: str) -> str:
                return tag.lower()

        obsidian = ObsidianProfile()
        drafts = DraftsProfile()

        assert isinstance(obsidian, MetadataProfile)
        assert isinstance(drafts, MetadataProfile)

        # Test different normalization behaviors
        assert obsidian.normalize_tag("Parent/Child") == "parent-child"
        assert drafts.normalize_tag("Parent/Child") == "parent/child"

    def test_profile_can_access_path_for_detection(self):
        """Profile can use path to influence extraction."""

        class PathAwareProfile:
            name = "path-aware"

            def extract_metadata(self, content: str, path: str) -> ExtractedMetadata:
                if "obsidian" in path:
                    return ExtractedMetadata(tags={"obsidian-doc"})
                return ExtractedMetadata(tags={"generic-doc"})

            def normalize_tag(self, tag: str) -> str:
                return tag.lower()

        profile = PathAwareProfile()

        result1 = profile.extract_metadata("content", "/obsidian/notes/file.md")
        result2 = profile.extract_metadata("content", "/drafts/file.md")

        assert "obsidian-doc" in result1.tags
        assert "generic-doc" in result2.tags


class TestStoredDocumentMetadataCreation:
    """Tests for StoredDocumentMetadata creation and initialization."""

    def test_initialization_with_required_fields(self):
        """Should create StoredDocumentMetadata with required fields."""
        metadata = StoredDocumentMetadata(
            document_id=123,
            profile_name="obsidian",
            tags={"python", "web"},
            source_tags=["Python", "Web"],
            extracted_at="2024-01-01T00:00:00Z",
        )

        assert metadata.document_id == 123
        assert metadata.profile_name == "obsidian"
        assert metadata.tags == {"python", "web"}
        assert metadata.source_tags == ["Python", "Web"]
        assert metadata.extracted_at == "2024-01-01T00:00:00Z"
        assert metadata.attributes == {}

    def test_initialization_with_all_fields(self):
        """Should create StoredDocumentMetadata with all fields."""
        metadata = StoredDocumentMetadata(
            document_id=456,
            profile_name="drafts",
            tags={"rust", "cli"},
            source_tags=["Rust", "CLI"],
            extracted_at="2024-01-15T12:30:00Z",
            attributes={"author": "Jane", "category": "tutorial"},
        )

        assert metadata.document_id == 456
        assert metadata.profile_name == "drafts"
        assert metadata.tags == {"rust", "cli"}
        assert metadata.source_tags == ["Rust", "CLI"]
        assert metadata.extracted_at == "2024-01-15T12:30:00Z"
        assert metadata.attributes == {"author": "Jane", "category": "tutorial"}

    def test_empty_tags_set(self):
        """Should handle empty tags set."""
        metadata = StoredDocumentMetadata(
            document_id=1,
            profile_name="generic",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )

        assert metadata.tags == set()
        assert metadata.source_tags == []

    def test_empty_attributes_dict(self):
        """Should use empty dict as default for attributes."""
        metadata = StoredDocumentMetadata(
            document_id=1,
            profile_name="generic",
            tags={"test"},
            source_tags=["Test"],
            extracted_at="2024-01-01T00:00:00Z",
        )

        assert metadata.attributes == {}

    def test_document_id_types(self):
        """Should handle different document ID values."""
        metadata1 = StoredDocumentMetadata(
            document_id=0,
            profile_name="generic",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )
        metadata2 = StoredDocumentMetadata(
            document_id=999999,
            profile_name="generic",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )

        assert metadata1.document_id == 0
        assert metadata2.document_id == 999999


class TestStoredDocumentMetadataEdgeCases:
    """Edge case tests for StoredDocumentMetadata."""

    def test_profile_name_variations(self):
        """Should handle various profile name formats."""
        metadata1 = StoredDocumentMetadata(
            document_id=1,
            profile_name="obsidian",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )
        metadata2 = StoredDocumentMetadata(
            document_id=2,
            profile_name="custom-profile-v2",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )
        metadata3 = StoredDocumentMetadata(
            document_id=3,
            profile_name="",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )

        assert metadata1.profile_name == "obsidian"
        assert metadata2.profile_name == "custom-profile-v2"
        assert metadata3.profile_name == ""

    def test_timestamp_formats(self):
        """Should handle various timestamp string formats."""
        metadata1 = StoredDocumentMetadata(
            document_id=1,
            profile_name="generic",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )
        metadata2 = StoredDocumentMetadata(
            document_id=2,
            profile_name="generic",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01 00:00:00",
        )
        metadata3 = StoredDocumentMetadata(
            document_id=3,
            profile_name="generic",
            tags=set(),
            source_tags=[],
            extracted_at="1704067200",  # Unix timestamp as string
        )

        assert metadata1.extracted_at == "2024-01-01T00:00:00Z"
        assert metadata2.extracted_at == "2024-01-01 00:00:00"
        assert metadata3.extracted_at == "1704067200"

    def test_large_tag_sets(self):
        """Should handle large numbers of tags."""
        many_tags = {f"tag{i}" for i in range(500)}
        many_source_tags = [f"Tag{i}" for i in range(500)]

        metadata = StoredDocumentMetadata(
            document_id=1,
            profile_name="generic",
            tags=many_tags,
            source_tags=many_source_tags,
            extracted_at="2024-01-01T00:00:00Z",
        )

        assert len(metadata.tags) == 500
        assert len(metadata.source_tags) == 500
        assert "tag250" in metadata.tags
        assert "Tag250" in metadata.source_tags

    def test_attributes_with_complex_types(self):
        """Should handle attributes with various value types."""
        metadata = StoredDocumentMetadata(
            document_id=1,
            profile_name="generic",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
            attributes={
                "string": "value",
                "number": 42,
                "float": 3.14,
                "list": ["a", "b", "c"],
                "nested_dict": {"key": "value"},
                "bool": True,
                "none": None,
            },
        )

        assert metadata.attributes["string"] == "value"
        assert metadata.attributes["number"] == 42
        assert metadata.attributes["float"] == 3.14
        assert metadata.attributes["list"] == ["a", "b", "c"]
        assert metadata.attributes["nested_dict"] == {"key": "value"}
        assert metadata.attributes["bool"] is True
        assert metadata.attributes["none"] is None

    def test_unicode_in_all_fields(self):
        """Should handle unicode characters in all fields."""
        metadata = StoredDocumentMetadata(
            document_id=1,
            profile_name="日本語プロファイル",
            tags={"タグ1", "emoji😀"},
            source_tags=["タグ1", "Emoji😀"],
            extracted_at="2024-01-01T00:00:00Z",
            attributes={"著者": "山田太郎", "絵文字": "🎉"},
        )

        assert metadata.profile_name == "日本語プロファイル"
        assert "タグ1" in metadata.tags
        assert "emoji😀" in metadata.tags
        assert metadata.attributes["著者"] == "山田太郎"
        assert metadata.attributes["絵文字"] == "🎉"


class TestExtractedMetadataEquality:
    """Tests for ExtractedMetadata equality comparisons."""

    def test_equal_empty_metadata(self):
        """Two empty metadata objects should be equal."""
        m1 = ExtractedMetadata()
        m2 = ExtractedMetadata()

        assert m1 == m2

    def test_equal_with_same_values(self):
        """Metadata with same values should be equal."""
        m1 = ExtractedMetadata(
            tags={"python", "web"},
            source_tags=["Python", "Web"],
            attributes={"author": "Alice"},
            extraction_source="obsidian",
        )
        m2 = ExtractedMetadata(
            tags={"python", "web"},
            source_tags=["Python", "Web"],
            attributes={"author": "Alice"},
            extraction_source="obsidian",
        )

        assert m1 == m2

    def test_not_equal_different_tags(self):
        """Metadata with different tags should not be equal."""
        m1 = ExtractedMetadata(tags={"python"})
        m2 = ExtractedMetadata(tags={"rust"})

        assert m1 != m2

    def test_not_equal_different_attributes(self):
        """Metadata with different attributes should not be equal."""
        m1 = ExtractedMetadata(attributes={"author": "Alice"})
        m2 = ExtractedMetadata(attributes={"author": "Bob"})

        assert m1 != m2


class TestStoredDocumentMetadataEquality:
    """Tests for StoredDocumentMetadata equality comparisons."""

    def test_equal_with_same_values(self):
        """Metadata with same values should be equal."""
        m1 = StoredDocumentMetadata(
            document_id=1,
            profile_name="obsidian",
            tags={"python"},
            source_tags=["Python"],
            extracted_at="2024-01-01T00:00:00Z",
            attributes={"author": "Alice"},
        )
        m2 = StoredDocumentMetadata(
            document_id=1,
            profile_name="obsidian",
            tags={"python"},
            source_tags=["Python"],
            extracted_at="2024-01-01T00:00:00Z",
            attributes={"author": "Alice"},
        )

        assert m1 == m2

    def test_not_equal_different_document_id(self):
        """Metadata with different document IDs should not be equal."""
        m1 = StoredDocumentMetadata(
            document_id=1,
            profile_name="obsidian",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )
        m2 = StoredDocumentMetadata(
            document_id=2,
            profile_name="obsidian",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )

        assert m1 != m2

    def test_not_equal_different_extracted_at(self):
        """Metadata with different extraction times should not be equal."""
        m1 = StoredDocumentMetadata(
            document_id=1,
            profile_name="obsidian",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-01T00:00:00Z",
        )
        m2 = StoredDocumentMetadata(
            document_id=1,
            profile_name="obsidian",
            tags=set(),
            source_tags=[],
            extracted_at="2024-01-02T00:00:00Z",
        )

        assert m1 != m2
