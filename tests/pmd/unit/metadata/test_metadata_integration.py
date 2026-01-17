"""Tests for metadata module integration.

Tests verify:
1. Cross-module interactions work correctly
2. Types are compatible across submodules
"""

import pytest


class TestCrossModuleInteractions:
    """Test that types flow correctly between submodules."""

    def test_extraction_produces_expected_metadata_type(self):
        """Extraction profiles return ExtractedMetadata compatible with model."""
        from pmd.metadata import GenericProfile, ExtractedMetadata

        profile = GenericProfile()
        content = """---
tags: [python, api]
title: Test
---
Content with #inline tag.
"""
        result = profile.extract_metadata(content, "/test.md")

        # Should be an ExtractedMetadata instance
        assert isinstance(result, ExtractedMetadata)
        assert "python" in result.tags
        assert "api" in result.tags
        assert "inline" in result.tags

    def test_query_inference_with_model_aliases(self):
        """Query inference can use aliases from model."""
        from pmd.metadata import create_default_matcher, load_default_aliases

        # Get aliases from model
        aliases = load_default_aliases()
        assert aliases.resolve("py") == "python"

        # Create matcher with default aliases
        matcher = create_default_matcher()
        tags = matcher.get_matching_tags("quick py script")
        assert "python" in tags

    def test_stored_metadata_compatible_with_extracted(self):
        """StoredDocumentMetadata can be created from ExtractedMetadata fields."""
        from pmd.metadata import ExtractedMetadata, StoredDocumentMetadata

        extracted = ExtractedMetadata(
            tags={"python", "web"},
            source_tags=["#python", "#web"],
            attributes={"title": "Test"},
            extraction_source="generic",
        )

        # Convert to stored format
        stored = StoredDocumentMetadata(
            document_id=1,
            profile_name=extracted.extraction_source,
            tags=extracted.tags,
            source_tags=extracted.source_tags,
            extracted_at="2024-01-01T00:00:00",
            attributes=extracted.attributes,
        )

        assert stored.tags == extracted.tags
        assert stored.source_tags == extracted.source_tags
        assert stored.profile_name == "generic"

    def test_ontology_expansion_with_query_scoring(self):
        """Ontology expansion produces tags compatible with query scoring."""
        from pmd.metadata import (
            Ontology,
            MetadataBoostConfig,
        )

        ontology = Ontology({
            "ml": {"children": ["ml/supervised", "ml/unsupervised"]},
            "ml/supervised": {"children": ["ml/supervised/regression"]},
        })

        # Expand tags - this would be used for scoring
        expanded = ontology.expand_for_matching(["ml/supervised/regression"])

        assert "ml/supervised/regression" in expanded
        assert expanded["ml/supervised/regression"] == 1.0
        assert "ml/supervised" in expanded
        assert expanded["ml/supervised"] == 0.7
        assert "ml" in expanded
        # Note: grandparent weight is 0.7 * 0.7 = 0.49
        assert abs(expanded["ml"] - 0.49) < 0.001

    def test_registry_provides_correct_profile_types(self):
        """Profile registry returns profiles compatible with extraction protocol."""
        from pmd.metadata import (
            get_default_profile_registry,
            ExtractedMetadata,
            MetadataProfile,
        )

        registry = get_default_profile_registry()

        # Generic profile
        generic = registry.get("generic")
        assert generic is not None
        assert hasattr(generic, "extract_metadata")
        assert hasattr(generic, "normalize_tag")

        # Test extraction
        content = "# Test\n\nSome #tag content."
        result = generic.extract_metadata(content, "/test.md")
        assert isinstance(result, ExtractedMetadata)


class TestPublicAPIConsistency:
    """Test that public API exports are consistent."""

    def test_all_model_types_exported(self):
        """Model types are exported from main metadata module."""
        from pmd.metadata import (
            ExtractedMetadata,
            MetadataProfile,
            StoredDocumentMetadata,
            Ontology,
            OntologyNode,
            TagAliases,
        )
        # If we get here without ImportError, types are exported
        assert True

    def test_all_extraction_types_exported(self):
        """Extraction types are exported from main metadata module."""
        from pmd.metadata import (
            GenericProfile,
            DraftsProfile,
            ObsidianProfile,
            MetadataProfileRegistry,
            get_default_profile_registry,
            FrontmatterResult,
            parse_frontmatter,
            extract_inline_tags,
            extract_tags_from_field,
        )
        assert True

    def test_all_query_types_exported(self):
        """Query types are exported from main metadata module."""
        from pmd.metadata import (
            LexicalTagMatcher,
            TagMatch,
            create_default_matcher,
            TagRetriever,
            TagSearchConfig,
            create_tag_retriever,
            MetadataBoostConfig,
            BoostResult,
            WeightedBoostResult,
            apply_metadata_boost,
            apply_metadata_boost_v2,
        )
        assert True

    def test_store_types_exported(self):
        """Store types are exported from pmd.store.repositories."""
        from pmd.store.repositories.metadata import DocumentMetadataRepository
        assert True

    def test_subpackage_imports_work(self):
        """Direct subpackage imports work correctly."""
        from pmd.metadata.model import Ontology, ExtractedMetadata
        from pmd.metadata.extraction import GenericProfile
        from pmd.metadata.query import LexicalTagMatcher
        from pmd.store.repositories.metadata import DocumentMetadataRepository
        assert True
