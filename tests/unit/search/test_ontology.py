"""Tests for tag ontology."""

import json
import pytest
from pathlib import Path
from tempfile import NamedTemporaryFile

from pmd.search.metadata.ontology import (
    Ontology,
    OntologyNode,
    load_ontology,
    load_default_ontology,
)


# Sample ontology for testing
SAMPLE_ADJACENCY = {
    "ml": {
        "children": ["ml/supervised", "ml/unsupervised"],
        "description": "Machine learning",
    },
    "ml/supervised": {
        "children": ["ml/supervised/classification", "ml/supervised/regression"],
        "description": "Supervised learning methods",
    },
    "ml/supervised/classification": {
        "children": [],
        "description": "Classification tasks",
    },
    "python": {
        "children": ["python/web", "python/testing"],
        "description": "Python programming",
    },
}


class TestOntologyBasics:
    """Tests for basic Ontology functionality."""

    def test_init_with_adjacency(self):
        """Should initialize with adjacency dict."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        assert ontology.has_tag("ml")
        assert ontology.has_tag("ml/supervised")

    def test_has_tag_parent(self):
        """Should recognize parent tags."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        assert ontology.has_tag("ml")
        assert ontology.has_tag("python")

    def test_has_tag_child(self):
        """Should recognize child tags."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        assert ontology.has_tag("ml/supervised")
        assert ontology.has_tag("python/web")

    def test_has_tag_unknown(self):
        """Should return False for unknown tags."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        assert not ontology.has_tag("unknown")
        assert not ontology.has_tag("ml/unknown")

    def test_get_description(self):
        """Should return description for known tags."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        assert ontology.get_description("ml") == "Machine learning"
        assert ontology.get_description("python") == "Python programming"

    def test_get_description_unknown(self):
        """Should return None for unknown tags."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        assert ontology.get_description("unknown") is None

    def test_all_tags(self):
        """Should return all known tags."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        all_tags = ontology.all_tags()

        assert "ml" in all_tags
        assert "ml/supervised" in all_tags
        assert "ml/supervised/classification" in all_tags
        assert "python" in all_tags
        assert "python/web" in all_tags

    def test_empty_ontology(self):
        """Should handle empty adjacency."""
        ontology = Ontology({})

        assert not ontology.has_tag("anything")
        assert ontology.all_tags() == set()


class TestOntologyGetAncestors:
    """Tests for get_ancestors method."""

    def test_direct_parent(self):
        """Should find direct parent."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        ancestors = ontology.get_ancestors("ml/supervised")

        assert ancestors == ["ml"]

    def test_multiple_ancestors(self):
        """Should find multiple levels of ancestors."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        ancestors = ontology.get_ancestors("ml/supervised/classification")

        assert ancestors == ["ml/supervised", "ml"]

    def test_no_ancestors(self):
        """Root tags should have no ancestors."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        ancestors = ontology.get_ancestors("ml")

        assert ancestors == []

    def test_max_hops_limit(self):
        """Should respect max_hops limit."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        ancestors = ontology.get_ancestors("ml/supervised/classification", max_hops=1)

        assert ancestors == ["ml/supervised"]
        assert "ml" not in ancestors

    def test_max_hops_zero(self):
        """max_hops=0 should return empty list."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        ancestors = ontology.get_ancestors("ml/supervised/classification", max_hops=0)

        assert ancestors == []

    def test_unknown_tag_ancestors(self):
        """Unknown tags should return empty list."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        ancestors = ontology.get_ancestors("unknown/tag")

        assert ancestors == []


class TestOntologyGetChildren:
    """Tests for get_children method."""

    def test_direct_children(self):
        """Should return direct children."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        children = ontology.get_children("ml")

        assert set(children) == {"ml/supervised", "ml/unsupervised"}

    def test_leaf_has_no_children(self):
        """Leaf tags should have no children."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        # ml/supervised/classification has empty children list
        children = ontology.get_children("ml/supervised/classification")

        assert children == []

    def test_unknown_tag_children(self):
        """Unknown tags should return empty list."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        children = ontology.get_children("unknown")

        assert children == []


class TestOntologyGetDescendants:
    """Tests for get_descendants method."""

    def test_all_descendants(self):
        """Should return all descendants."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        descendants = ontology.get_descendants("ml")

        # Should include all ml/* tags
        assert "ml/supervised" in descendants
        assert "ml/unsupervised" in descendants
        assert "ml/supervised/classification" in descendants
        assert "ml/supervised/regression" in descendants

    def test_direct_children_only(self):
        """max_depth=1 should return only direct children."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        descendants = ontology.get_descendants("ml", max_depth=1)

        assert "ml/supervised" in descendants
        assert "ml/unsupervised" in descendants
        assert "ml/supervised/classification" not in descendants

    def test_leaf_has_no_descendants(self):
        """Leaf tags should have no descendants."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        descendants = ontology.get_descendants("ml/supervised/classification")

        assert descendants == []


class TestOntologyExpandForMatching:
    """Tests for expand_for_matching method."""

    def test_single_tag_exact_match(self):
        """Single tag should get weight 1.0."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        expanded = ontology.expand_for_matching(["ml"])

        assert expanded["ml"] == 1.0

    def test_expands_to_parent(self):
        """Should expand to parent with reduced weight."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)

        expanded = ontology.expand_for_matching(["ml/supervised"])

        assert expanded["ml/supervised"] == 1.0
        assert expanded["ml"] == pytest.approx(0.7)

    def test_expands_multiple_levels(self):
        """Should expand multiple levels with decreasing weights."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)

        expanded = ontology.expand_for_matching(["ml/supervised/classification"])

        assert expanded["ml/supervised/classification"] == 1.0
        assert expanded["ml/supervised"] == pytest.approx(0.7)
        assert expanded["ml"] == pytest.approx(0.49)  # 0.7 * 0.7

    def test_multiple_tags(self):
        """Should expand multiple query tags."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)

        expanded = ontology.expand_for_matching(["ml/supervised", "python/web"])

        assert expanded["ml/supervised"] == 1.0
        assert expanded["ml"] == pytest.approx(0.7)
        assert expanded["python/web"] == 1.0
        assert expanded["python"] == pytest.approx(0.7)

    def test_accepts_set(self):
        """Should accept set of tags."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        expanded = ontology.expand_for_matching({"ml", "python"})

        assert "ml" in expanded
        assert "python" in expanded

    def test_max_hops_limit(self):
        """Should respect max_hops limit."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)

        expanded = ontology.expand_for_matching(
            ["ml/supervised/classification"],
            max_hops=1,
        )

        assert "ml/supervised/classification" in expanded
        assert "ml/supervised" in expanded
        assert "ml" not in expanded

    def test_no_duplicate_weights(self):
        """Overlapping expansions should keep highest weight."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)

        # Both tags would expand to "ml" - keep highest weight
        expanded = ontology.expand_for_matching([
            "ml/supervised/classification",  # ml gets 0.49
            "ml/supervised",                  # ml gets 0.7
        ])

        assert expanded["ml"] == pytest.approx(0.7)  # Highest weight kept

    def test_exact_match_overrides_expansion(self):
        """Exact match should override expansion weight."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.7)

        # ml/supervised expands to ml with 0.7, but ml is also exact match
        expanded = ontology.expand_for_matching(["ml/supervised", "ml"])

        assert expanded["ml"] == 1.0  # Exact match wins

    def test_empty_tags(self):
        """Empty tag list should return empty dict."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        expanded = ontology.expand_for_matching([])

        assert expanded == {}

    def test_unknown_tag_no_expansion(self):
        """Unknown tags should still get weight 1.0."""
        ontology = Ontology(SAMPLE_ADJACENCY)

        expanded = ontology.expand_for_matching(["unknown/tag"])

        assert expanded["unknown/tag"] == 1.0
        assert len(expanded) == 1

    def test_custom_parent_weight(self):
        """Should use custom parent_weight."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.5)

        expanded = ontology.expand_for_matching(["ml/supervised/classification"])

        assert expanded["ml/supervised"] == pytest.approx(0.5)
        assert expanded["ml"] == pytest.approx(0.25)  # 0.5 * 0.5


class TestOntologyNode:
    """Tests for OntologyNode dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        node = OntologyNode(tag="test")

        assert node.tag == "test"
        assert node.children == []
        assert node.description == ""

    def test_with_values(self):
        """Should accept all values."""
        node = OntologyNode(
            tag="ml",
            children=["ml/supervised"],
            description="Machine learning",
        )

        assert node.tag == "ml"
        assert node.children == ["ml/supervised"]
        assert node.description == "Machine learning"


class TestLoadOntology:
    """Tests for loading ontology from files."""

    def test_load_from_json_file(self):
        """Should load ontology from JSON file."""
        data = {
            "tags": {
                "test": {
                    "children": ["test/child"],
                    "description": "Test tag",
                },
            },
            "parent_weight": 0.8,
        }

        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            ontology = load_ontology(f.name)

        assert ontology.has_tag("test")
        assert ontology.has_tag("test/child")
        assert ontology.parent_weight == 0.8

    def test_load_flat_format(self):
        """Should load flat JSON format (without 'tags' wrapper)."""
        data = {
            "test": {
                "children": ["test/child"],
            },
        }

        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()

            ontology = load_ontology(f.name)

        assert ontology.has_tag("test")

    def test_load_nonexistent_file(self):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_ontology("/nonexistent/path.json")


class TestLoadDefaultOntology:
    """Tests for default ontology loading."""

    def test_returns_ontology(self):
        """Should return an Ontology instance."""
        ontology = load_default_ontology()

        assert isinstance(ontology, Ontology)

    def test_has_some_tags(self):
        """Default ontology should have some tags defined."""
        ontology = load_default_ontology()

        # At minimum, should have fallback tags
        assert len(ontology.all_tags()) > 0


class TestOntologyEdgeCases:
    """Edge case tests for Ontology."""

    def test_deep_hierarchy(self):
        """Should handle deeply nested hierarchies."""
        adjacency = {}
        prev_tag = "root"
        adjacency[prev_tag] = {"children": ["root/l1"]}

        for i in range(1, 10):
            tag = f"root/{'l' + str(i)}"
            next_tag = f"root/{'l' + str(i+1)}"
            adjacency[tag] = {"children": [next_tag]}
            prev_tag = tag

        ontology = Ontology(adjacency)

        # Should find all ancestors of deepest tag
        ancestors = ontology.get_ancestors("root/l10")
        assert len(ancestors) == 10

    def test_wide_hierarchy(self):
        """Should handle wide hierarchies (many children)."""
        children = [f"root/child{i}" for i in range(100)]
        ontology = Ontology({
            "root": {"children": children},
        })

        assert len(ontology.get_children("root")) == 100

    def test_tag_names_with_special_characters(self):
        """Should handle tags with special characters."""
        ontology = Ontology({
            "python-3": {
                "children": ["python-3/web-dev", "python-3/ml_ops"],
            },
        })

        assert ontology.has_tag("python-3")
        assert ontology.has_tag("python-3/web-dev")

        ancestors = ontology.get_ancestors("python-3/web-dev")
        assert ancestors == ["python-3"]

    def test_parent_weight_zero(self):
        """parent_weight=0 should not expand to ancestors."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=0.0)

        expanded = ontology.expand_for_matching(["ml/supervised"])

        assert expanded["ml/supervised"] == 1.0
        # Parent would have weight 0, should still be included but with 0
        assert expanded.get("ml", 0) == 0.0

    def test_parent_weight_one(self):
        """parent_weight=1.0 should give full weight to ancestors."""
        ontology = Ontology(SAMPLE_ADJACENCY, parent_weight=1.0)

        expanded = ontology.expand_for_matching(["ml/supervised"])

        assert expanded["ml/supervised"] == 1.0
        assert expanded["ml"] == 1.0
