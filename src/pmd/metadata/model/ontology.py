"""Tag ontology for parent-child relationships.

Provides hierarchical tag matching where documents with child tags
can be boosted when queries match parent concepts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class OntologyNode:
    """A node in the tag ontology.

    Attributes:
        tag: The canonical tag (e.g., "#subject/ml").
        children: List of child tag names.
        description: Optional human-readable description.
    """

    tag: str
    children: list[str] = field(default_factory=list)
    description: str = ""


class Ontology:
    """Tag ontology with parent-child relationships.

    Provides methods to navigate the tag hierarchy:
    - get_ancestors: Walk up the hierarchy to find parent tags
    - get_children: Get direct children of a tag
    - expand_for_matching: Expand query tags to include ancestors with weights

    Example:
        ontology = Ontology({
            "ml": {"children": ["ml/supervised", "ml/unsupervised"]},
            "ml/supervised": {"children": ["ml/supervised/regression"]},
        })

        # Get ancestors of a leaf tag
        ontology.get_ancestors("ml/supervised/regression")
        # ["ml/supervised", "ml"]

        # Expand for matching (child-to-parent only)
        ontology.expand_for_matching(["ml/supervised/regression"])
        # {"ml/supervised/regression": 1.0, "ml/supervised": 0.7, "ml": 0.49}
    """

    def __init__(
        self,
        adjacency: dict[str, dict[str, Any]],
        *,
        parent_weight: float = 0.7,
    ):
        """Initialize the ontology.

        Args:
            adjacency: Tag hierarchy as adjacency list.
                Keys are tag names, values are dicts with:
                - "children": list of child tag names
                - "description": optional description
            parent_weight: Weight multiplier for each level of ancestry.
                Default 0.7 means grandparents get 0.49 weight.
        """
        self.adjacency = adjacency
        self.parent_weight = parent_weight
        self._parent_map = self._build_parent_map()

    def _build_parent_map(self) -> dict[str, str]:
        """Build reverse mapping from child to parent.

        Returns:
            Dictionary mapping each child tag to its parent tag.
        """
        parent_map: dict[str, str] = {}
        for parent, data in self.adjacency.items():
            children = data.get("children", [])
            for child in children:
                parent_map[child] = parent
        return parent_map

    def get_ancestors(self, tag: str, max_hops: int = 10) -> list[str]:
        """Get parent tags up to max_hops levels.

        Args:
            tag: The tag to find ancestors for.
            max_hops: Maximum number of parent levels to traverse.

        Returns:
            List of ancestor tags from closest to farthest.

        Example:
            ontology.get_ancestors("ml/supervised/regression", max_hops=2)
            # ["ml/supervised", "ml"]
        """
        ancestors: list[str] = []
        current = tag

        for _ in range(max_hops):
            parent = self._parent_map.get(current)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break

        return ancestors

    def get_children(self, tag: str) -> list[str]:
        """Get direct children of a tag.

        Args:
            tag: The parent tag.

        Returns:
            List of direct child tags.
        """
        data = self.adjacency.get(tag, {})
        return data.get("children", [])

    def get_descendants(self, tag: str, max_depth: int = 10) -> list[str]:
        """Get all descendants of a tag (children, grandchildren, etc.).

        Args:
            tag: The parent tag.
            max_depth: Maximum depth to traverse.

        Returns:
            List of all descendant tags.
        """
        descendants: list[str] = []
        to_visit = self.get_children(tag)
        depth = 0

        while to_visit and depth < max_depth:
            next_level = []
            for child in to_visit:
                descendants.append(child)
                next_level.extend(self.get_children(child))
            to_visit = next_level
            depth += 1

        return descendants

    def has_tag(self, tag: str) -> bool:
        """Check if a tag exists in the ontology.

        Args:
            tag: Tag to check.

        Returns:
            True if the tag is known (as parent or child).
        """
        return tag in self.adjacency or tag in self._parent_map

    def get_description(self, tag: str) -> str | None:
        """Get the description for a tag.

        Args:
            tag: The tag.

        Returns:
            Description if available, None otherwise.
        """
        data = self.adjacency.get(tag, {})
        return data.get("description")

    def expand_for_matching(
        self,
        query_tags: list[str] | set[str],
        max_hops: int = 2,
    ) -> dict[str, float]:
        """Expand query tags for document matching.

        For each query tag, includes ancestors with progressively
        reduced weights. This allows documents with child tags to
        be boosted when queries match parent concepts.

        Note: We do NOT expand parent-to-children. If query matches
        "ml", we don't automatically boost "ml/supervised" documents.
        This prevents precision loss.

        Args:
            query_tags: Tags inferred from the query.
            max_hops: Maximum ancestor levels to include.

        Returns:
            Dictionary mapping tag -> weight, where weight < 1.0
            for expanded ancestor tags.

        Example:
            expand_for_matching(["ml/supervised/regression"])
            # {
            #     "ml/supervised/regression": 1.0,  # Exact match
            #     "ml/supervised": 0.7,             # Parent
            #     "ml": 0.49,                       # Grandparent
            # }
        """
        expanded: dict[str, float] = {}

        for tag in query_tags:
            # Exact match gets full weight
            if tag not in expanded or expanded[tag] < 1.0:
                expanded[tag] = 1.0

            # Add ancestors with progressively reduced weight
            weight = self.parent_weight
            for ancestor in self.get_ancestors(tag, max_hops=max_hops):
                # Only update if not already present with higher weight
                if ancestor not in expanded or expanded[ancestor] < weight:
                    expanded[ancestor] = weight
                weight *= self.parent_weight

        return expanded

    def all_tags(self) -> set[str]:
        """Get all tags in the ontology.

        Returns:
            Set of all known tags (parents and children).
        """
        tags = set(self.adjacency.keys())
        tags.update(self._parent_map.keys())
        return tags


def load_ontology(path: Path | str) -> Ontology:
    """Load ontology from a JSON file.

    Args:
        path: Path to the ontology JSON file.

    Returns:
        Configured Ontology instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    adjacency = data.get("tags", data)  # Support both wrapped and flat formats
    parent_weight = data.get("parent_weight", 0.7)

    return Ontology(adjacency, parent_weight=parent_weight)


def load_default_ontology() -> Ontology:
    """Load the default tag ontology from package data.

    Returns:
        Ontology instance with default hierarchy.
    """
    data_dir = Path(__file__).parent / "data"
    ontology_file = data_dir / "tag_ontology.json"

    if ontology_file.exists():
        return load_ontology(ontology_file)

    # Fallback to minimal built-in ontology
    return Ontology({
        "python": {
            "children": ["python/web", "python/ml", "python/testing"],
            "description": "Python programming language",
        },
        "machine-learning": {
            "children": ["machine-learning/supervised", "machine-learning/unsupervised"],
            "description": "Machine learning and AI",
        },
        "machine-learning/supervised": {
            "children": ["machine-learning/supervised/classification", "machine-learning/supervised/regression"],
            "description": "Supervised learning methods",
        },
    })
