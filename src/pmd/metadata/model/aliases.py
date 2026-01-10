"""Tag alias management for metadata normalization.

Provides a centralized system for tag aliases, used both during
extraction (optional) and query-time inference.
"""

from __future__ import annotations

import json
from pathlib import Path


class TagAliases:
    """Tag alias registry for normalizing tags.

    Maps alias terms to their canonical tag names. This supports
    both query-time inference (matching "py" to "python") and
    optional extraction-time normalization.

    Example:
        aliases = TagAliases()
        aliases.register("py", "python")
        aliases.register("js", "javascript")

        aliases.resolve("py")  # "python"
        aliases.resolve("python")  # "python" (passthrough)
        aliases.resolve("unknown")  # "unknown" (passthrough)
    """

    def __init__(self) -> None:
        """Initialize empty alias registry."""
        self._aliases: dict[str, str] = {}

    def register(self, alias: str, canonical: str) -> None:
        """Register an alias for a canonical tag.

        Args:
            alias: The alias term (e.g., "py").
            canonical: The canonical tag it maps to (e.g., "python").
        """
        self._aliases[alias.lower()] = canonical.lower()

    def register_many(self, aliases: dict[str, str]) -> None:
        """Register multiple aliases at once.

        Args:
            aliases: Dictionary mapping alias -> canonical tag.
        """
        for alias, canonical in aliases.items():
            self.register(alias, canonical)

    def resolve(self, tag: str) -> str:
        """Resolve a tag to its canonical form.

        If the tag is a known alias, returns the canonical tag.
        Otherwise returns the tag unchanged.

        Args:
            tag: Tag to resolve.

        Returns:
            Canonical tag name.
        """
        return self._aliases.get(tag.lower(), tag.lower())

    def is_alias(self, term: str) -> bool:
        """Check if a term is a registered alias.

        Args:
            term: Term to check.

        Returns:
            True if the term is a known alias.
        """
        return term.lower() in self._aliases

    def get_canonical(self, alias: str) -> str | None:
        """Get the canonical tag for an alias.

        Args:
            alias: The alias to look up.

        Returns:
            Canonical tag if alias exists, None otherwise.
        """
        return self._aliases.get(alias.lower())

    def all_aliases(self) -> dict[str, str]:
        """Get all registered aliases.

        Returns:
            Copy of the alias dictionary.
        """
        return dict(self._aliases)

    def clear(self) -> None:
        """Clear all registered aliases."""
        self._aliases.clear()


def load_aliases(path: Path | str) -> TagAliases:
    """Load aliases from a JSON file.

    Expected format:
    {
        "aliases": {
            "py": "python",
            "js": "javascript",
            ...
        }
    }

    Args:
        path: Path to the aliases JSON file.

    Returns:
        Configured TagAliases instance.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

    aliases = TagAliases()
    alias_dict = data.get("aliases", {})
    aliases.register_many(alias_dict)

    return aliases


def load_default_aliases() -> TagAliases:
    """Load default tag aliases from package data.

    Returns:
        TagAliases instance with default mappings.
    """
    data_dir = Path(__file__).parent / "data"
    aliases_file = data_dir / "tag_aliases.json"

    if aliases_file.exists():
        return load_aliases(aliases_file)

    # Fallback to minimal built-in aliases
    aliases = TagAliases()
    aliases.register_many({
        "py": "python",
        "js": "javascript",
        "ts": "typescript",
        "api": "api",
        "db": "database",
        "ml": "machine-learning",
        "k8s": "kubernetes",
        "doc": "documentation",
        "docs": "documentation",
    })
    return aliases
