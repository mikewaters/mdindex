"""Custom exceptions for PMD."""


class PMDError(Exception):
    """Base exception for all PMD errors."""

    pass


class DatabaseError(PMDError):
    """Database operation failed."""

    pass


class SourceCollectionError(PMDError):
    """Source collection operation failed."""

    pass


class SourceCollectionNotFoundError(SourceCollectionError):
    """Source collection does not exist."""

    pass


class SourceCollectionExistsError(SourceCollectionError):
    """Source collection already exists."""

    pass


# Backwards compatibility aliases (deprecated)
CollectionError = SourceCollectionError
CollectionNotFoundError = SourceCollectionNotFoundError
CollectionExistsError = SourceCollectionExistsError


class DocumentError(PMDError):
    """Document operation failed."""

    pass


class DocumentNotFoundError(DocumentError):
    """Document does not exist."""

    def __init__(self, path: str, suggestions: list[str] | None = None):
        """Initialize exception with path and suggestions.

        Args:
            path: Path to the document that was not found.
            suggestions: List of suggested alternative paths.
        """
        self.path = path
        self.suggestions = suggestions or []
        super().__init__(f"Document not found: {path}")


class LLMError(PMDError):
    """LLM operation failed."""

    pass


class ModelNotFoundError(LLMError):
    """Required model not available."""

    pass


class SearchError(PMDError):
    """Search operation failed."""

    pass


class EmbeddingError(PMDError):
    """Embedding generation failed."""

    pass


class FormatError(PMDError):
    """Output formatting failed."""

    pass


class VirtualPathError(PMDError):
    """Virtual path parsing/resolution failed."""

    pass
