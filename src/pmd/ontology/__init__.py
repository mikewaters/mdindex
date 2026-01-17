"""Tag ontology and matching module.

Provides:
- Ontology: Tag hierarchy management
- TagAliases: Tag alias management
- LexicalTagMatcher: Tag matching using ontology
- OntologyMetadataBooster: Search result boosting based on tag metadata
"""

from pmd.ontology.model import Ontology, OntologyNode
from pmd.ontology.aliases import TagAliases
from pmd.ontology.inference import LexicalTagMatcher
from pmd.ontology.retrieval import TagRetriever
from pmd.ontology.scoring import ScoredResult

__all__ = [
    "Ontology",
    "TagNode",
    "TagAliases",
    "LexicalTagMatcher",
    "TagRetriever",
    "ScoredResult",
]
