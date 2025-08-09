"""Data models for the adverse media system."""

# Expose all model classes for easier imports
from .entity import EntityContext, EntityCandidate, DisambiguationResult
from .search import SearchResult
from .classification import LLMClassificationDetails
from .conflict import ConflictSeverity, ResolutionMethod
from .results import ClassifiedArticleResult, ResolvedArticleResult

# Now you can do: from adverse_media_system.models import EntityCandidate