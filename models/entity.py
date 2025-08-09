# adverse_media_system/models/entity.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal

@dataclass
class EntityContext:
    """
    Represents structured context extracted from user input about an entity.
    Enhanced for Nigerian/African KYC with realistic web-searchable fields.
    """
    # Existing core fields (keep as-is)
    persons: List[str] = field(default_factory=list)
    organizations: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    
    # NEW: Nigerian/African-focused KYC fields
    aliases: List[str] = field(default_factory=list)  # "Wizkid", "OBO", stage names
    
    # Keep existing utility fields
    additional_context: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class EntityCandidate:
    """
    Represents a single candidate entity found during the disambiguation process.
    This candidate is typically found from search results or internal databases.
    """
    name: str
    confidence_score: float  # Changed from 'confidence' to match agent usage
    context_match: Dict[str, float] = field(default_factory=dict)  # Changed from 'context_match_scores' to match agent usage
    description: str = ""  # Changed from Optional[str] to str with default to match agent usage
    aliases: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Changed from 'Dict' to 'Dict[str, Any]'
    source_url: str = ""  # Changed from Optional[str] to str with default to match agent usage
    search_snippet: str = ""  # Changed from Optional[str] to str with default to match agent usage
    # Removed entity_id as it's not used in the agent
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "confidence_score": self.confidence_score,
            "context_match": self.context_match,
            "description": self.description,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "source_url": self.source_url,
            "search_snippet": self.search_snippet,
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityCandidate':
        return cls(
        name=data['name'],
        confidence_score=data['confidence_score'],
        context_match=data.get('context_match', {}),
        description=data.get('description', ''),
        aliases=data.get('aliases', []),
        metadata=data.get('metadata', {}),
        source_url=data.get('source_url', ''),
        search_snippet=data.get('search_snippet', ''),
    )
    
@dataclass
class DisambiguationResult:
    """
    The final output of the Entity Disambiguation Agent, detailing the resolution.
    """
    status: Literal["resolved", "needs_review", "no_matches", "error"]
    resolved_entity: Optional[EntityCandidate] = None
    all_candidates: List[EntityCandidate] = field(default_factory=list)
    confidence_score: float = 0.0  # Changed from 'overall_confidence' to match agent usage
    reasoning: str = ""
    review_reason: str = ""  # Changed from Optional[str] to str with default to match agent usage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "resolved_entity": self.resolved_entity.to_dict() if self.resolved_entity else None, # Calls to_dict on EntityCandidate
            "all_candidates": [candidate.to_dict() for candidate in self.all_candidates], # Calls to_dict on EntityCandidate
            "confidence_score": self.confidence_score,
            "reasoning": self.reasoning,
            "review_reason": self.review_reason,
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DisambiguationResult':
        return cls(
        status=data['status'],
        resolved_entity=EntityCandidate.from_dict(data['resolved_entity']) if data.get('resolved_entity') else None,
        all_candidates=[EntityCandidate.from_dict(candidate) for candidate in data.get('all_candidates', [])],
        confidence_score=data.get('confidence_score', 0.0),
        reasoning=data.get('reasoning', ''),
        review_reason=data.get('review_reason', ''),
    )
