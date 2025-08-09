# adverse_media_system/models/conflict.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from datetime import datetime # Added for the resolved_at timestamp

# Import necessary enums and dataclasses from previous models
from models.classification import (
    AdverseMediaCategory,
    EntityInvolvementType)

from models.results import ClassifiedArticleResult # The input to conflict resolution
from models.search import SearchResult
from models.enums import ConflictSeverity, ResolutionMethod

from datetime import datetime
from typing import Dict, List, Optional, Any
from models.classification import EntityInvolvementType, AdverseMediaCategory




# --- Enums for Conflict Resolution (Based directly on your agent's logic) ---


# --- Data Models for Conflict Resolution ---
@dataclass
class ResolvedArticleResult:
    """
    Represents the final, reconciled classification for a single article
    after potential conflicts have been resolved by the ConflictResolutionAgent.
    It builds upon the ClassifiedArticleResult.
    """
    # Contains the original classified data for full context
    classified_article: ClassifiedArticleResult

    # Final reconciled classifications
    final_entity_involvement: EntityInvolvementType
    final_adverse_category: AdverseMediaCategory
    final_overall_confidence: float # Overall confidence after resolution

    # Details about the conflict resolution process
    resolution_method: ResolutionMethod = ResolutionMethod.NO_CONFLICT
    conflict_details: Optional[str] = None # Description of the conflict (if any)
    resolution_reasoning: Optional[str] = None # Explanation of how the conflict was resolved
    conflict_severity: Optional[ConflictSeverity] = None # Added based on your agent's logic

    # Final system-level flags after resolution
    is_deemed_adverse: bool = False # True if the article is ultimately considered adverse after resolution
    requires_further_human_review: bool = False # If resolution wasn't fully automated or confident enough
    further_review_reason: Optional[str] = None # Reason for further human review

    # Timestamp of when this resolution occurred
    resolved_at: datetime = field(default_factory=datetime.utcnow) # Changed to datetime object for consistency

    # Optional: Additional context from external search, if used for human review
    # This stores the *search results* which can be our SearchResult dataclass, not raw dicts
    external_search_context: List[SearchResult] = field(default_factory=list)

    # General metadata for resolution process
    metadata: Dict[str, Any] = field(default_factory=dict)

    # PROPERLY INDENTED CLASS METHOD:
    @classmethod
    def from_resolution(
        cls,
        classified_article: ClassifiedArticleResult,
        resolution_method: ResolutionMethod,
        resolution_details: Optional[Dict] = None
    ) -> 'ResolvedArticleResult':
        """Factory method to create ResolvedArticleResult from resolution process"""
        if resolution_details is None:
            resolution_details = {}
            
        final_involvement: EntityInvolvementType = EntityInvolvementType.NEUTRAL # Sensible default
        final_category: AdverseMediaCategory = AdverseMediaCategory.NOT_ADVERSE # Fixed: not Optional
        final_overall_confidence: float = 0.0
        
        is_deemed_adverse: bool = False
        requires_further_human_review: bool = False
        further_review_reason: Optional[str] = None
        conflict_severity: Optional[ConflictSeverity] = None
        external_search_context: List[SearchResult] = []
        
        # Extract initial LLM classifications (already type-safe)
        llm_1 = classified_article.llm_1_classification
        llm_2 = classified_article.llm_2_classification # Could be None if no conflict

        # Determine final classification based on resolution method
        if resolution_method == ResolutionMethod.NO_CONFLICT:
            # LLMs agreed and confidence was sufficient
            final_involvement = llm_1.entity_involvement
            final_category = llm_1.adverse_media_category
            final_overall_confidence = max(llm_1.involvement_confidence, llm_1.category_confidence)
            is_deemed_adverse = (final_category != AdverseMediaCategory.NOT_ADVERSE)

        elif resolution_method == ResolutionMethod.HUMAN_REVIEW_REQUIRED:
            # Automated resolution failed, requires human input
            requires_further_human_review = True
            further_review_reason = resolution_details.get("reason", "Critical conflict or complex issue requires human review.")
            conflict_severity = ConflictSeverity(resolution_details.get("conflict_severity", ConflictSeverity.HIGH.value))
            
            # If external search context was provided, include it
            if "external_search_results" in resolution_details:
                external_search_context = [SearchResult(**sr_dict) for sr_dict in resolution_details["external_search_results"]]

        else: # CONFIDENCE_BASED, REASONING_ARBITRATION, CONSERVATIVE_DEFAULT, EXTERNAL_SEARCH_CONTEXT, LOW_CONFIDENCE_AGREEMENT
            # These methods provide a specific resolved classification
            final_involvement_str = resolution_details.get("final_involvement")
            final_category_str = resolution_details.get("final_category")
            
            if final_involvement_str:
                final_involvement = EntityInvolvementType(final_involvement_str)
            if final_category_str:
                final_category = AdverseMediaCategory(final_category_str)
            
            final_overall_confidence = resolution_details.get("confidence", 0.0)
            
            # For rule-based methods (if not already set by reasoning_arbitration)
            if resolution_method in [ResolutionMethod.CONFIDENCE_BASED, ResolutionMethod.CONSERVATIVE_DEFAULT] and final_overall_confidence == 0.0:
                if llm_1 and llm_2: # Ensure LLMs exist
                    final_overall_confidence = max(llm_1.involvement_confidence, llm_2.involvement_confidence,
                                                   llm_1.category_confidence, llm_2.category_confidence)
                elif llm_1: # If only one LLM classification is available
                    final_overall_confidence = max(llm_1.involvement_confidence, llm_1.category_confidence)

            is_deemed_adverse = (final_category != AdverseMediaCategory.NOT_ADVERSE)
            conflict_severity = ConflictSeverity(resolution_details.get("conflict_severity", ConflictSeverity.LOW.value))

        # Instantiate and return the ResolvedArticleResult dataclass
        return cls(
            classified_article=classified_article,
            final_entity_involvement=final_involvement,
            final_adverse_category=final_category,
            final_overall_confidence=final_overall_confidence,
            resolution_method=resolution_method,
            conflict_details=resolution_details.get("explanation", None) or resolution_details.get("reason", None),
            resolution_reasoning=resolution_details.get("explanation", None),
            is_deemed_adverse=is_deemed_adverse,
            requires_further_human_review=requires_further_human_review,
            further_review_reason=further_review_reason,
            conflict_severity=conflict_severity,
            external_search_context=external_search_context,
            metadata=resolution_details # Store full resolution_details in metadata for debugging/auditing
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "classified_article": self.classified_article.to_dict(),
            "final_entity_involvement": self.final_entity_involvement.value,
            "final_adverse_category": self.final_adverse_category.value,
            "final_overall_confidence": self.final_overall_confidence,
            "resolution_method": self.resolution_method.value,
            "conflict_details": self.conflict_details,
            "resolution_reasoning": self.resolution_reasoning,
            "conflict_severity": self.conflict_severity.value if self.conflict_severity else None,
            "is_deemed_adverse": self.is_deemed_adverse,
            "requires_further_human_review": self.requires_further_human_review,
            "further_review_reason": self.further_review_reason,
            "resolved_at": self.resolved_at.isoformat(),
            "external_search_context": [result.to_dict() for result in self.external_search_context],
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResolvedArticleResult':
        return cls(
            classified_article=ClassifiedArticleResult.from_dict(data['classified_article']),
            final_entity_involvement=EntityInvolvementType(data['final_entity_involvement']),
            final_adverse_category=AdverseMediaCategory(data['final_adverse_category']),
            final_overall_confidence=data['final_overall_confidence'],
            resolution_method=ResolutionMethod(data.get('resolution_method', ResolutionMethod.NO_CONFLICT.value)),
            conflict_details=data.get('conflict_details'),
            resolution_reasoning=data.get('resolution_reasoning'),
            conflict_severity=ConflictSeverity(data['conflict_severity']) if data.get('conflict_severity') else None,
            is_deemed_adverse=data.get('is_deemed_adverse', False),
            requires_further_human_review=data.get('requires_further_human_review', False),
            further_review_reason=data.get('further_review_reason'),
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else datetime.utcnow(),
            external_search_context=[SearchResult.from_dict(result) for result in data.get('external_search_context', [])],
            metadata=data.get('metadata', {})
        )