from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
# Import necessary Enums and other models
from models.search import SearchResult
from models.entity import DisambiguationResult 
from models.classification import (
    AdverseMediaCategory,
    EntityInvolvementType,
    LLMClassificationDetails 
)
from models.enums import ConflictSeverity, ResolutionMethod

@dataclass
class ClassifiedArticleResult:
    """
    Represents a fully classified article with primary and optional secondary LLM classifications.
    This is the output format expected by the ConflictResolutionAgent.
    """
    # Basic article information
    article_title: str
    article_link: str
    raw_article_text: str
    entity_name: str  
    
    # Primary LLM classification (always present)
    llm_1_classification: LLMClassificationDetails
    
    # Secondary LLM classification (present if secondary LLM was used)
    llm_2_classification: Optional[LLMClassificationDetails] = None
    
    # Conflict detection information
    has_conflict: bool = False
    conflict_type: Optional[str] = None  # "disagreement", "low_confidence", or "none"
    
    # Confidence metrics for conflict analysis
    min_involvement_confidence: Optional[float] = None
    min_category_confidence: Optional[float] = None
    
    # Detailed confidence information
    confidence_details: Optional[Dict[str, float]] = None
    
    # Processing metadata
    classified_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "article_title": self.article_title,
            "article_link": self.article_link,
            "raw_article_text": self.raw_article_text,
            "entity_name": self.entity_name,
            "llm_1_classification": self.llm_1_classification.to_dict(),
            "llm_2_classification": self.llm_2_classification.to_dict() if self.llm_2_classification else None,
            "has_conflict": self.has_conflict,
            "conflict_type": self.conflict_type,
            "min_involvement_confidence": self.min_involvement_confidence,
            "min_category_confidence": self.min_category_confidence,
            "confidence_details": self.confidence_details,
            "classified_at": self.classified_at.isoformat(),
            "metadata": self.metadata,
            # Include computed properties for convenience
            "primary_involvement": self.primary_involvement.value,
            "primary_category": self.primary_category.value,
            "average_involvement_confidence": self.average_involvement_confidence,
            "average_category_confidence": self.average_category_confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassifiedArticleResult':
        return cls(
        article_title=data['article_title'],
        article_link=data['article_link'],
        raw_article_text=data['raw_article_text'],
        entity_name=data['entity_name'],
        llm_1_classification=LLMClassificationDetails.from_dict(data['llm_1_classification']),
        llm_2_classification=LLMClassificationDetails.from_dict(data['llm_2_classification']) if data.get('llm_2_classification') else None,
        has_conflict=data.get('has_conflict', False),
        conflict_type=data.get('conflict_type'),
        min_involvement_confidence=data.get('min_involvement_confidence'),
        min_category_confidence=data.get('min_category_confidence'),
        confidence_details=data.get('confidence_details'),
        classified_at=datetime.fromisoformat(data['classified_at']) if data.get('classified_at') else None,
        metadata=data.get('metadata')
    )


    
    def __post_init__(self):
        """Set default values after initialization"""
        if self.classified_at is None:
            self.classified_at = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

    # CORRECTLY INDENTED PROPERTIES START HERE
    @property
    def primary_involvement(self) -> EntityInvolvementType:
        """Get the primary entity involvement classification."""
        return self.llm_1_classification.entity_involvement
        
    @property
    def primary_category(self) -> AdverseMediaCategory:
        """Get the primary adverse media category."""
        return self.llm_1_classification.adverse_media_category
        
    @property
    def average_involvement_confidence(self) -> float:
        """Calculate average involvement confidence across LLMs."""
        if self.llm_2_classification:
            return (self.llm_1_classification.involvement_confidence + 
                    self.llm_2_classification.involvement_confidence) / 2
        return self.llm_1_classification.involvement_confidence
        
    @property
    def average_category_confidence(self) -> float:
        """Calculate average category confidence across LLMs."""
        if self.llm_2_classification:
            return (self.llm_1_classification.category_confidence + 
                    self.llm_2_classification.category_confidence) / 2
        return self.llm_1_classification.category_confidence

# Conversion function to transform ClassificationAgent dict output to dataclass
def dict_to_classified_article_result(classification_dict: Dict, entity_name: str) -> ClassifiedArticleResult:
    """
    Convert the dictionary output from ClassificationAgent to ClassifiedArticleResult dataclass.
    
    Args:
        classification_dict: The dict returned by ClassificationAgent._classify_single_article
        entity_name: The entity name being analyzed
    
    Returns:
        ClassifiedArticleResult dataclass instance
    """
    
    # Convert primary LLM classification dict to dataclass
    llm_1_dict = classification_dict["llm_1_classification"]
    llm_1_classification = LLMClassificationDetails(
        entity_involvement=EntityInvolvementType(llm_1_dict["entity_involvement"]),
        adverse_media_category=AdverseMediaCategory(llm_1_dict["adverse_media_category"]),
        involvement_reasoning=llm_1_dict["involvement_reasoning"],
        category_reasoning=llm_1_dict["category_reasoning"],
        involvement_confidence=llm_1_dict["involvement_confidence"],
        category_confidence=llm_1_dict["category_confidence"]
    )
    
    # Convert secondary LLM classification if present
    llm_2_classification = None
    if "llm_2_classification" in classification_dict and classification_dict["llm_2_classification"]:
        llm_2_dict = classification_dict["llm_2_classification"]
        llm_2_classification = LLMClassificationDetails(
            entity_involvement=EntityInvolvementType(llm_2_dict["entity_involvement"]),
            adverse_media_category=AdverseMediaCategory(llm_2_dict["adverse_media_category"]),
            involvement_reasoning=llm_2_dict["involvement_reasoning"],
            category_reasoning=llm_2_dict["category_reasoning"],
            involvement_confidence=llm_2_dict["involvement_confidence"],
            category_confidence=llm_2_dict["category_confidence"]
        )
    
    return ClassifiedArticleResult(
        article_title=classification_dict["article_title"],
        article_link=classification_dict["article_link"],
        raw_article_text=classification_dict["raw_article_text"],
        entity_name=entity_name,
        llm_1_classification=llm_1_classification,
        llm_2_classification=llm_2_classification,
        has_conflict=classification_dict.get("has_conflict", False),
        conflict_type=classification_dict.get("conflict_type"),
        min_involvement_confidence=classification_dict.get("min_involvement_confidence"),
        min_category_confidence=classification_dict.get("min_category_confidence"),
        confidence_details=classification_dict.get("confidence_details"),
        metadata=classification_dict.get("metadata", {})
    )

@dataclass
class ResolvedArticleResult: # Renamed from ResolvedArticleOutput to match our established convention
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

    @classmethod
    def from_resolution(cls, classified_article: ClassifiedArticleResult, resolution_data: Dict[str, Any]) -> 'ResolvedArticleResult':
        """
        Create a ResolvedArticleResult from conflict resolution process.
        
        Args:
            classified_article: The original classified article
            resolution_data: Dictionary containing resolution details
        
        Returns:
            ResolvedArticleResult instance
        """
        return cls(
            classified_article=classified_article,
            final_entity_involvement=EntityInvolvementType(resolution_data['final_entity_involvement']),
            final_adverse_category=AdverseMediaCategory(resolution_data['final_adverse_category']),
            final_overall_confidence=resolution_data['final_overall_confidence'],
            resolution_method=ResolutionMethod(resolution_data.get('resolution_method', ResolutionMethod.NO_CONFLICT.value)),
            conflict_details=resolution_data.get('conflict_details'),
            resolution_reasoning=resolution_data.get('resolution_reasoning'),
            conflict_severity=ConflictSeverity(resolution_data['conflict_severity']) if resolution_data.get('conflict_severity') else None,
            is_deemed_adverse=resolution_data.get('is_deemed_adverse', False),
            requires_further_human_review=resolution_data.get('requires_further_human_review', False),
            further_review_reason=resolution_data.get('further_review_reason'),
            external_search_context=resolution_data.get('external_search_context', []),
            metadata=resolution_data.get('metadata', {})
        )
    



