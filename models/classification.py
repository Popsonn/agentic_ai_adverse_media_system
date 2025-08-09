# adverse_media_system/models/classification.py

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from models.search import SearchResult


# --- Enums for Classification ---
class EntityInvolvementType(Enum):
    """
    Defines the type of involvement an entity has in an adverse media event.
    Matches the Literal used in ClassifyEntityInvolvement DSPy Signature.
    """
    PERPETRATOR = "perpetrator"
    VICTIM = "victim"
    NEUTRAL = "neutral"


class AdverseMediaCategory(Enum):
    """
    Defines the categories of adverse media events, aligned with FATF taxonomy.
    The values match the Literal strings used in ClassifyArticleCategory DSPy Signature.
    """
    FRAUD_FINANCIAL_CRIME = "Fraud & Financial Crime"
    CORRUPTION_BRIBERY = "Corruption & Bribery"
    ORGANIZED_CRIME = "Organized Crime"
    TERRORISM_EXTREMISM = "Terrorism & Extremism"
    SANCTIONS_EVASION = "Sanctions Evasion"
    OTHER_SERIOUS_CRIMES = "Other Serious Crimes"


# --- Data Models for Classification ---
@dataclass
class LLMClassificationDetails:
    """
    Represents the detailed output of a single LLM classification for an article.
    Field names match the actual implementation in ClassificationAgent.
    Types are enforced using the Enums for robustness.
    """
    entity_involvement: EntityInvolvementType
    adverse_media_category: AdverseMediaCategory
    involvement_reasoning: str
    category_reasoning: str
    involvement_confidence: float
    category_confidence: float
    #metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_involvement": self.entity_involvement.value, # Assuming it's an Enum
            "adverse_media_category": self.adverse_media_category.value, # Assuming it's an Enum
            "involvement_reasoning": self.involvement_reasoning,
            "category_reasoning": self.category_reasoning,
            "involvement_confidence": self.involvement_confidence,
            "category_confidence": self.category_confidence,
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMClassificationDetails':
        return cls(
        entity_involvement=EntityInvolvementType(data['entity_involvement']),
        adverse_media_category=AdverseMediaCategory(data['adverse_media_category']),
        involvement_reasoning=data['involvement_reasoning'],
        category_reasoning=data['category_reasoning'],
        involvement_confidence=data['involvement_confidence'],
        category_confidence=data['category_confidence']
    )


@dataclass
class ClassificationResult:
    """
    Represents the classification result for a single article.
    Matches the structure used in ClassificationAgent's classified_articles.
    Only includes fields that ClassificationAgent actually produces.
    """
    article_title: str
    article_link: str
    raw_article_text: str
    llm_1_classification: LLMClassificationDetails
    llm_2_classification: Optional[LLMClassificationDetails] = None
    has_conflict: bool = False  # Set by ClassificationAgent when LLMs disagree
    
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
    
    @property
    def is_adverse(self) -> bool:
        """Determine if the article represents adverse media."""
        return self.primary_involvement == EntityInvolvementType.PERPETRATOR


# --- Utility Functions ---
def convert_dict_to_classification_details(classification_dict: dict) -> LLMClassificationDetails:
    """
    Convert dictionary format from ClassificationAgent to LLMClassificationDetails.
    
    Args:
        classification_dict: Dictionary containing classification results
        
    Returns:
        LLMClassificationDetails: Structured classification data
    """
    return LLMClassificationDetails(
        entity_involvement=EntityInvolvementType(classification_dict["entity_involvement"]),
        adverse_media_category=AdverseMediaCategory(classification_dict["adverse_media_category"]),
        involvement_reasoning=classification_dict["involvement_reasoning"],
        category_reasoning=classification_dict["category_reasoning"],
        involvement_confidence=float(classification_dict["involvement_confidence"]),
        category_confidence=float(classification_dict["category_confidence"]),
        metadata=classification_dict.get("metadata", {})
    )


def convert_agent_result_to_classification_result(agent_result: dict) -> ClassificationResult:
    """
    Convert ClassificationAgent result dict to ClassificationResult object.
    
    Args:
        agent_result: Dictionary from ClassificationAgent.classified_articles
        
    Returns:
        ClassificationResult: Structured classification result
    """
    llm_1_details = convert_dict_to_classification_details(agent_result["llm_1_classification"])
    
    llm_2_details = None
    if agent_result.get("llm_2_classification"):
        llm_2_details = convert_dict_to_classification_details(agent_result["llm_2_classification"])
    
    return ClassificationResult(
        article_title=agent_result["article_title"],
        article_link=agent_result["article_link"],
        raw_article_text=agent_result["raw_article_text"],
        llm_1_classification=llm_1_details,
        llm_2_classification=llm_2_details,
        has_conflict=agent_result.get("has_conflict", False)
    )