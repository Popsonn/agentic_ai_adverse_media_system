# adverse_media_system/agents/conflict_resolution/signatures.py

import dspy
from typing import Literal
from dspy.signatures import InputField, OutputField

from models.classification import EntityInvolvementType, AdverseMediaCategory, LLMClassificationDetails


class ResolveConflictByReasoning(dspy.Signature):
    """
    Given an article and two conflicting classifications with their reasoning,
    determine the more accurate classification and explain why.
    Now supports optional external context for enhanced resolution.
    """
    # Article and entity context
    article_text: str = InputField(desc="The full text of the news article")
    entity_name: str = InputField(desc="The name of the entity mentioned in the article")

    # LLM 1 classification
    llm1_involvement: str = InputField(desc="LLM 1's entity involvement classification")
    llm1_involvement_reasoning: str = InputField(desc="LLM 1's reasoning for the involvement classification")
    llm1_category: str = InputField(desc="LLM 1's adverse media category classification")
    llm1_category_reasoning: str = InputField(desc="LLM 1's reasoning for the category classification")
    llm1_involvement_confidence: float = InputField(desc="LLM 1's confidence score for involvement (0.0 - 1.0)")
    llm1_category_confidence: float = InputField(desc="LLM 1's confidence score for category (0.0 - 1.0)")

    # LLM 2 classification
    llm2_involvement: str = InputField(desc="LLM 2's entity involvement classification")
    llm2_involvement_reasoning: str = InputField(desc="LLM 2's reasoning for the involvement classification")
    llm2_category: str = InputField(desc="LLM 2's adverse media category classification")
    llm2_category_reasoning: str = InputField(desc="LLM 2's reasoning for the category classification")
    llm2_involvement_confidence: float = InputField(desc="LLM 2's confidence score for involvement (0.0 - 1.0)")
    llm2_category_confidence: float = InputField(desc="LLM 2's confidence score for category (0.0 - 1.0)")

    # NEW: Optional external context - this makes your enhanced resolver work
    external_context_summary: str = InputField(
        desc="Summary of external search context and relevant information (optional)", 
        default=""
    )

    # Resolution outputs - Updated to match your enum values exactly
    resolved_entity_involvement: Literal[
        "PERPETRATOR",  # EntityInvolvementType.PERPETRATOR.value
        "VICTIM",       # EntityInvolvementType.VICTIM.value
        "NEUTRAL"       # EntityInvolvementType.NEUTRAL.value
    ] = OutputField(desc="Final resolved classification for entity involvement")

    resolved_adverse_media_category: Literal[
        "FRAUD_FINANCIAL_CRIME",    # AdverseMediaCategory.FRAUD_FINANCIAL_CRIME.value
        "CORRUPTION_BRIBERY",       # AdverseMediaCategory.CORRUPTION_BRIBERY.value
        "ORGANIZED_CRIME",          # AdverseMediaCategory.ORGANIZED_CRIME.value
        "TERRORISM_EXTREMISM",      # AdverseMediaCategory.TERRORISM_EXTREMISM.value
        "SANCTIONS_EVASION",        # AdverseMediaCategory.SANCTIONS_EVASION.value
        "OTHER_SERIOUS_CRIMES"
    ] = OutputField(desc="Final resolved classification for adverse media category")

    resolution_explanation: str = OutputField(desc="Explanation of how the final decision was reached, including any external context considered")
    resolution_confidence: float = OutputField(desc="Confidence score (0.0 to 1.0) in the final resolution")