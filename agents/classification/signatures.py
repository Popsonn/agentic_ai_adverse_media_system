# adverse_media_system/agents/classification/signatures.py

import dspy
from typing import Literal
from dspy.signatures import InputField, OutputField

class ClassifyEntityInvolvement(dspy.Signature):
    """Determine if the entity is a perpetrator, victim, or neutral in the news article."""
    entity_name: str = InputField(desc="The name of the entity (person or company)")
    article_text: str = InputField(desc="The full text of the news article")
    entity_involvement: Literal['perpetrator', 'victim', 'neutral'] = OutputField(
        desc="Classification of the entity's involvement in the article")
    reasoning: str = OutputField(desc="Explanation of why this classification was chosen")
    involvement_confidence: float = OutputField(
        desc="Confidence score (0.0 to 1.0) for the entity involvement classification. 1.0 is highly confident, 0.0 is completely uncertain."
    )

class ClassifyArticleCategory(dspy.Signature):
    """Classify the news article into one of the FATF-aligned Adverse Media Taxonomy categories."""
    article_text: str = InputField(desc="The full text of the news article")
    entity_name: str = InputField(desc="The name of the entity (person or company)")
    entity_involvement: Literal['perpetrator', 'victim', 'neutral'] = InputField(
        desc="Classification of the entity's involvement in the article")
    category: Literal[
        'Fraud & Financial Crime',
        'Corruption & Bribery',
        'Organized Crime',
        'Terrorism & Extremism',
        'Sanctions Evasion',
        'Other Serious Crimes'
    ] = OutputField(desc="The FATF category that best fits the article")
    reasoning: str = OutputField(desc="Explanation of why this classification was chosen")
    category_confidence: float = OutputField(
        desc="Confidence score (0.0 to 1.0) for the adverse media category classification. 1.0 is highly confident, 0.0 is completely uncertain."
    )