import dspy
from agents.classification.signatures import ClassifyEntityInvolvement, ClassifyArticleCategory
from services.llm_service import LLMService
class AdverseMediaArticleClassifier(dspy.Module):
    """
    Module for analyzing adverse media mentions of entities.
    Combines entity involvement analysis and category classification
    using DSPy chain-of-thought reasoning.
    This module will be run per article.
    """
    def __init__(self, llm_service: LLMService, llm_role: str = "primary"):
        super().__init__()
        self.llm_service = llm_service
        self.llm_role = llm_role
        self.lm = llm_service.get_model(llm_role)
        
        with dspy.context(lm=self.lm):
            self.analyze_involvement = dspy.ChainOfThought(ClassifyEntityInvolvement)
            self.analyze_category = dspy.ChainOfThought(ClassifyArticleCategory)

    def forward(self, entity_name: str, article_text: str):
        with dspy.context(lm=self.lm):
            involvement_result = self.analyze_involvement(
                entity_name=entity_name,
                article_text=article_text
            )

            category_result = self.analyze_category(
                article_text=article_text,
                entity_name=entity_name,
                entity_involvement=involvement_result.entity_involvement
            )

            return dspy.Prediction(
                entity_involvement=involvement_result.entity_involvement,
                adverse_media_category=category_result.category,
                involvement_reasoning=involvement_result.reasoning,
                category_reasoning=category_result.reasoning,
                involvement_confidence=involvement_result.involvement_confidence,
                category_confidence=category_result.category_confidence
            )

