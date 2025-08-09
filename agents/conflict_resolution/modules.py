# conflict_resolution/modules.py
from agents.conflict_resolution.signatures import ResolveConflictByReasoning
from models.classification import LLMClassificationDetails, AdverseMediaCategory, EntityInvolvementType
import dspy
from typing import Optional, Dict, Tuple

class DSPyConflictResolver(dspy.Module):
    """
    Enhanced DSPy module to resolve conflicts using LLM's reasoning abilities.
    Now supports external context integration for better resolution.
    """
    
    def __init__(self):
        super().__init__()
        self.resolve = dspy.ChainOfThought(ResolveConflictByReasoning)
        # Add external context analysis signature if available

    def forward(self, article_text: str, entity_name: str,
                llm_1_classification: LLMClassificationDetails,
                llm_2_classification: LLMClassificationDetails) -> Tuple[EntityInvolvementType, AdverseMediaCategory, str, float]:
        """
        Original forward method - maintains backward compatibility.
        """
        return self._resolve_conflict(
            article_text=article_text,
            entity_name=entity_name,
            llm_1_classification=llm_1_classification,
            llm_2_classification=llm_2_classification,
            external_context=None
        )

    def forward_with_context(self, article_text: str, entity_name: str,
                           llm_1_classification: LLMClassificationDetails,
                           llm_2_classification: LLMClassificationDetails,
                           external_context: Optional[Dict] = None) -> Tuple[EntityInvolvementType, AdverseMediaCategory, str, float]:
        """
        Enhanced forward method with external context support.
        
        Args:
            article_text: The full article text
            entity_name: Name of the entity being analyzed
            llm_1_classification: First LLM's classification details
            llm_2_classification: Second LLM's classification details
            external_context: Optional external context from search results
            
        Returns:
            Tuple of (resolved_involvement, resolved_category, explanation, confidence)
        """
        return self._resolve_conflict(
            article_text=article_text,
            entity_name=entity_name,
            llm_1_classification=llm_1_classification,
            llm_2_classification=llm_2_classification,
            external_context=external_context
        )

    def _resolve_conflict(self, article_text: str, entity_name: str,
                         llm_1_classification: LLMClassificationDetails,
                         llm_2_classification: LLMClassificationDetails,
                         external_context: Optional[Dict] = None) -> Tuple[EntityInvolvementType, AdverseMediaCategory, str, float]:
        """
        Internal method that handles the actual conflict resolution logic.
        """
        # Validate inputs
        if not llm_1_classification or not llm_2_classification:
            raise ValueError("Both LLM classifications must be provided")

        # Ensure we have all required fields
        required_fields = ['entity_involvement', 'adverse_media_category', 'involvement_reasoning',
                          'category_reasoning', 'involvement_confidence', 'category_confidence']

        for field in required_fields:
            if not hasattr(llm_1_classification, field) or not hasattr(llm_2_classification, field):
                raise ValueError(f"Missing required field: {field}")

        try:
            # Prepare external context string if available
            context_summary = ""
            if external_context and external_context.get("has_relevant_content"):
                context_summary = self._format_external_context(external_context)

            # Call the DSPy signature with enhanced context
            prediction = self.resolve(
                article_text=article_text,
                entity_name=entity_name,
                llm1_involvement=llm_1_classification.entity_involvement.value,
                llm1_involvement_reasoning=llm_1_classification.involvement_reasoning,
                llm1_category=llm_1_classification.adverse_media_category.value,
                llm1_category_reasoning=llm_1_classification.category_reasoning,
                llm1_involvement_confidence=llm_1_classification.involvement_confidence,
                llm1_category_confidence=llm_1_classification.category_confidence,
                llm2_involvement=llm_2_classification.entity_involvement.value,
                llm2_involvement_reasoning=llm_2_classification.involvement_reasoning,
                llm2_category=llm_2_classification.adverse_media_category.value,
                llm2_category_reasoning=llm_2_classification.category_reasoning,
                llm2_involvement_confidence=llm_2_classification.involvement_confidence,
                llm2_category_confidence=llm_2_classification.category_confidence,
                # Add context summary if available (you may need to update your signature)
                external_context_summary=context_summary
            )

            # Convert DSPy output (strings) back to our Enum types
            resolved_involvement = EntityInvolvementType(prediction.resolved_entity_involvement)
            resolved_category = AdverseMediaCategory(prediction.resolved_adverse_media_category)

            # Enhance explanation with context info if used
            explanation = prediction.resolution_explanation
            if external_context and external_context.get("has_relevant_content"):
                explanation += f" (Enhanced with external context from {external_context.get('relevant_sources', 0)} sources)"

            return resolved_involvement, resolved_category, explanation, prediction.resolution_confidence

        except ValueError as e:
            # Handle enum conversion errors
            raise ValueError(f"Error converting DSPy output to enum types: {e}")
        except Exception as e:
            # Handle other DSPy-related errors
            raise RuntimeError(f"DSPy resolution failed: {e}")

    def _format_external_context(self, external_context: Dict) -> str:
        """
        Format external context for inclusion in DSPy prompt.
        Keep it concise to avoid token limits.
        """
        if not external_context.get("has_relevant_content"):
            return ""

        relevant_content = external_context.get("relevant_content", [])
        if not relevant_content:
            return ""

        # Create a concise summary
        summary_parts = []
        for i, content in enumerate(relevant_content[:2]):  # Limit to top 2 sources
            source = content.get("source", "Unknown")
            title = content.get("title", "")[:100]  # Limit title length
            text_snippet = content.get("content", "")[:300]  # Limit content length
            
            summary_parts.append(f"Source {i+1} ({source}): {title}... {text_snippet}...")

        return f"External context: {' | '.join(summary_parts)}"