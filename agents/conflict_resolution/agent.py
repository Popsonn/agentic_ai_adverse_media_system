# Enhanced conflict_resolution_agent.py with better BraveSearchService integration

import dspy
import os
import requests
import logging
from typing import Dict, List, Literal, Optional, Any
from enum import Enum
from datetime import datetime
from agents.conflict_resolution.signatures import ResolveConflictByReasoning
from agents.conflict_resolution.modules import DSPyConflictResolver

# Import base classes
from core.base_agent import BaseDSPyAgent, AgentMetrics
from config.settings import BaseAgentConfig
from services.llm_service import LLMService
from services.brave_search_service import BraveSearchService
from config.settings import ConflictResolutionConfig
# Import all necessary models and state
from core.state import AdverseMediaState, AgentStatus
from models.classification import (
    AdverseMediaCategory,
    EntityInvolvementType,
    LLMClassificationDetails
)

from models.enums import (
    ResolutionMethod,
    ConflictSeverity
)

from models.results import (
    ClassifiedArticleResult,
    ResolvedArticleResult
)

from models.search import SearchResult

# Import custom exceptions
from core.exceptions import (
    AgentConfigurationError,
    AgentExecutionError,
    DSPyExecutionError,
    SearchError,
    APIError,
    ErrorSeverity
)

class HybridConflictResolutionAgent():
    """
    Enhanced agent responsible for resolving conflicts between LLM classifications.
    Now leverages full content extraction from BraveSearchService for better resolution.
    """
    def __init__(self, 
                 config: ConflictResolutionConfig, 
                 logger: logging.Logger, 
                 llm_service: LLMService,
                 brave_search_service: Optional[BraveSearchService] = None):
        """
        Initializes the HybridConflictResolutionAgent with enhanced search capabilities.
        """
        #super().__init__(config, logger, llm_service)
        self.config = config
        self.logger = logger
        self.llm_service = llm_service
        self.agent_name = self.__class__.__name__
        
        
        if not isinstance(config, ConflictResolutionConfig):
            raise AgentConfigurationError(
                f"Config must be ConflictResolutionConfig instance, got {type(config)}",
                config_key="conflict_resolution_config_type",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL
            )
        
        self.cr_config = config
        
        # Initialize DSPy resolver with enhanced signature for external context
        self.dspy_resolver = None
        if self.cr_config.enable_reasoning:
            try:
                self.dspy_resolver = DSPyConflictResolver()
                self.logger.debug("DSPy resolver initialized successfully")
            except Exception as e:
                error_msg = f"Failed to initialize DSPy resolver: {e}"
                self.logger.error(error_msg)
                raise AgentConfigurationError(
                    error_msg,
                    config_key="dspy_resolver",
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.HIGH
                ) from e

        # Enhanced BraveSearchService initialization
        self.brave_search_service = None
        if self.cr_config.enable_external_search:
            if brave_search_service:
                self.brave_search_service = brave_search_service
                self.logger.debug("BraveSearchService injected successfully with content extraction")
            else:
                try:
                    # Initialize with content extraction enabled
                    self.brave_search_service = BraveSearchService(
                        logger=self.logger, 
                        extract_content=True
                    )
                    self.logger.debug("BraveSearchService initialized with content extraction")
                except Exception as e:
                    error_msg = f"Failed to initialize BraveSearchService: {e}"
                    self.logger.warning(error_msg)
                    self.cr_config.enable_external_search = False
                    self.brave_search_service = None

        self.logger.info(f"Enhanced HybridConflictResolutionAgent initialized with reasoning={self.cr_config.enable_reasoning}, enhanced_search={self.cr_config.enable_external_search}")

    def process_article_for_conflict(self, article: ClassifiedArticleResult) -> ResolvedArticleResult:
        """
        Enhanced conflict resolution with better search integration.
        """
        # Validate input data
        is_valid, error_msg = self._validate_article_data(article)
        if not is_valid:
            self.logger.error(error_msg)
            return self._create_resolved_article(
                classified_article=article,
                resolution_method=ResolutionMethod.HUMAN_REVIEW_REQUIRED,
                resolution_details={
                    "conflict_severity": ConflictSeverity.HIGH.value,
                    "reason": f"Data validation failed: {error_msg}"
                }
            )

        llm_1 = article.llm_1_classification
        llm_2 = article.llm_2_classification if article.has_conflict else None

        # Layer 0: Check for no conflict or low confidence agreement
        if not article.has_conflict:
            if (llm_1.involvement_confidence < self.cr_config.minimal_absolute_confidence_for_rule or
                llm_1.category_confidence < self.cr_config.minimal_absolute_confidence_for_rule):
                
                # Use search to validate low confidence agreement
                if self.cr_config.enable_external_search and self.brave_search_service:
                    search_validation = self._validate_with_external_search(article, llm_1)
                    if search_validation["validation_successful"]:
                        return self._create_resolved_article(
                            classified_article=article,
                            resolution_method=ResolutionMethod.EXTERNAL_VALIDATION,
                            resolution_details=search_validation
                        )
                
                self.logger.debug(f"Low confidence agreement for '{article.article_title}'. Flagging for human review.")
                return self._create_resolved_article(
                    classified_article=article,
                    resolution_method=ResolutionMethod.HUMAN_REVIEW_REQUIRED,
                    resolution_details={
                        "conflict_severity": ConflictSeverity.LOW.value,
                        "reason": "LLMs agreed but confidence below threshold and external validation failed."
                    }
                )
            else:
                return self._create_resolved_article(classified_article=article, resolution_method=ResolutionMethod.NO_CONFLICT)

        # From here, we know there's a conflict
        if llm_2 is None:
            return self._create_resolved_article(
                classified_article=article,
                resolution_method=ResolutionMethod.HUMAN_REVIEW_REQUIRED,
                resolution_details={
                    "conflict_severity": ConflictSeverity.HIGH.value,
                    "reason": "Internal error: Conflict detected but LLM 2 classification is missing."
                }
            )

        confidence_gaps = self._calculate_confidence_gap(llm_1, llm_2)
        involvement_diff = confidence_gaps["involvement_gap"]
        category_diff = confidence_gaps["category_gap"]
        conflict_severity = self._assess_conflict_severity(llm_1, llm_2)

        # Layer 1: Critical conflict escalation
        if conflict_severity == ConflictSeverity.CRITICAL and self.cr_config.escalate_critical:
            self.logger.debug(f"Critical conflict (Perpetrator vs Victim) for '{article.article_title}'. Flagging for human review.")
            return self._create_resolved_article(
                classified_article=article,
                resolution_method=ResolutionMethod.HUMAN_REVIEW_REQUIRED,
                resolution_details={
                    "conflict_severity": conflict_severity.value,
                    "reason": "Perpetrator vs Victim disagreement requires human review"
                }
            )

        # Layer 2: Rule-based resolution
        resolved_by_rules_data = self._resolve_by_rules(llm_1, llm_2, involvement_diff, category_diff)
        if resolved_by_rules_data["resolved_successfully"]:
            self.logger.debug(f"Conflict for '{article.article_title}' resolved by rule-based approach.")
            return self._create_resolved_article(
                classified_article=article,
                resolution_method=ResolutionMethod.CONFIDENCE_BASED,
                resolution_details=resolved_by_rules_data
            )

        # Layer 3: Enhanced external search analysis BEFORE LLM arbitration
        external_context = None
        if self.cr_config.enable_external_search and self.brave_search_service:
            try:
                external_context = self._get_external_context(article)
                if external_context and external_context["has_relevant_content"]:
                    # Try to resolve using external context analysis
                    resolution_result = self._resolve_with_external_context(article, llm_1, llm_2, external_context)
                    if resolution_result["resolved_successfully"]:
                        self.logger.debug(f"External context resolved conflict for '{article.article_title}'.")
                        return self._create_resolved_article(
                            classified_article=article,
                            resolution_method=ResolutionMethod.EXTERNAL_CONTEXT_ANALYSIS,
                            resolution_details=resolution_result
                        )
            except Exception as e:
                self.logger.warning(f"External context analysis failed for '{article.article_title}': {e}")

        # Layer 4: LLM Reasoning with external context
        if self.cr_config.enable_reasoning and self.dspy_resolver:
            try:
                with self.llm_service.use_model('arbitration'):
                    # Enhanced DSPy call with external context
                    final_involvement_enum, final_category_enum, resolution_explanation, resolution_confidence = \
                        self.dspy_resolver.forward_with_context(
                            article_text=article.raw_article_text,
                            entity_name=article.entity_name,
                            llm_1_classification=llm_1,
                            llm_2_classification=llm_2,
                            external_context=external_context
                        )

                if resolution_confidence >= self.cr_config.llm_arbitration_min_confidence:
                    self.logger.debug(f"Enhanced LLM arbitration resolved conflict for '{article.article_title}' with confidence {resolution_confidence}.")
                    return self._create_resolved_article(
                        classified_article=article,
                        resolution_method=ResolutionMethod.REASONING_ARBITRATION,
                        resolution_details={
                            "final_involvement": final_involvement_enum.value,
                            "final_category": final_category_enum.value,
                            "explanation": resolution_explanation,
                            "confidence": resolution_confidence,
                            "resolved_by": "Enhanced_LLM_Arbitration_With_Context",
                            "external_context_used": bool(external_context)
                        }
                    )

            except Exception as e:
                self.logger.error(f"Enhanced DSPy arbitration error for '{article.article_title}': {e}")

        # Layer 5: Final fallback with rich context
        self.logger.debug(f"All automated resolution failed for '{article.article_title}'. Providing rich context for human review.")
        return self._create_resolved_article(
            classified_article=article,
            resolution_method=ResolutionMethod.HUMAN_REVIEW_REQUIRED,
            resolution_details={
                "conflict_severity": conflict_severity.value,
                "reason": "No automated resolution succeeded after all enhanced layers.",
                "external_context": external_context,
                "confidence_analysis": confidence_gaps,
                "rule_analysis": resolved_by_rules_data
            }
        )

    def _get_external_context(self, article: ClassifiedArticleResult) -> Optional[Dict]:
        """
        Enhanced method to get external context with content analysis.
        """
        try:
            search_query = self._construct_search_query(article)
            search_results = self.brave_search_service.search(search_query, count=3)

            if not search_results:
                return None

            # Analyze the extracted content
            relevant_content = []
            total_content_length = 0
            
            for result in search_results:
                if result.content and len(result.content.strip()) > 100:
                    # Basic relevance check
                    entity_mentions = article.entity_name.lower() in result.content.lower()
                    title_overlap = any(word.lower() in result.content.lower() 
                                      for word in article.article_title.split() 
                                      if len(word) > 3)
                    
                    if entity_mentions or title_overlap:
                        relevant_content.append({
                            "source": result.source_domain,
                            "title": result.title,
                            "content": result.content[:2000],  # Limit content length
                            "url": result.url,
                            "relevance_indicators": {
                                "entity_mentioned": entity_mentions,
                                "title_overlap": title_overlap
                            }
                        })
                        total_content_length += len(result.content)

            return {
                "has_relevant_content": len(relevant_content) > 0,
                "search_query": search_query,
                "total_sources": len(search_results),
                "relevant_sources": len(relevant_content),
                "total_content_length": total_content_length,
                "relevant_content": relevant_content,
                "search_timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Error getting external context for '{article.article_title}': {e}")
            return None

    def _resolve_with_external_context(self, article: ClassifiedArticleResult, 
                                     llm_1: LLMClassificationDetails, 
                                     llm_2: LLMClassificationDetails,
                                     external_context: Dict) -> Dict:
        """
        Attempt to resolve conflict using external context analysis.
        This could use simple heuristics or a separate LLM call.
        """
        if not external_context["has_relevant_content"]:
            return {"resolved_successfully": False, "reason": "No relevant external content"}

        try:
            # Simple heuristic-based resolution using external context
            relevant_content = external_context["relevant_content"]
            
            # Combine all relevant content for analysis
            combined_context = " ".join([content["content"] for content in relevant_content])
            
            # Basic sentiment and involvement analysis
            entity_name = article.entity_name.lower()
            
            # Look for involvement indicators in external content
            perpetrator_indicators = ["charged", "accused", "convicted", "sentenced", "arrested", "indicted"]
            victim_indicators = ["victim", "targeted", "suffered", "injured", "harmed", "died"]
            neutral_indicators = ["witness", "observer", "unrelated", "mentioned"]
            
            perpetrator_score = sum(1 for indicator in perpetrator_indicators 
                                  if indicator in combined_context.lower())
            victim_score = sum(1 for indicator in victim_indicators 
                             if indicator in combined_context.lower())
            neutral_score = sum(1 for indicator in neutral_indicators 
                              if indicator in combined_context.lower())
            
            # Simple resolution logic
            if perpetrator_score > victim_score and perpetrator_score > neutral_score:
                suggested_involvement = EntityInvolvementType.PERPETRATOR
                confidence = min(0.8, 0.5 + (perpetrator_score * 0.1))
            elif victim_score > perpetrator_score and victim_score > neutral_score:
                suggested_involvement = EntityInvolvementType.VICTIM  
                confidence = min(0.8, 0.5 + (victim_score * 0.1))
            else:
                suggested_involvement = EntityInvolvementType.NEUTRAL
                confidence = 0.6

            # Only resolve if confidence is reasonable and aligns with one of the LLM predictions
            if confidence >= 0.7 and (suggested_involvement == llm_1.entity_involvement or 
                                    suggested_involvement == llm_2.entity_involvement):
                
                # Choose the category from the LLM that matches the involvement
                if suggested_involvement == llm_1.entity_involvement:
                    final_category = llm_1.adverse_media_category
                else:
                    final_category = llm_2.adverse_media_category

                return {
                    "resolved_successfully": True,
                    "final_involvement": suggested_involvement.value,
                    "final_category": final_category.value,
                    "confidence": confidence,
                    "explanation": f"External context analysis suggests {suggested_involvement.value} based on {perpetrator_score + victim_score + neutral_score} relevant indicators",
                    "external_indicators": {
                        "perpetrator_score": perpetrator_score,
                        "victim_score": victim_score,
                        "neutral_score": neutral_score
                    },
                    "sources_used": len(relevant_content)
                }

            return {
                "resolved_successfully": False,
                "reason": f"External context confidence too low ({confidence:.2f}) or no LLM alignment",
                "analysis": {
                    "suggested_involvement": suggested_involvement.value,
                    "confidence": confidence,
                    "perpetrator_score": perpetrator_score,
                    "victim_score": victim_score,
                    "neutral_score": neutral_score
                }
            }

        except Exception as e:
            return {
                "resolved_successfully": False,
                "reason": f"External context analysis failed: {e}"
            }

    def _validate_with_external_search(self, article: ClassifiedArticleResult, 
                                     llm_classification: LLMClassificationDetails) -> Dict:
        """
        Validate low-confidence agreements using external search.
        """
        try:
            external_context = self._get_external_context(article)
            if not external_context or not external_context["has_relevant_content"]:
                return {"validation_successful": False, "reason": "No external context available"}

            # Simple validation: if external context supports the LLM decision, increase confidence
            resolution_result = self._resolve_with_external_context(
                article, llm_classification, llm_classification, external_context
            )

            if (resolution_result.get("resolved_successfully") and 
                resolution_result.get("final_involvement") == llm_classification.entity_involvement.value):
                
                return {
                    "validation_successful": True,
                    "final_involvement": llm_classification.entity_involvement.value,
                    "final_category": llm_classification.adverse_media_category.value,
                    "confidence": min(0.85, llm_classification.involvement_confidence + 0.2),
                    "explanation": "External search validated low-confidence LLM agreement",
                    "external_validation": resolution_result
                }

            return {"validation_successful": False, "reason": "External context did not validate LLM decision"}

        except Exception as e:
            return {"validation_successful": False, "reason": f"Validation failed: {e}"}

    # Keep existing helper methods unchanged...
    def _validate_input_state(self, state: AdverseMediaState):
        """Extended validation specific to conflict resolution requirements."""
        super()._validate_input_state(state)
        
        if hasattr(state, 'classified_articles') and state.classified_articles is not None:
            if not isinstance(state.classified_articles, list):
                raise AgentExecutionError(
                    f"classified_articles must be a list, got {type(state.classified_articles)}",
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.HIGH,
                    context={"expected_type": "list", "received_type": type(state.classified_articles).__name__}
                )

    def _validate_article_data(self, article: ClassifiedArticleResult) -> tuple[bool, str]:
        """Validate that article has required fields and proper structure."""
        if not article.llm_1_classification:
            return False, "Article missing llm_1_classification"
        
        llm_1 = article.llm_1_classification
        if not all([llm_1.entity_involvement, llm_1.adverse_media_category, llm_1.involvement_confidence is not None]):
            return False, "LLM 1 classification missing required attributes"
        
        if article.has_conflict:
            if not article.llm_2_classification:
                return False, "Article marked as having conflict but missing llm_2_classification"
            
            llm_2 = article.llm_2_classification
            if not all([llm_2.entity_involvement, llm_2.adverse_media_category, llm_2.involvement_confidence is not None]):
                return False, "LLM 2 classification missing required attributes"
                
        return True, ""

    def _construct_search_query(self, article: ClassifiedArticleResult) -> str:
        """Construct a well-formatted search query."""
        components = []

        entity_name = article.metadata.get('entity_name', '').strip() or article.entity_name.strip()
        if entity_name:
            components.append(f'"{entity_name}"')

        article_title = article.article_title.strip()
        if article_title:
            title_words = article_title.split()
            meaningful_words = [word for word in title_words if len(word) > 3 and
                                word.lower() not in ['the', 'and', 'for', 'are', 'but', 'not', 'you', 'all']]
            if meaningful_words:
                components.extend(meaningful_words[:3])

        components.extend(["adverse media", "involvement"])
        return " ".join(components)

    def _assess_conflict_severity(self, llm_1: LLMClassificationDetails, llm_2: LLMClassificationDetails) -> ConflictSeverity:
        """Assess conflict severity using Enum."""
        involvement_1 = llm_1.entity_involvement
        involvement_2 = llm_2.entity_involvement

        if {involvement_1, involvement_2} == {EntityInvolvementType.PERPETRATOR, EntityInvolvementType.VICTIM}:
            return ConflictSeverity.CRITICAL
        
        if (EntityInvolvementType.NEUTRAL in {involvement_1, involvement_2}) and \
           (EntityInvolvementType.PERPETRATOR in {involvement_1, involvement_2} or EntityInvolvementType.VICTIM in {involvement_1, involvement_2}):
            return ConflictSeverity.HIGH
        
        if llm_1.adverse_media_category != llm_2.adverse_media_category:
            return ConflictSeverity.MEDIUM
        
        return ConflictSeverity.LOW

    def _calculate_confidence_gap(self, llm_1: LLMClassificationDetails, llm_2: LLMClassificationDetails) -> Dict:
        """Calculate confidence gaps for both involvement and category."""
        involvement_gap = abs(llm_1.involvement_confidence - llm_2.involvement_confidence)
        category_gap = abs(llm_1.category_confidence - llm_2.category_confidence)

        return {
            "involvement_gap": involvement_gap,
            "category_gap": category_gap,
            "max_gap": max(involvement_gap, category_gap),
            "avg_gap": (involvement_gap + category_gap) / 2
        }

    def _resolve_by_rules(self, llm_1: LLMClassificationDetails, llm_2: LLMClassificationDetails, 
                          involvement_diff: float, category_diff: float) -> Dict:
        """Attempt to resolve conflict using rule-based confidence and conservative approach."""
        final_involvement: Optional[EntityInvolvementType] = None
        final_category: Optional[AdverseMediaCategory] = None
        involvement_method = "unresolved"
        category_method = "unresolved"

        # Resolve Involvement
        involvement_winner_llm_obj = llm_1 if llm_1.involvement_confidence > llm_2.involvement_confidence else llm_2

        if involvement_diff >= self.cr_config.confidence_gap_for_conservative:
            if involvement_winner_llm_obj.involvement_confidence >= self.cr_config.minimal_absolute_confidence_for_rule:
                final_involvement = involvement_winner_llm_obj.entity_involvement
                involvement_method = "confidence_based_winner_high_abs"
            else:
                involvement_method = "confidence_based_winner_absolute_confidence_too_low"
        else:
            conservative_involvement = self._apply_conservative_approach_involvement(
                llm_1.entity_involvement, llm_2.entity_involvement
            )

            if conservative_involvement == EntityInvolvementType.NEUTRAL and \
               max(llm_1.involvement_confidence, llm_2.involvement_confidence) < self.cr_config.minimal_absolute_confidence_for_rule:
                involvement_method = "conservative_low_confidence_neutral_unresolved"
            else:
                final_involvement = conservative_involvement
                involvement_method = "conservative_approach"

        # Resolve Category
        category_winner_llm_obj = llm_1 if llm_1.category_confidence > llm_2.category_confidence else llm_2

        if category_diff >= self.cr_config.confidence_gap_for_conservative:
            if category_winner_llm_obj.category_confidence >= self.cr_config.minimal_absolute_confidence_for_rule:
                final_category = category_winner_llm_obj.adverse_media_category
                category_method = "confidence_based_winner_high_abs"
            else:
                category_method = "confidence_based_winner_absolute_confidence_too_low"
        else:
            if max(llm_1.category_confidence, llm_2.category_confidence) >= self.cr_config.minimal_absolute_confidence_for_rule:
                final_category = category_winner_llm_obj.adverse_media_category
                category_method = "higher_confidence_low_gap"
            else:
                category_method = "low_confidence_and_low_gap_unresolved"

        resolved_successfully = (final_involvement is not None) and (final_category is not None)
        resolution_confidence = 0.0
        if resolved_successfully:
            resolution_confidence = max(llm_1.involvement_confidence, llm_2.involvement_confidence,
                                       llm_1.category_confidence, llm_2.category_confidence)

        return {
            "resolved_successfully": resolved_successfully,
            "final_involvement": final_involvement.value if final_involvement else None,
            "final_category": final_category.value if final_category else None,
            "involvement_method": involvement_method,
            "category_method": category_method,
            "involvement_confidence_gap": involvement_diff,
            "category_confidence_gap": category_diff,
            "explanation": f"Involvement: {involvement_method}, Category: {category_method}",
            "confidence": resolution_confidence
        }

    def _apply_conservative_approach_involvement(self, involvement_1: EntityInvolvementType, 
                                                involvement_2: EntityInvolvementType) -> EntityInvolvementType:
        """Apply conservative approach for entity involvement."""
        if EntityInvolvementType.PERPETRATOR in {involvement_1, involvement_2}:
            return EntityInvolvementType.PERPETRATOR
        elif EntityInvolvementType.VICTIM in {involvement_1, involvement_2}:
            return EntityInvolvementType.VICTIM
        else:
            return EntityInvolvementType.NEUTRAL

        # THIS IS THE "NEW WAY" 
    def _create_resolved_article(self, classified_article: ClassifiedArticleResult, 
                                resolution_method: ResolutionMethod,
                                resolution_details: Optional[Dict] = None) -> ResolvedArticleResult:
        """
        Create a ResolvedArticleResult using the dataclass factory method.
        
        This method now correctly combines the resolution details and method
        into a single dictionary to match the new from_resolution signature.
        """
        # Create the resolution_data dictionary from the method's parameters.
        resolution_data = {}
        if resolution_details:
            resolution_data.update(resolution_details)
        
        # Add the resolution method to the dictionary.
        resolution_data['resolution_method'] = resolution_method.value
        
        # We must also ensure the final classifications are in the dictionary
        # if they are not already. This is a best practice.
        if 'final_entity_involvement' not in resolution_data:
            # Example: if no conflict, use the primary classification
            resolution_data['final_entity_involvement'] = classified_article.primary_involvement.value
            resolution_data['final_adverse_category'] = classified_article.primary_category.value
            resolution_data['final_overall_confidence'] = classified_article.average_involvement_confidence
        
        # 🔥 FIX: Add the missing is_deemed_adverse logic 🔥
        if 'is_deemed_adverse' not in resolution_data:
            final_category = AdverseMediaCategory(resolution_data['final_adverse_category'])
            final_involvement = EntityInvolvementType(resolution_data['final_entity_involvement'])
            
            # Define what makes something adverse based on your actual enum values
            adverse_categories = {
                AdverseMediaCategory.FRAUD_FINANCIAL_CRIME,
                AdverseMediaCategory.CORRUPTION_BRIBERY,
                AdverseMediaCategory.ORGANIZED_CRIME,
                AdverseMediaCategory.TERRORISM_EXTREMISM,
                AdverseMediaCategory.SANCTIONS_EVASION,
                AdverseMediaCategory.OTHER_SERIOUS_CRIMES
            }
            
            # Something is adverse if:
            # 1. Entity is a PERPETRATOR in any adverse category, OR
            # 2. Entity is involved in any adverse category (even as victim)
            resolution_data['is_deemed_adverse'] = (
                final_involvement == EntityInvolvementType.PERPETRATOR or
                (final_category in adverse_categories and final_involvement != EntityInvolvementType.NEUTRAL)
            )
        
        # Now, call the factory method with the new, correct signature.
        return ResolvedArticleResult.from_resolution(
            classified_article=classified_article,
            resolution_data=resolution_data
        )