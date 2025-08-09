import dspy
import logging
from typing import Literal, List, Dict, Optional, Callable
from dspy import Signature, ChainOfThought, Predict
from dspy.signatures import InputField, OutputField
from dataclasses import dataclass


# Import from your base classes
from core.base_agent import BaseDSPyAgent, AgentMetrics
from core.state import AdverseMediaState, AgentStatus
from core.exceptions import (
    AgentExecutionError, 
    AgentConfigurationError, 
    DSPyExecutionError,
    ErrorSeverity
)
from config.settings import BaseAgentConfig
from services.llm_service import LLMService
from config.settings import ClassificationAgentConfig
from agents.classification.modules import AdverseMediaArticleClassifier 
from models.results import dict_to_classified_article_result


class ClassificationAgent(BaseDSPyAgent):
    """
    Agent responsible for classifying articles using one or two LLMs.
    Focuses solely on classification - conflict resolution is handled by the next agent.
    Now inherits from BaseDSPyAgent for better error handling, metrics, and retry logic.
    """
    
    def __init__(self, 
                 config: ClassificationAgentConfig, 
                 logger: logging.Logger,
                 llm_service: LLMService):
        """
        Initialize the ClassificationAgent with dependency injection.
        
        Args:
            config: ClassificationAgentConfig instance
            logger: Logger instance
            llm_service: LLM service providing access to configured models
        """
        # Initialize BaseDSPyAgent with primary LLM from service
        super().__init__(config, logger, llm_service=llm_service)

        self._validate_configuration()
        
        # Store service for dependency injection
        self.llm_service = llm_service
        
        # Cast config to our specific type
        self.config: ClassificationAgentConfig = config
        
        # Create classifiers using the service
        self.primary_classifier = AdverseMediaArticleClassifier(
            llm_service=self.llm_service, 
            llm_role=self.config.primary_llm_role
        )
        
        # Create secondary classifier if configured and available
        self.secondary_classifier = None
        if self.config.require_secondary_llm or self._has_secondary_model():
            try:
                self.secondary_classifier = AdverseMediaArticleClassifier(
                    llm_service=self.llm_service, 
                    llm_role=self.config.secondary_llm_role
                )
            except ValueError as e:
                # Secondary model not available, log warning and continue
                self.logger.warning(f"Secondary LLM role '{self.config.secondary_llm_role}' not available: {e}")
                if self.config.require_secondary_llm:
                    raise AgentConfigurationError(
                        f"Secondary LLM is required but not available: {e}",
                        config_key="require_secondary_llm",
                        agent_name=self.agent_name,
                        severity=ErrorSeverity.CRITICAL
                    )
        
        self.logger.info(f"🔍 SECONDARY LLM DEBUG:")
        self.logger.info(f"  require_secondary_llm = {self.config.require_secondary_llm}")
        self.logger.info(f"  secondary_llm_role = '{self.config.secondary_llm_role}'")
        self.logger.info(f"  _has_secondary_model() = {self._has_secondary_model()}")
        self.logger.info(f"  Available models in service: {list(self.llm_service.models.keys())}")
        self.logger.info(f"  secondary_classifier created = {self.secondary_classifier is not None}")
        
        if self.secondary_classifier is None:
            self.logger.error("❌ SECONDARY CLASSIFIER IS NULL!")
        else:
            self.logger.info("✅ SECONDARY CLASSIFIER CREATED SUCCESSFULLY")

    def _validate_configuration(self):
        """Validate configuration parameters"""
        if not (0 <= self.config.confidence_threshold <= 1):
            raise AgentConfigurationError(
                f"Confidence threshold must be between 0 and 1, got {self.config.confidence_threshold}",
                config_key="confidence_threshold",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL
            )
    
        if self.config.max_classification_errors < 1:
            raise AgentConfigurationError(
                f"Max classification errors must be at least 1, got {self.config.max_classification_errors}",
                config_key="max_classification_errors",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL
            )
    
    def _has_secondary_model(self) -> bool:
        """Check if secondary model is available in the service"""
        try:
            self.llm_service.get_model(self.config.secondary_llm_role)
            return True
        except ValueError:
            return False

    def _run_implementation(self, state: AdverseMediaState) -> AdverseMediaState:
        """
        Main implementation of the classification logic with unified state management.
        """
        agent_name = "classification_agent"
        state.start_agent(agent_name)
        self._update_agent_status(state, f"Starting classification for entity: {getattr(state, 'entity_name', 'Unknown')}")

        try:
            # Use BaseAgent's validation
            self._validate_input_state(state)
            
            # Validate specific requirements
            if not state.entity_name:
                error = AgentConfigurationError(
                    "No entity name provided for classification.",
                    config_key="entity_name",
                    agent_name=agent_name,
                    severity=ErrorSeverity.CRITICAL
                )
                self._handle_error(state, error)
                state.fail_agent(agent_name, error)
                return state
                
            if not state.filtered_search_results:
                self._update_agent_status(state, "No articles received for classification.", "WARNING")
                state.complete_agent(
                    agent_name,
                    AgentStatus.COMPLETED
                )
                return state

            # Initialize classification results
            state.classified_articles = []
            state.classification_errors = 0
            state.classification_metrics = {}

            
            # Use BaseAgent's batch processing for better error handling
            articles_to_process = [
                (i, article_data) for i, article_data in enumerate(state.filtered_search_results)
                if self._should_process_article(article_data, state)
            ]
            
            if not articles_to_process:
                self._update_agent_status(state, "No valid articles found for classification.", "WARNING")
                state.complete_agent(
                    agent_name,
                    AgentStatus.COMPLETED
                )
                return state
            
            # Process articles with enhanced error handling
            self._process_articles_batch(state, articles_to_process)
            
            # Set completion status and finalize
            self._finalize_classification_status(state)
            
            # Complete agent with output data
            state.complete_agent(
                agent_name,
                AgentStatus.COMPLETED,
                output_data={
                    "classified_articles": state.classified_articles,
                    "classification_metrics": state.classification_metrics
                }
            )
            return state
            
        except Exception as e:
            error = AgentExecutionError(
                f"Classification failed: {e}",
                agent_name=agent_name,
                severity=ErrorSeverity.CRITICAL,
                context={"entity_name": getattr(state, 'entity_name', 'unknown')}
            )
            self._handle_error(state, error)
            state.fail_agent(agent_name, error)
            return state

    def _should_process_article(self, article_data, state: AdverseMediaState) -> bool:
        article_content = getattr(article_data, 'content', '').strip()
        article_title = getattr(article_data, 'title', 'Unknown')

        if not article_content:
            if self.config.skip_articles_without_content:
                self._update_agent_status(
                    state,
                    f"Skipping article '{article_title}' due to missing content",
                    "WARNING"
            )
                return False
        return True

    def _process_articles_batch(self, state: AdverseMediaState, articles_to_process: List[tuple]):
        """Process articles using BaseAgent's safe processing methods"""
        classification_errors = 0
        
        for i, article_data in articles_to_process:
            article_title = getattr(article_data, 'title', f'Article {i+1}')
            
            # Use BaseAgent's safe processing
            result = self._safe_process_item(
                func=lambda article: self._classify_single_article(state.entity_name, article),
                state=state,
                item=article_data,
                item_id=article_title
            )
            
            if result is not None:
                classified_article = dict_to_classified_article_result(result, state.entity_name)
                state.classified_articles.append(classified_article)
                self._update_agent_status(state, f"Successfully classified: {article_title}")
            else:
                classification_errors += 1
                state.classification_errors += 1  # Track on state
                
                # Check if we should stop processing
                if classification_errors >= self.config.max_classification_errors:
                    error = AgentExecutionError(
                        f"Too many classification errors ({classification_errors}) - stopping processing.",
                        agent_name=self.agent_name,
                        severity=ErrorSeverity.HIGH,
                        context={"max_errors": self.config.max_classification_errors}
                    )
                    self._handle_error(state, error)
                    state.set_review_required(
                    "Multiple classification failures", 
                    self.agent_name
                    )
                    return
                

    def _classify_single_article(self, entity_name: str, article_data) -> Dict:
        """
        Classify a single article using primary (and optionally secondary) LLM.
        Uses BaseAgent's retry logic for robust API calls.
        """
        # Extract raw content
        article_text = article_data.content
        article_title = getattr(article_data, 'title', 'Unknown')
        article_link = getattr(article_data, 'url', '')
        
        # ===== ADD THIS CONTENT PREPROCESSING =====
        # Truncate article content to stay within token limits
        processed_article_text = self._preprocess_article_content(article_text, entity_name)
        # ==========================================
        
        # Primary LLM classification with retry logic
        primary_analysis = self._execute_with_llm(
            self.primary_classifier,
            entity_name=entity_name,
            article_text=processed_article_text  # Now truncated/preprocessed content
        )

        
        # Build classification result
        classified_info = {
            "article_title": article_title,
            "article_link": article_link,
            "raw_article_text": article_text,
            "processed_article_text": processed_article_text,
            "llm_1_classification": {
                "entity_involvement": primary_analysis.entity_involvement,
                "adverse_media_category": primary_analysis.adverse_media_category,
                "involvement_reasoning": primary_analysis.involvement_reasoning,
                "category_reasoning": primary_analysis.category_reasoning,
                "involvement_confidence": primary_analysis.involvement_confidence,
                "category_confidence": primary_analysis.category_confidence,
            },
            "has_conflict": False  # Default - no conflict with single LLM
        }

        # Secondary LLM classification if available
        if self.secondary_classifier:
            try:
                secondary_analysis = self._execute_with_llm(
                    self.secondary_classifier,
                    entity_name=entity_name,
                    article_text=processed_article_text
                )
                self.logger.info(f"✅ SECONDARY CLASSIFICATION SUCCESSFUL for: {article_title}")

                
                classified_info["llm_2_classification"] = {
                    "entity_involvement": secondary_analysis.entity_involvement,
                    "adverse_media_category": secondary_analysis.adverse_media_category,
                    "involvement_reasoning": secondary_analysis.involvement_reasoning,
                    "category_reasoning": secondary_analysis.category_reasoning,
                    "involvement_confidence": secondary_analysis.involvement_confidence,
                    "category_confidence": secondary_analysis.category_confidence,
                }
                
                # Enhanced conflict detection with detailed information
                conflict_info = self._detect_conflicts(primary_analysis, secondary_analysis)
                classified_info.update(conflict_info)

            except Exception as e:
                self.logger.error(f"❌ SECONDARY LLM FAILED for article '{article_title}': {str(e)}")
                self.logger.error(f"❌ EXCEPTION TYPE: {type(e)}")
                self.logger.warning(
                    f"Secondary LLM failed for article '{article_title}': {str(e)}. "
                    f"Continuing with primary classification only."
                )
                # Add failure info to result but don't raise
                classified_info["llm_2_classification"] = {
                    "error": str(e),
                    "status": "failed",
                    "message": f"Secondary LLM processing failed: {str(e)}"
                }

        return classified_info
    
    def _preprocess_article_content(self, article_text: str, entity_name: str) -> str:
        """
        Preprocess article content to stay within token limits while preserving relevance.
        """
        if not article_text or not article_text.strip():
            return article_text
        
        # Rough token estimation (4 chars ≈ 1 token)
        MAX_TOKENS = 4000  # Conservative limit for Groq models
        MAX_CHARS = MAX_TOKENS * 4
        
        # If already short enough, return as-is
        if len(article_text) <= MAX_CHARS:
            return article_text
        
        # Strategy 1: Extract relevant paragraphs mentioning the entity
        entity_paragraphs = []
        other_paragraphs = []
        
        paragraphs = article_text.split('\n\n')
        entity_name_lower = entity_name.lower()
        
        for para in paragraphs:
            if entity_name_lower in para.lower():
                entity_paragraphs.append(para)
            else:
                other_paragraphs.append(para)
        
        # Start with entity-relevant paragraphs
        result_text = '\n\n'.join(entity_paragraphs)
        
        # Add other paragraphs until we hit the limit
        for para in other_paragraphs:
            potential_text = result_text + '\n\n' + para
            if len(potential_text) <= MAX_CHARS:
                result_text = potential_text
            else:
                break
        
        # If still too long, truncate smartly
        if len(result_text) > MAX_CHARS:
            # Take first part + last part with entity mentions prioritized
            first_part = result_text[:MAX_CHARS//2]
            last_part = result_text[-MAX_CHARS//2:]
            result_text = first_part + "\n\n[... content truncated ...]\n\n" + last_part
        
        # Final safety truncation
        if len(result_text) > MAX_CHARS:
            result_text = result_text[:MAX_CHARS-50] + "\n\n[... truncated for length ...]"
        
        self.logger.info(f"Article preprocessed: {len(article_text)} -> {len(result_text)} chars")
        return result_text
    
    def _detect_conflicts(self, primary_analysis, secondary_analysis) -> dict:
        """
        Detect conflicts between primary and secondary LLM analyses.
        Returns detailed conflict information.
        """
        # Check for disagreement
        has_disagreement = (
            primary_analysis.entity_involvement != secondary_analysis.entity_involvement or
            primary_analysis.adverse_media_category != secondary_analysis.adverse_media_category
        )
    
        # Check confidence for both classifications
        min_involvement_confidence = min(
            primary_analysis.involvement_confidence, 
            secondary_analysis.involvement_confidence
        )
        min_category_confidence = min(
            primary_analysis.category_confidence, 
            secondary_analysis.category_confidence
        )
    
        has_low_confidence = (
            min_involvement_confidence < self.config.confidence_threshold or
            min_category_confidence < self.config.confidence_threshold
        )
    
        # Determine conflict type
        if has_disagreement:
            conflict_type = "disagreement"
        elif has_low_confidence:
            conflict_type = "low_confidence"
        else:
            conflict_type = "none"
    
        return {
            "has_conflict": has_disagreement or has_low_confidence,
            "conflict_type": conflict_type,
            "min_involvement_confidence": min_involvement_confidence,
            "min_category_confidence": min_category_confidence,
            "confidence_details": {
                "primary_involvement": primary_analysis.involvement_confidence,
                "secondary_involvement": secondary_analysis.involvement_confidence,
                "primary_category": primary_analysis.category_confidence,
                "secondary_category": secondary_analysis.category_confidence,
            }
        }

    def _finalize_classification_status(self, state: AdverseMediaState):
        """Set final classification status based on results"""
        if state.classified_articles:
            self._update_agent_status(
                state, 
                f"Classification completed successfully. Processed {len(state.classified_articles)} articles."
            )
        else:
            self._update_agent_status(state, "Classification completed - no articles processed.", "WARNING")

        # Calculate and store conflict statistics
        total_articles = len(state.classified_articles)
        
        # FIXED: Use attribute access instead of .get() on ClassifiedArticleResult objects
        conflicted_articles = [a for a in state.classified_articles if a.has_conflict]

        if total_articles > 0:
            conflict_stats = {
                "total_articles": total_articles,
                "conflicted_articles": len(conflicted_articles),
                "conflict_rate": len(conflicted_articles) / total_articles,
                # FIXED: Use attribute access instead of .get()
                "disagreement_conflicts": len([a for a in conflicted_articles if a.conflict_type == "disagreement"]),
                "low_confidence_conflicts": len([a for a in conflicted_articles if a.conflict_type == "low_confidence"])
            }
        
            self.logger.info(f"Classification conflict statistics: {conflict_stats}")
            # Add to state for monitoring
            state.classification_metrics = conflict_stats
        else:
            state.classification_metrics = {"total_articles": 0, "conflicted_articles": 0, "conflict_rate": 0.0}

    def get_classification_metrics(self, state: AdverseMediaState = None) -> Dict:
        base_metrics = self.get_metrics()
    
    # Get model information from service (static)
        available_models = self.llm_service.list_models()
    
    # Static configuration metrics (always available)
        static_metrics = {
        "primary_llm_role": self.config.primary_llm_role,
        "secondary_llm_role": self.config.secondary_llm_role if self.secondary_classifier else None,
        "primary_llm_model": available_models.get(self.config.primary_llm_role, "Unknown"),
        "secondary_llm_model": available_models.get(self.config.secondary_llm_role, "None") if self.secondary_classifier else None,
        "has_secondary_llm": self.secondary_classifier is not None,
        "available_models": list(available_models.keys()),
        "confidence_threshold": self.config.confidence_threshold,
    }
    
    # Dynamic metrics (only if state provided)
        dynamic_metrics = {}
        if state and hasattr(state, 'classification_metrics'):
            dynamic_metrics = {
            "conflict_metrics": state.classification_metrics,
            "total_articles_processed": len(getattr(state, 'classified_articles', [])),
            "classification_errors": getattr(state, 'classification_errors', 0),
        }
    
        return {
        **base_metrics.__dict__,
        **static_metrics,
        **dynamic_metrics
    }