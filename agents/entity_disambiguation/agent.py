from typing import List, Optional, Dict, Literal, Tuple
from dataclasses import dataclass, field
import logging
import json
from urllib.parse import urlparse
import asyncio
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from functools import wraps
import dspy
from config.settings import BaseAgentConfig
from core.exceptions import AgentConfigurationError, AgentInitializationError, AgentExecutionError, DSPyExecutionError, SearchError, ErrorSeverity
from services.llm_service import LLMService # Assuming this exists
from services.tavily_client import TavilyService # Assuming this exists
from services.embedding_service import EmbeddingService # Already refactored
from services.ner_service import NERService, NEREntity # NEW: Import NERService and NEREntity
from core.base_agent import BaseDSPyAgent
# Import entity models from the centralized location
from models.entity import EntityContext, EntityCandidate, DisambiguationResult
from core.state import AdverseMediaState, AgentStatus
from config.entity_constants import ENTITY_CREDIBLE_SOURCES, ENTITY_EXCLUDED_DOMAINS

# Import DSPy signatures from the signatures module
from agents.entity_disambiguation.signatures import SmartContextExtractor, EntityValidator
from config.settings import EntityDisambiguationAgentConfig

#import logging
#logging.basicConfig(level=logging.DEBUG)



# ------------------ CIRCUIT BREAKER PATTERN ------------------

class CircuitBreaker:
    """Simple circuit breaker for external service calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e


# ------------------ ENHANCED ENTITY DISAMBIGUATION AGENT ------------------

class EntityDisambiguationAgent(BaseDSPyAgent):
    """Production-ready entity disambiguation with search integration and conflict resolution"""
    
    def __init__(self,
                 config: EntityDisambiguationAgentConfig, 
                 logger: logging.Logger,
                 llm_service: 'LLMService', # Dependency injected LLMService
                 tavily_service: Optional[TavilyService] = None, # Inject TavilyService
                 embedding_service: Optional[EmbeddingService] = None, # Inject EmbeddingService
                 ner_service: Optional[NERService] = None): # NEW: Inject NERService
        
        # Ensure config is of the correct type
        if not isinstance(config, EntityDisambiguationAgentConfig):
            raise AgentConfigurationError(
                f"Config must be an instance of EntityDisambiguationAgentConfig, got {type(config)}.",
                config_key="entity_disambiguation_config_type",
                severity=ErrorSeverity.CRITICAL
            )
        
        super().__init__(config, logger, llm_service=llm_service) # Pass llm_service to BaseDSPyAgent
        
        # Specific settings for this agent
        self.confidence_threshold = config.confidence_threshold
        self.review_threshold = config.review_threshold
        self.max_candidates = config.max_candidates
        self.max_context_terms = config.max_context_terms
        
        # Initialize Embedding Service (NEW)
        if embedding_service:
            self.embedding_service = embedding_service
        else:
            self.embedding_service = EmbeddingService() # Use default model if not provided

        # Initialize NER Service (NEW)
        if ner_service:
            self.ner_service = ner_service
        else:
            # Use model name from config if provided, otherwise default in NERService
            model_name = config.ner_model if config.ner_model else "dbmdz/bert-large-cased-finetuned-conll03-english"
            self.ner_service = NERService(model_name=model_name)
        
        # DSPy programs - configured using the injected LLM via _execute_with_llm
        self.context_extractor = dspy.ChainOfThought(SmartContextExtractor)
        self.entity_validator = dspy.ChainOfThought(EntityValidator)
        
        # Initialize search client using injected TavilyService or create one
        if tavily_service:
            self.tavily_service = tavily_service
        else:
            if not config.tavily_api_key:
                raise AgentInitializationError(
                    "Tavily API key is required for EntityDisambiguationAgent when TavilyService is not provided.",
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.CRITICAL
                )
            self.tavily_service = TavilyService(api_key=config.tavily_api_key)
        
        # Initialize circuit breakers for external services
        self.search_circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30)
        self.embedding_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        self.ner_circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
        # Thread pool for parallel operations
        self.thread_pool = ThreadPoolExecutor(max_workers=3)
        
        # Simple in-memory cache for search results
        self._search_cache = {}
        self._embedding_cache = {}
        
        self.logger.info(f"EntityDisambiguationAgent '{self.agent_name}' initialized.")

    def _run_implementation(self, state: AdverseMediaState) -> AdverseMediaState:
        print("About to call debug line for _run_implementation!")
        self.logger.debug(f"=== REAL _run_implementation CALLED with entity: {state.entity_name} ===")
        agent_name = "entity_disambiguation_agent"
        state.start_agent(agent_name)  # Mark agent as in-progress
        self._update_agent_status(state, f"Starting disambiguation for: {state.entity_name}")

        try:
            # Step 1: Extract context from user input
            context = self._retry_with_backoff(self._extract_context, state.user_input, state.entity_name)
            self._update_agent_status(state, "Context extracted successfully.")

            self.logger.debug(f"DEBUG: About to search for candidates")

            # Step 2: Search for potential entity candidates
            candidates = self._retry_with_backoff(self._search_for_candidates, state.entity_name, context)
            self._update_agent_status(state, f"Found {len(candidates)} potential candidates.")
            
            self.logger.debug(f"DEBUG: About to score {len(candidates)} candidates")

            if not candidates:
                result = DisambiguationResult(
                    status="no_matches",
                    reasoning="No potential matches found in search results",
                    confidence_score=0.0
                )
            else:
                self.logger.debug(f"DEBUG: Calling _score_candidates now...")
            
                scored_candidates = self._retry_with_backoff(self._score_candidates_simplified, candidates, context, state.entity_name, state.user_input)
                self.logger.debug(f"DEBUG: _score_candidates returned {len(scored_candidates)} candidates")
            
                self._update_agent_status(state, "Candidates scored and validated.")

                scored_candidates = self._apply_context_penalties(scored_candidates, context)
                self._update_agent_status(state, "Context penalties applied.")

            # Step 4: Apply decision logic
                result = self._make_decision(scored_candidates, state.entity_name, context)

            #update state  
            state.resolved_entity = result.resolved_entity
            state.all_candidates = result.all_candidates or []
            state.disambiguation_confidence = result.confidence_score or 0.0
            state.disambiguation_result = result

            confidence = result.confidence_score or 0.0
            
            self._update_agent_status(state, f"Disambiguation result: {result.status} | Confidence: {confidence:.3f}", level="DEBUG") # Or INFO if it's important for general logs

            # Log outcome
            try:
                if result.status == "resolved":
                    entity_name = getattr(state.resolved_entity, 'name', str(state.resolved_entity)) if state.resolved_entity else 'Unknown'
                    self._update_agent_status(state, f"✅ Resolved: {entity_name} ({result.confidence_score:.2f})", level="INFO")
                    state.complete_agent(agent_name, 
                                     AgentStatus.COMPLETED,
                                     output_data={
                        "resolved_entity": state.resolved_entity,
                        "all_candidates": state.all_candidates,
                        "disambiguation_confidence": state.disambiguation_confidence,
                        "disambiguation_result": state.disambiguation_result,
                    })
            
                elif result.status == "needs_review":
                    self._update_agent_status(state, f"⚠️ Ambiguous entity: {result.review_reason}", level="WARNING")
                    state.set_review_required(f"Entity disambiguation ambiguous: {result.review_reason}", "entity_disambiguation_agent")
                    state.complete_agent(agent_name, AgentStatus.NEEDS_REVIEW,
                                     output_data={
                        "resolved_entity": state.resolved_entity,
                        "all_candidates": state.all_candidates,
                        "disambiguation_confidence": state.disambiguation_confidence,
                        "disambiguation_result": state.disambiguation_result,
                    })
            
                elif result.status == "no_matches":
                    self._update_agent_status(state, "❌ No entity matches found", level="WARNING")
                    state.complete_agent(agent_name, AgentStatus.COMPLETED,
                                     output_data={
                        "resolved_entity": state.resolved_entity,
                       "all_candidates": state.all_candidates,
                       "disambiguation_confidence": state.disambiguation_confidence,
                        "disambiguation_result": state.disambiguation_result,

                    })
            
                else:  # result.status == "error"
                    self._update_agent_status(state, f"💥 Error: {result.reasoning}", level="ERROR")
                    error = AgentExecutionError(f"Disambiguation error: {result.reasoning}", 
                               agent_name=self.agent_name, 
                               severity=ErrorSeverity.CRITICAL)
                    state.fail_agent("entity_disambiguation_agent", error)  # This handles error logging
                    return state

            except Exception as completion_error:
                wrapped_error = AgentExecutionError(
                f"error completing disambiguation agent: {str(completion_error)}",
                agent_name=self.agent_name,
                context={"entity_name": state.entity_name, "completion_error": type(completion_error).__name__},
                severity=ErrorSeverity.CRITICAL
            )
                state.fail_agent("entity_disambiguation_agent", wrapped_error)
                return state
            
            self._update_agent_status(state, f"Disambiguation completed with status: {result.status}")
            return state
        
        except Exception as e:
            wrapped_error = AgentExecutionError(
            f"Critical error during disambiguation for '{state.entity_name}': {str(e)}",
            agent_name=self.agent_name,
            context={"entity_name": state.entity_name, "original_error_type": type(e).__name__},
            severity=ErrorSeverity.CRITICAL
        )
            state.fail_agent("entity_disambiguation_agent", wrapped_error)
            return state

    def _extract_context(self, user_input: str, entity_name: str) -> EntityContext:
        """Enhanced context extraction with Nigerian KYC focus and simplified NER fallback"""
        context = EntityContext()
        
        if not user_input.strip():
            return context
            
        try:
            # PRIMARY: DSPy extraction using the enhanced signature
            dspy_result = self._execute_with_llm(
                self.context_extractor, 
                text=user_input, 
                entity_name=entity_name
            )

            # ENHANCED DEBUG BLOCK
            print(f"=== ENHANCED CONTEXT EXTRACTION DEBUG ===")
            print(f"User input: '{user_input}'")
            print(f"Entity name: '{entity_name}'")
            print(f"DSPy extracted roles: {dspy_result.roles}")
            print(f"DSPy extracted locations: {dspy_result.locations}")
            print(f"DSPy extracted organizations: {dspy_result.organizations}")
            print(f"DSPy extracted aliases: {dspy_result.aliases}")
            print(f"DSPy extracted additional_context: {dspy_result.additional_context}")
            print("=============================================")
            
            # Extract fields that match your signature
            context.roles = self._safe_json_parse(dspy_result.roles, [])
            context.locations = self._safe_json_parse(dspy_result.locations, [])
            context.organizations = self._safe_json_parse(dspy_result.organizations, [])
            context.aliases = self._safe_json_parse(dspy_result.aliases, [])
            context.additional_context = self._safe_json_parse(dspy_result.additional_context, {})
            
            dspy_success = True
            
        except DSPyExecutionError as e:
            # DSPy failed, log but continue with NER fallback
            self.logger.warning(f"DSPy context extraction failed: {e.message}. Using minimal NER fallback.")
            dspy_success = False
        except Exception as e:
            # Any other error in DSPy
            self.logger.warning(f"Unexpected error in DSPy context extraction: {str(e)}. Using minimal NER fallback.")
            dspy_success = False
            
        # SIMPLIFIED NER FALLBACK (only if DSPy completely fails)
        if not dspy_success:
            try:
                # Minimal NER extraction for basic context
                ner_results = self.ner_circuit_breaker.call(self.ner_service.extract_entities, user_input)
                for ent in ner_results:
                    entity_text = ent.text
                    entity_type = ent.label 
                    confidence = ent.confidence
                    
                    # Only extract basic entities for fallback
                    if entity_type == 'ORG':
                        context.organizations.append(entity_text)
                    elif entity_type == 'LOC':
                        context.locations.append(entity_text)
                        
                    context.confidence_scores[f"ner_fallback_{entity_type}_{entity_text}"] = confidence
                    
                self.logger.info(f"NER fallback extracted {len(context.organizations)} orgs, {len(context.locations)} locations")
                    
            except Exception as e:
                self.logger.warning(f"NER fallback also failed: {str(e)}. Using minimal context.")
                
        # Enhanced deduplication for all fields
        context.persons = self._deduplicate_preserving_order(context.persons)
        context.organizations = self._deduplicate_preserving_order(context.organizations)
        context.locations = self._deduplicate_preserving_order(context.locations)
        context.roles = self._deduplicate_preserving_order(context.roles)
        context.aliases = self._deduplicate_preserving_order(context.aliases)
        
        # Use helper method for consistent context counting
        total_context_items = self._count_context_richness(context)
        
        if total_context_items == 0:
            raise AgentExecutionError(
                f"No meaningful context extracted for entity '{entity_name}' from input: '{user_input[:100]}'",
                agent_name=self.agent_name,
                context={"user_input": user_input[:100], "dspy_success": dspy_success},
                severity=ErrorSeverity.HIGH
            )
        
        self.logger.info(f"✅ Context extraction complete: {total_context_items} total context items extracted")
        return context

    def _count_context_richness(self, context: EntityContext) -> int:
        """Count all extracted context items consistently - future-proof helper method"""
        # All list-type context fields that actually exist
        list_fields = [
            context.roles, 
            context.organizations, 
            context.locations, 
            context.aliases, 
        ]
        
        # Count items in all list fields
        list_count = sum(len(field) for field in list_fields)   
        return list_count

    def _search_for_candidates(self, entity_name: str, context: EntityContext) -> List[EntityCandidate]:
        """Search for potential entity matches using multiple strategies (parallelized)"""
        
        # ENHANCED DEBUG: Log search initiation
        self.logger.info(f"🔍 TAVILY SEARCH INITIATION")
        self.logger.info(f"Entity name: '{entity_name}'")
        self.logger.info(f"Context available: roles={len(context.roles)}, orgs={len(context.organizations)}, locations={len(context.locations)}, aliases={len(context.aliases)}")
        
        # Prepare search queries
        search_tasks = []
        
        # Strategy 1: Basic entity search
        basic_query = f'"{entity_name}"'
        search_tasks.append(("basic", basic_query))
        
        # Strategy 2: Context-enhanced search
        if context.roles or context.organizations or context.locations:
            context_parts = []
            
            # Add aliases if available
            if context.aliases:
                primary_alias = context.aliases[0]
                context_parts.append(f"also known as {primary_alias}")
            
            if context.roles:
                context_parts.append(f"A {context.roles[0]}")  # "A Musician"               
            if context.organizations:
                context_parts.append(f"Works at {context.organizations[0]}")               
            if context.locations:
                context_parts.append(f"Lives in {context.locations[0]}")               
            
            if context_parts:
                # Remove quotes from entity name in natural language query
                enhanced_query = f"{entity_name}. {'. '.join(context_parts)}"  
            else:
                enhanced_query = f'"{entity_name}" {" ".join(context.roles + context.organizations + context.locations)}'             
            search_tasks.append(("enhanced", enhanced_query))          
        
        # Strategy 3: Biographical search (for people)
        bio_query = f'"{entity_name}" AND (biography OR profile OR about)'
        search_tasks.append(("biographical", bio_query))
        
        # ENHANCED DEBUG: Log all planned searches
        self.logger.info(f"🎯 PLANNED SEARCH STRATEGIES ({len(search_tasks)} total):")
        for i, (search_type, query) in enumerate(search_tasks, 1):
            self.logger.info(f"  {i}. {search_type.upper()}: '{query}'")
        
        # Execute searches in parallel
        all_candidates = []
        futures = []
        
        for search_type, query in search_tasks:
            future = self.thread_pool.submit(self._perform_search, query, search_type, context, entity_name)
            futures.append(future)
        
        # Collect results as they complete
        for future in as_completed(futures):
            try:
                candidates = future.result(timeout=30)  # 30 second timeout per search
                all_candidates.extend(candidates)
            except Exception as e:
                self.logger.warning(f"Parallel search failed: {str(e)}")
                # Continue with other search results
                
        # Deduplicate by URL and name similarity
        #unique_candidates = self._deduplicate_candidates(all_candidates)
        all_candidates.sort(key=lambda x: x.metadata.get("search_score", 0.0), reverse=True)
        
        
        # ENHANCED DEBUG: Final results summary
        #self.logger.info(f"📊 SEARCH RESULTS SUMMARY:")
        #self.logger.info(f"  Total raw results: {len(all_candidates)}")
        #self.logger.info(f"  After deduplication: {len(unique_candidates)}")
        #self.logger.info(f"  Final candidates (top {self.max_candidates}): {len(unique_candidates[:self.max_candidates])}")
        
        # ADD DEBUG HERE:
        #print(f"=== SEARCH CANDIDATES DEBUG ===")
        #print(f"Total candidates found: {len(unique_candidates[:self.max_candidates])}")
        #for i, candidate in enumerate(unique_candidates[:self.max_candidates]):
            #print(f"Candidate {i+1}:")
            #print(f"  Name: {candidate.name}")
            #print(f"  Source: {candidate.source_url}")
            #print(f"  Description: {candidate.description[:200]}...")
            #print(f"  Metadata: {candidate.metadata}")
        #print("==============================")
        self.logger.info(f"Search completed: {len(all_candidates)} candidates found")
        return all_candidates[:self.max_candidates * 2]

    def _perform_search(self, query: str, search_type: str, context: EntityContext, entity_name: str) -> List[EntityCandidate]:
            """Enhanced search with better parameters for biographical content and proper constants usage"""
            
            # Check cache first
            cache_key = f"{search_type}:{query}"
            if cache_key in self._search_cache:
                self.logger.debug(f"✅ CACHE HIT for: {cache_key}")
                return self._search_cache[cache_key]
            
            # ENHANCED DEBUG: Pre-search logging
            self.logger.info(f"🚀 EXECUTING ENHANCED TAVILY SEARCH:")
            self.logger.info(f"  Search Type: {search_type}")
            self.logger.info(f"  Query String: '{query}'")
            
            try:
                # ENHANCED SEARCH PARAMETERS
                search_params = {
                    "query": query,
                    "search_depth": "advanced",        # CRITICAL: Changed from "basic"
                    "max_results": 8,                  # Increased from 5 for better diversity
                    "topic": "general",                # Changed from "news" - more biographical content
                    "include_raw_content": True,       # Get full content for better analysis
                    "include_answer": True,            # Get AI-generated answer
                }
                
                # SMART DOMAIN TARGETING using constants
                
                # Time-based filtering for more recent, comprehensive profiles
                if search_type == "basic":
                    search_params["days"] = 365  # Last year for most current info
                    
                # ENHANCED DEBUG: Log exact parameters being sent
                self.logger.info(f"📋 ENHANCED TAVILY SEARCH PARAMETERS:")
                for key, value in search_params.items():
                    if isinstance(value, list) and len(value) > 3:
                        self.logger.info(f"  {key}: {value[:3]}... (+{len(value)-3} more)")
                    else:
                        self.logger.info(f"  {key}: {value}")
                
                # Record search start time
                search_start_time = time.time()
                
                # Use circuit breaker for search
                response_results = self.search_circuit_breaker.call(
                    self.tavily_service.search,
                    **search_params
                )
                
                # Record search completion time
                search_duration = time.time() - search_start_time
                
                # ADD THE NEW DEBUG LOGGING HERE - RIGHT AFTER THE TAVILY CALL
                self.logger.info(f"🔍 DEBUG TAVILY RESPONSE:")
                self.logger.info(f"Query sent: '{query}'")
                self.logger.info(f"Results count: {len(response_results) if response_results else 0}")
                if response_results:
                    for i, result in enumerate(response_results):
                        score = result.get('score', 0)
                        url = result.get('url', '')
                        title = result.get('title', '')
                        self.logger.info(f"  {i+1}. Score: {score:.3f} | {title} | {url}")
                
                # ENHANCED DEBUG: Log raw Tavily response
                self.logger.info(f"⚡ ENHANCED TAVILY SEARCH COMPLETED:")
                self.logger.info(f"  Duration: {search_duration:.2f}s")
                self.logger.info(f"  Raw results count: {len(response_results) if response_results else 0}")
                
                # ENHANCED DEBUG: Log raw response structure (first result only to avoid spam)
                if response_results:
                    self.logger.debug(f"🔍 SAMPLE RAW TAVILY RESULT (first result):")
                    sample_result = response_results[0]
                    for key, value in sample_result.items():
                        if isinstance(value, str) and len(value) > 200:
                            self.logger.debug(f"  {key}: {value[:200]}... (truncated)")
                        else:
                            self.logger.debug(f"  {key}: {value}")
                            
                    # Log all result URLs for visibility
                    self.logger.info(f"📄 ALL RESULT URLs:")
                    for i, result in enumerate(response_results, 1):
                        url = result.get("url", "NO_URL")
                        title = result.get("title", "NO_TITLE")
                        domain = urlparse(url).netloc if url != "NO_URL" else "NO_DOMAIN"
                        self.logger.info(f"  {i}. {title} -> {domain}")
                else:
                    self.logger.warning(f"⚠️ NO RESULTS returned from Tavily for query: '{query}'")
                
                # QUALITY FILTERING - Enhanced candidate creation
                candidates = []
                for result in response_results:
                    # Quality score based on result characteristics
                    quality_score = self._calculate_result_quality(result, entity_name, context)
                    
                    if quality_score < 0.05:  # Skip low-quality results
                        self.logger.debug(f"❌ Skipping low-quality result: {result.get('title', 'Unknown')} (score: {quality_score:.2f})")
                        continue
                    
                    candidate_name = self._extract_name_from_result(result)
                    
                    candidate = EntityCandidate(
                        name=candidate_name,
                        confidence_score=0.0,  # Will be calculated later
                        context_match={},
                        description=result.get("content", ""),
                        source_url=result.get("url", ""),
                        search_snippet=result.get("content", "")[:500],
                        metadata={
                            "title": result.get("title", ""),
                            "search_type": search_type,
                            "domain": urlparse(result.get("url", "")).netloc,
                            "quality_score": quality_score,
                            "has_raw_content": bool(result.get("raw_content")),
                            "search_score": result.get("score", 0.0),  # Tavily's relevance score
                            "search_query": query,
                            "search_duration": search_duration,
                            "has_ai_answer": bool(result.get("answer")),
                            "content_length": len(result.get("content", "")),
                            "tavily_score_boost": result.get("score", 0.0) > 0.7 
                        }
                    )
                    candidates.append(candidate)
                    
                    # ENHANCED DEBUG: Log candidate creation
                    self.logger.debug(f"✨ HIGH-QUALITY CANDIDATE CREATED:")
                    self.logger.debug(f"  Extracted Name: '{candidate_name}'")
                    self.logger.debug(f"  Quality Score: {quality_score:.2f}")
                    self.logger.debug(f"  Source Domain: {candidate.metadata.get('domain', 'N/A')}")
                    self.logger.debug(f"  Content Length: {candidate.metadata.get('content_length', 0)} chars")
                    self.logger.debug(f"  Tavily Score: {candidate.metadata.get('search_score', 0.0)}")
                
                # Cache the result
                self._search_cache[cache_key] = candidates
                
                # Limit cache size (keep it manageable)
                if len(self._search_cache) > 100:
                    # Remove oldest entries
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]
                    self.logger.debug(f"🗑️ Cache cleanup: removed {oldest_key}")
                
                # ENHANCED DEBUG: Final search summary
                self.logger.info(f"✅ ENHANCED SEARCH COMPLETED for '{search_type}':")
                self.logger.info(f"  Total results received: {len(response_results) if response_results else 0}")
                self.logger.info(f"  High-quality candidates extracted: {len(candidates)}")
                self.logger.info(f"  Search duration: {search_duration:.2f}s")
                
                return candidates
                
            except Exception as e:
                # ENHANCED DEBUG: Log search failure details
                self.logger.error(f"💥 ENHANCED TAVILY SEARCH FAILED:")
                self.logger.error(f"  Search Type: {search_type}")
                self.logger.error(f"  Query: '{query}'")
                self.logger.error(f"  Error Type: {type(e).__name__}")
                self.logger.error(f"  Error Message: {str(e)}")
                
                # Wrap into a custom SearchError
                raise SearchError(
                    f"Enhanced Tavily search failed for query '{query}': {str(e)}",
                    service_name="Tavily",
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.HIGH,
                    context={"query": query, "search_type": search_type, "search_params": search_params}
                ) from e
        
    def _calculate_result_quality(self, result: Dict, entity_name: str, context: EntityContext) -> float:
        """Calculate quality score for search result with DSPy context integration"""
        quality_score = 0.0
        title = result.get("title", "").lower()
        content = result.get("content", "").lower()
        url = result.get("url", "").lower()
        entity_lower = entity_name.lower()

        # FIXED Title relevance (40% weight)
        title_score = 0.0

        # Enhanced perfect matches - handle all common formats
        perfect_match_patterns = [
            title.startswith(entity_lower),
            f"{entity_lower} -" in title,
            f"- {entity_lower}" in title,
            title.startswith(f"{entity_lower} |"),
            title.startswith(f"{entity_lower}:"),
            # NEW: Handle LinkedIn-style formats
            title.endswith(f"- {entity_lower}"),        # "Data Scientist - Deborah Popoola"
            f" {entity_lower} " in title,               # "... Deborah Popoola ..."
            title.startswith(f"{entity_lower} •"),      # LinkedIn bullet format
            f"{entity_lower} - " in title,              # "Deborah Popoola - Data Scientist"
        ]
        
        if any(perfect_match_patterns):
            title_score = 0.4  # Perfect title match
        elif entity_lower in title:
            title_score = 0.2  # Entity mentioned in title
        else:
            # Smart name matching - check for name variations
            name_parts = [part for part in entity_lower.split() if len(part) > 2]
        
            if len(name_parts) >= 2:
                # Check if most name parts appear in title
                parts_in_title = sum(1 for part in name_parts if part in title)
                if parts_in_title >= len(name_parts) * 0.7:  # 70% of name parts present
                    title_score = 0.15  # Partial but strong match
                elif parts_in_title >= 2:  # At least 2 name parts
                    title_score = 0.1   # Weak but potential match

        quality_score += title_score

        # URL relevance (20% weight) - UNCHANGED, this was correct
        domain_score = 0.0
        if any(domain in url for domain in ENTITY_CREDIBLE_SOURCES):
            domain_score = 0.2
        quality_score += domain_score

        # FIXED: DSPy Context scoring (40% weight MAX)
        dspy_raw_score = self._calculate_dspy_context_score(content, context)  # Returns 0.0-0.70
        
        # CRITICAL FIX: Normalize to intended 40% maximum
        context_contribution = min(dspy_raw_score * (0.40 / 0.70), 0.40)
        quality_score += context_contribution

        # Penalty for excluded domains - UNCHANGED
        if any(domain in url for domain in ENTITY_EXCLUDED_DOMAINS):
            quality_score *= 0.5

        # Enhanced logging with fix details
        self.logger.debug(f"📊 FIXED QUALITY SCORE for '{title[:50]}...': {quality_score:.3f}")
        self.logger.debug(f"  Title score: {title_score:.3f} (40% max)")
        self.logger.debug(f"  Domain score: {domain_score:.3f} (20% max)")
        self.logger.debug(f"  DSPy raw score: {dspy_raw_score:.3f} (was returning 0-0.70)")
        self.logger.debug(f"  Context contribution: {context_contribution:.3f} (40% max - NORMALIZED)")
        
        return min(quality_score, 1.0)


    # 4. UNCHANGED _calculate_dspy_context_score method (this is correct as-is)
    def _calculate_dspy_context_score(self, content: str, context: EntityContext) -> float:
        """Calculate quality score based on core DSPy-extracted context matches"""
        dspy_score = 0.0
        content_lower = content.lower()
        
        self.logger.debug(f"    Scoring content with {self._count_context_richness(context)} context items")
        
        # ALIASES - Highest weight (30%) - Critical for public figures
        if context.aliases:
            alias_matches = sum(1 for alias in context.aliases if alias.lower() in content_lower)
            alias_score = min(alias_matches / len(context.aliases), 1.0) * 0.30
            dspy_score += alias_score
            self.logger.debug(f"    Alias matches: {alias_matches}/{len(context.aliases)} = {alias_score:.3f}")
        
        # ROLES - High weight (25%) - Professional identity
        if context.roles:
            role_matches = sum(1 for role in context.roles if role.lower() in content_lower)
            role_score = min(role_matches / len(context.roles), 1.0) * 0.25
            dspy_score += role_score
            self.logger.debug(f"    Role matches: {role_matches}/{len(context.roles)} = {role_score:.3f}")
        
        # ORGANIZATIONS - High weight (20%) - Institutional affiliation
        if context.organizations:
            org_matches = sum(1 for org in context.organizations if org.lower() in content_lower)
            org_score = min(org_matches / len(context.organizations), 1.0) * 0.20
            dspy_score += org_score
            self.logger.debug(f"    Organization matches: {org_matches}/{len(context.organizations)} = {org_score:.3f}")
        
        # LOCATIONS - Medium weight (15%) - Geographic context
        if context.locations:
            location_matches = sum(1 for loc in context.locations if loc.lower() in content_lower)
            location_score = min(location_matches / len(context.locations), 1.0) * 0.15
            dspy_score += location_score
            self.logger.debug(f"    Location matches: {location_matches}/{len(context.locations)} = {location_score:.3f}")
        
        # ADDITIONAL CONTEXT - Lower weight (10%) - Awards, achievements, etc.
        if context.additional_context:
            additional_matches = 0
            total_additional_items = 0
            
            for key, value in context.additional_context.items():
                if isinstance(value, list):
                    for item in value[:3]:  # Limit to top 3 items per key
                        total_additional_items += 1
                        if str(item).lower() in content_lower:
                            additional_matches += 1
                elif isinstance(value, str) and len(value) < 50:  # Avoid long text
                    total_additional_items += 1
                    if value.lower() in content_lower:
                        additional_matches += 1
            
            if total_additional_items > 0:
                additional_score = min(additional_matches / total_additional_items, 1.0) * 0.10
                dspy_score += additional_score
                self.logger.debug(f"    Additional context matches: {additional_matches}/{total_additional_items} = {additional_score:.3f}")
        
        # MULTI-DIMENSIONAL BONUS - Reward articles matching multiple context types
        dimensions_matched = 0
        dimension_checks = [
            (context.aliases, lambda: any(alias.lower() in content_lower for alias in context.aliases)),
            (context.roles, lambda: any(role.lower() in content_lower for role in context.roles)),
            (context.organizations, lambda: any(org.lower() in content_lower for org in context.organizations)),
            (context.locations, lambda: any(loc.lower() in content_lower for loc in context.locations))
        ]
        
        for context_field, check_func in dimension_checks:
            if context_field and check_func():
                dimensions_matched += 1
        
        # Progressive bonus for multiple dimensions (max 10% bonus)
        if dimensions_matched >= 2:
            dimension_bonus = min((dimensions_matched - 1) * 0.03, 0.10)  # 3% per additional dimension
            dspy_score += dimension_bonus
            self.logger.debug(f"    Multi-dimension bonus: {dimensions_matched} dimensions = {dimension_bonus:.3f}")
        
        final_score = min(dspy_score, 0.70)  # Cap at 70% for context scoring
        self.logger.debug(f"    TOTAL DSPy context score: {final_score:.3f} (from {dspy_score:.3f})")
        
        return final_score
    
    def _score_candidates_simplified(self, candidates: List[EntityCandidate], 
                                    context: EntityContext, 
                                    entity_name: str, 
                                    user_context: str) -> List[EntityCandidate]:
        """Fixed scoring: Always considers context + Tavily with smart trust bonuses"""
        
        print(f"=== FIXED SCORING DEBUG ===")
        print(f"Scoring {len(candidates)} candidates")
        print(f"Available context: {self._count_context_richness(context)} total items")

        print(f"=== CANDIDATES TO BE SCORED ===")
        print(f"Total candidates: {len(candidates)}")
        for i, candidate in enumerate(candidates):
            print(f"Candidate {i+1}:")
            print(f"  Name: {candidate.name}")
            print(f"  Source: {candidate.source_url}")
            print(f"  Description: {candidate.description[:200]}...")
            print(f"  Tavily Score: {candidate.metadata.get('search_score', 0.0):.3f}")
            print(f"  Quality Score: {candidate.metadata.get('quality_score', 0.0):.3f}")
            print(f"  Domain: {candidate.metadata.get('domain', 'unknown')}")
            print(f"  Search Type: {candidate.metadata.get('search_type', 'unknown')}")
            print(f"  Title: {candidate.metadata.get('title', 'No title')}")
            print("---")
        print("================================")
        
        for candidate in candidates:
            tavily_score = candidate.metadata.get("search_score", 0.0)
            source_url = candidate.source_url.lower()
            
            # Check if high-trust domain
            is_high_trust = any(domain in source_url for domain in ENTITY_CREDIBLE_SOURCES)
            
            # ALWAYS CALCULATE CONTEXT SCORE (no more bypassing!)
            context_score = 0.0
            description = candidate.description.lower()
            title = candidate.metadata.get("title", "").lower()
            combined_content = f"{title} {description}"
            
            # ALIASES - Highest weight (30% of context score)
            if context.aliases:
                alias_matches = sum(1 for alias in context.aliases if alias.lower() in combined_content)
                alias_score = (alias_matches / len(context.aliases)) * 0.30
                context_score += alias_score
                
            # ROLES - High weight (25% of context score)  
            if context.roles:
                role_matches = sum(1 for role in context.roles if role.lower() in combined_content)
                role_score = (role_matches / len(context.roles)) * 0.25
                context_score += role_score
                
            # ORGANIZATIONS - High weight (20% of context score)
            if context.organizations:
                org_matches = sum(1 for org in context.organizations if org.lower() in combined_content)
                org_score = (org_matches / len(context.organizations)) * 0.20
                context_score += org_score
                
            # LOCATIONS - Medium weight (15% of context score)
            if context.locations:
                loc_matches = sum(1 for loc in context.locations if loc.lower() in combined_content)
                loc_score = (loc_matches / len(context.locations)) * 0.15
                context_score += loc_score
                
            # ADDITIONAL CONTEXT - Lower weight (10% of context score)
            if context.additional_context:
                additional_matches = 0
                total_additional_items = 0
                
                for key, value in context.additional_context.items():
                    if isinstance(value, list):
                        for item in value[:3]:  # Limit to top 3 items per key
                            total_additional_items += 1
                            if str(item).lower() in combined_content:
                                additional_matches += 1
                    elif isinstance(value, str) and len(value) < 50:
                        total_additional_items += 1
                        if value.lower() in combined_content:
                            additional_matches += 1
                
                if total_additional_items > 0:
                    additional_score = (additional_matches / total_additional_items) * 0.10
                    context_score += additional_score
            
            # Cap context score at 1.0
            context_score = min(context_score, 1.0)
            
            # SMART WEIGHTING BASED ON TRUST LEVEL
            if is_high_trust:
                # High-trust domains: Give more weight to Tavily (they're usually accurate)
                tavily_weight = 0.65
                context_weight = 0.35
                trust_bonus = 1.08  # 8% bonus for credible sources
                score_category = "high-trust"
            else:
                # Regular domains: Give more weight to context (need stronger validation)
                tavily_weight = 0.55
                context_weight = 0.45
                trust_bonus = 1.0   # No bonus
                score_category = "standard"
            
            # CALCULATE BASE SCORE
            base_score = (tavily_score * tavily_weight) + (context_score * context_weight)
            
            # APPLY TRUST BONUS AND QUALITY THRESHOLDS
            if is_high_trust and tavily_score >= 0.8 and context_score >= 0.3:
                # Premium scoring: High-trust + high Tavily + decent context
                candidate.confidence_score = min(base_score * 1.12, 0.95)  # 12% bonus
                score_reason = f"Premium: High-trust + strong signals (T:{tavily_score:.3f}, C:{context_score:.3f})"
                
            elif is_high_trust and tavily_score >= 0.7:
                # Good high-trust source
                candidate.confidence_score = min(base_score * trust_bonus, 0.90)
                score_reason = f"High-trust domain (T:{tavily_score:.3f}, C:{context_score:.3f})"
                
            elif tavily_score >= 0.8 and context_score >= 0.4:
                # Strong signals across the board  
                candidate.confidence_score = min(base_score * 1.05, 0.85)  # 5% bonus
                score_reason = f"Strong overall signals (T:{tavily_score:.3f}, C:{context_score:.3f})"
                
            else:
                # Standard scoring
                candidate.confidence_score = min(base_score * trust_bonus, 0.80)
                score_reason = f"{score_category.title()} scoring (T:{tavily_score:.3f}, C:{context_score:.3f})"
            
            # QUALITY FLOOR: Minimum thresholds
            if context_score < 0.1 and tavily_score < 0.5:
                candidate.confidence_score = min(candidate.confidence_score, 0.25)  # Cap low-quality results
                score_reason += " [quality-capped]"
            
            # Store debug info
            candidate.context_match = {
                "fixed_score": candidate.confidence_score,
                "tavily_score": tavily_score,
                "context_score": context_score,
                "is_high_trust": is_high_trust,
                "tavily_weight": tavily_weight,
                "context_weight": context_weight,
                "trust_bonus": trust_bonus,
                "score_reason": score_reason
            }
            
            print(f"Candidate: {candidate.name[:50]}...")
            print(f"  Tavily score: {tavily_score:.3f}")
            print(f"  Context score: {context_score:.3f}")
            print(f"  High trust domain: {is_high_trust}")
            print(f"  Weights: T={tavily_weight:.2f}, C={context_weight:.2f}")
            print(f"  Trust bonus: {trust_bonus:.2f}x")
            print(f"  Final score: {candidate.confidence_score:.3f}")
            print(f"  Reason: {score_reason}")
            print("---")
        
        # Sort by confidence
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        return candidates
    
    def _apply_context_penalties(self, candidates: List[EntityCandidate], context: EntityContext) -> List[EntityCandidate]:
        """Apply smart semantic penalties for clear contradictions using EmbeddingService"""
        
        # Early return if no context or candidates
        if not candidates:
            return candidates
        
        if not any([context.aliases, context.roles, context.organizations, context.locations]):
            self.logger.debug("No context available - skipping penalties")
            return candidates
        
        self.logger.info(f"🔍 APPLYING SEMANTIC CONTEXT PENALTIES to {len(candidates)} candidates")
        self.logger.info(f"Available context: {self._count_context_richness(context)} total items")
        
        # Configuration constants
        class PenaltyConfig:
            # Content length thresholds
            MIN_CONTENT_FOR_ALIAS_PENALTY = 150
            EXTENSIVE_CONTENT_THRESHOLD = 300
            
            # Alias penalties
            ALIAS_MISSING_BIOGRAPHY = 0.25
            ALIAS_FUZZY_BIOGRAPHY = 0.15
            ALIAS_MISSING_EXTENSIVE = 0.15
            ALIAS_FUZZY_EXTENSIVE = 0.08
            
            # Semantic similarity thresholds
            ROLE_VERY_LOW_THRESHOLD = 0.25
            ROLE_MEDIUM_LOW_THRESHOLD = 0.45
            ORG_LOW_THRESHOLD = 0.30
            ORG_MEDIUM_THRESHOLD = 0.55
            # Semantic penalties
            ROLE_MISMATCH_PENALTY = 0.20
            ROLE_WEAK_PENALTY = 0.10
            ORG_MISMATCH_PENALTY = 0.15
            ORG_WEAK_PENALTY = 0.08
            
            # Other limits
            MINIMUM_FINAL_SCORE = 0.1
            MAX_TOTAL_PENALTY = 0.50  # Cap total penalty
        
        for candidate in candidates:
            original_score = candidate.confidence_score
            penalty = 0.0
            penalty_reasons = []
            
            # Get content for analysis
            content = candidate.description.lower() if candidate.description else ""
            title = candidate.metadata.get("title", "").lower()
            full_text = f"{title} {content}".strip()
            
            if not full_text:
                self.logger.debug(f"No content for candidate '{candidate.name}' - skipping penalties")
                continue
            
            # ===================================================================
            # 1. ALIAS MISMATCH - Enhanced exact matching with fuzzy backup
            # ===================================================================
            if context.aliases and len(context.aliases) > 0:
                # Check for exact mentions first
                exact_alias_mentioned = any(alias.lower() in full_text for alias in context.aliases)
                
                # Check for fuzzy matches (partial names, common variations)
                fuzzy_alias_mentioned = False
                if not exact_alias_mentioned:
                    for alias in context.aliases:
                        alias_parts = [part for part in alias.lower().split() if len(part) > 2]
                        if len(alias_parts) > 1:  # Multi-word alias
                            # Check if all significant parts appear
                            if all(part in full_text for part in alias_parts):
                                fuzzy_alias_mentioned = True
                                break
                            # Check if last name (usually most significant) appears
                            if len(alias_parts[-1]) > 3 and alias_parts[-1] in full_text:
                                fuzzy_alias_mentioned = True
                                break
                
                alias_mentioned = exact_alias_mentioned or fuzzy_alias_mentioned
                
                # Apply penalty if substantial content exists but no alias mentioned
                if not alias_mentioned and len(full_text) > PenaltyConfig.MIN_CONTENT_FOR_ALIAS_PENALTY:
                    biographical_indicators = ['biography', 'profile', 'about', 'born', 'career', 'known for', 'life', 'personal']
                    seems_biographical = any(indicator in full_text for indicator in biographical_indicators)
                    
                    if seems_biographical:
                        current_penalty = PenaltyConfig.ALIAS_FUZZY_BIOGRAPHY if fuzzy_alias_mentioned else PenaltyConfig.ALIAS_MISSING_BIOGRAPHY
                        penalty += current_penalty
                        penalty_type = "alias_fuzzy_biography" if fuzzy_alias_mentioned else "alias_missing_biography"
                        penalty_reasons.append(f"{penalty_type}(-{current_penalty:.2f})")
                        self.logger.debug(f"  ❌ Alias issue in biographical content: Expected {context.aliases}")
                    elif len(full_text) > PenaltyConfig.EXTENSIVE_CONTENT_THRESHOLD:
                        current_penalty = PenaltyConfig.ALIAS_FUZZY_EXTENSIVE if fuzzy_alias_mentioned else PenaltyConfig.ALIAS_MISSING_EXTENSIVE
                        penalty += current_penalty
                        penalty_type = "alias_fuzzy_extensive" if fuzzy_alias_mentioned else "alias_missing_extensive"
                        penalty_reasons.append(f"{penalty_type}(-{current_penalty:.2f})")
                        self.logger.debug(f"  ❌ Alias issue in extensive content: Expected {context.aliases}")
            
            # ===================================================================
            # 2. ROLE/PROFESSION SEMANTIC MISMATCH - Robust approach
            # ===================================================================
            if context.roles and len(context.roles) > 0 and self.embedding_service:
                try:
                    # Use robust embedding method
                    if hasattr(self.embedding_service, 'max_similarity_with_list_robust'):
                        role_similarity, embedding_success = self.embedding_service.max_similarity_with_list_robust(
                            full_text, context.roles, text_limit=400
                        )
                    else:
                        # Fallback to original method
                        role_similarity = self.embedding_service.max_similarity_with_list(
                            full_text, context.roles, text_limit=400
                        )
                        embedding_success = True
                    
                    if embedding_success:
                        self.logger.debug(f"  🔍 Role similarity: {role_similarity:.3f}")
                        
                        # Apply penalty based on similarity threshold
                        if role_similarity < PenaltyConfig.ROLE_VERY_LOW_THRESHOLD:
                            penalty += PenaltyConfig.ROLE_MISMATCH_PENALTY
                            penalty_reasons.append(f"role_semantic_mismatch(-{PenaltyConfig.ROLE_MISMATCH_PENALTY:.2f})")
                            self.logger.debug(f"  ❌ Role semantic mismatch: Expected {context.roles} (similarity: {role_similarity:.3f})")
                        elif role_similarity < PenaltyConfig.ROLE_MEDIUM_LOW_THRESHOLD:
                            penalty += PenaltyConfig.ROLE_WEAK_PENALTY
                            penalty_reasons.append(f"role_semantic_weak(-{PenaltyConfig.ROLE_WEAK_PENALTY:.2f})")
                            self.logger.debug(f"  ⚠️ Role semantic weak match: Expected {context.roles} (similarity: {role_similarity:.3f})")
                        else:
                            self.logger.debug(f"  ✅ Role semantic match: Expected {context.roles} (similarity: {role_similarity:.3f})")
                    else:
                        self.logger.warning("Role embedding calculation failed - skipping role penalty")
                        
                except Exception as e:
                    self.logger.warning(f"Role semantic similarity check failed: {e}")
            
            # ===================================================================
            # 3. ORGANIZATION SEMANTIC MISMATCH - Robust approach
            # ===================================================================
            if context.organizations and len(context.organizations) > 0 and self.embedding_service:
                try:
                    # Use robust embedding method
                    if hasattr(self.embedding_service, 'max_similarity_with_list_robust'):
                        org_similarity, embedding_success = self.embedding_service.max_similarity_with_list_robust(
                            full_text, context.organizations, text_limit=400
                        )
                    else:
                        # Fallback to original method
                        org_similarity = self.embedding_service.max_similarity_with_list(
                            full_text, context.organizations, text_limit=400
                        )
                        embedding_success = True
                    
                    if embedding_success:
                        self.logger.debug(f"  🔍 Organization similarity: {org_similarity:.3f}")
                        
                        # Apply penalty based on similarity threshold
                        if org_similarity < PenaltyConfig.ORG_LOW_THRESHOLD:
                            penalty += PenaltyConfig.ORG_MISMATCH_PENALTY
                            penalty_reasons.append(f"org_semantic_mismatch(-{PenaltyConfig.ORG_MISMATCH_PENALTY:.2f})")
                            self.logger.debug(f"  ❌ Organization semantic mismatch: Expected {context.organizations} (similarity: {org_similarity:.3f})")
                        elif org_similarity < PenaltyConfig.ORG_MEDIUM_THRESHOLD:
                            penalty += PenaltyConfig.ORG_WEAK_PENALTY
                            penalty_reasons.append(f"org_semantic_weak(-{PenaltyConfig.ORG_WEAK_PENALTY:.2f})")
                            self.logger.debug(f"  ⚠️ Organization semantic weak match: Expected {context.organizations} (similarity: {org_similarity:.3f})")
                        else:
                            self.logger.debug(f"  ✅ Organization semantic match: Expected {context.organizations} (similarity: {org_similarity:.3f})")
                    else:
                        self.logger.warning("Organization embedding calculation failed - skipping org penalty")
                        
                except Exception as e:
                    self.logger.warning(f"Organization semantic similarity check failed: {e}")
            
            # ===================================================================
            # Location penalties removed - too ambiguous for reliable matching
            # Focus on more reliable signals: aliases, roles, and organizations
            # ===================================================================
            # ===================================================================
            # APPLY PENALTY with safeguards
            # ===================================================================
            if penalty > 0:
                # Cap total penalty to prevent over-penalization
                penalty = min(penalty, PenaltyConfig.MAX_TOTAL_PENALTY)
                
                # Apply penalty but maintain minimum score
                candidate.confidence_score = max(
                    original_score - penalty, 
                    PenaltyConfig.MINIMUM_FINAL_SCORE
                )
                
                # Add penalty info to metadata for debugging
                candidate.metadata["context_penalty_applied"] = penalty
                candidate.metadata["context_penalty_reasons"] = penalty_reasons
                candidate.metadata["original_score_before_penalty"] = original_score
                
                self.logger.info(f"  ⚖️ SEMANTIC PENALTY APPLIED: '{candidate.name[:40]}...'")
                self.logger.info(f"     {original_score:.3f} → {candidate.confidence_score:.3f} (penalty: {penalty:.3f})")
                self.logger.info(f"     Reasons: {', '.join(penalty_reasons)}")
            else:
                # No penalty - good semantic alignment
                self.logger.debug(f"  ✅ No penalty for '{candidate.name[:30]}...': Good semantic alignment")
        
        # Re-sort candidates after penalty application
        candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        self.logger.info(f"✅ SEMANTIC CONTEXT PENALTIES COMPLETE")
        if candidates:
            self.logger.info(f"   New top candidate: '{candidates[0].name}' (score: {candidates[0].confidence_score:.3f})")
        
        return candidates

    def _make_decision(self, candidates: List[EntityCandidate], entity_name: str, context: EntityContext) -> DisambiguationResult:
        """Apply decision logic with professional KYC standards using full context richness"""
        
        # Use the proper context richness calculation
        context_richness = self._count_context_richness(context)
        
        # ADD THIS DEBUG BLOCK AT THE START:
        print(f"=== DECISION MAKING DEBUG ===")
        print(f"Number of candidates: {len(candidates)}")
        if candidates:
            print(f"Top candidate: {candidates[0].name}")
            print(f"Top confidence: {candidates[0].confidence_score:.3f}")
            print(f"Confidence threshold: {self.confidence_threshold}")
            print(f"Review threshold: {self.review_threshold}")
            if len(candidates) > 1:
                print(f"Second candidate: {candidates[1].name}")
                print(f"Second confidence: {candidates[1].confidence_score:.3f}")
                print(f"Confidence gap: {candidates[0].confidence_score - candidates[1].confidence_score:.3f}")
        print(f"Context richness: {context_richness} total items")
        print("============================")

        # NEW: Entity consolidation step (if you have this method)
        # self.logger.debug(f"🔍 DECISION MAKING: Before consolidation - {len(candidates)} candidates")
        # candidates = self._consolidate_same_entities(candidates)
        # self.logger.debug(f"🔍 DECISION MAKING: After consolidation - {len(candidates)} candidates")

        if not candidates:
            return DisambiguationResult(
                status="no_matches",
                reasoning="No candidates found"
            )
        
        top_candidate = candidates[0]
        
        self.logger.debug(f"Decision making: Top confidence={top_candidate.confidence_score:.3f}, Context richness={context_richness}")
        
        # Case 1: HIGH CONFIDENCE - Always accept regardless of context
        if top_candidate.confidence_score >= self.review_threshold:
            if len(candidates) == 1 or candidates[1].confidence_score < self.confidence_threshold:
                return DisambiguationResult(
                    status="resolved",
                    resolved_entity=top_candidate,
                    all_candidates=candidates,
                    confidence_score=top_candidate.confidence_score,
                    reasoning=f"High confidence match: {top_candidate.confidence_score:.3f}"
                )
        
        # Case 2: Multiple high confidence matches (conflict)
        high_conf_candidates = [c for c in candidates if c.confidence_score >= self.confidence_threshold]
        if len(high_conf_candidates) >= 2:
            confidence_gap = candidates[0].confidence_score - candidates[1].confidence_score
            if confidence_gap < 0.05:  # Very close scores
                return DisambiguationResult(
                    status="needs_review",
                    all_candidates=candidates,
                    confidence_score=top_candidate.confidence_score,
                    reasoning="Multiple high-confidence matches found",
                    review_reason=f"Top 2 candidates have similar confidence: {candidates[0].confidence_score:.3f} vs {candidates[1].confidence_score:.3f}"
                )
        
        # Case 3: MEDIUM CONFIDENCE with RICH CONTEXT - Accept (now uses full context richness)
        if top_candidate.confidence_score >= self.confidence_threshold and context_richness >= 3:
            return DisambiguationResult(
                status="resolved",
                resolved_entity=top_candidate,
                all_candidates=candidates,
                confidence_score=top_candidate.confidence_score,
                reasoning=f"Good match with rich context: {top_candidate.confidence_score:.3f} (context: {context_richness} items)"
            )
        
        # Case 4: MEDIUM CONFIDENCE but LIMITED CONTEXT - Standard threshold
        if top_candidate.confidence_score >= self.confidence_threshold:
            return DisambiguationResult(
                status="resolved",
                resolved_entity=top_candidate,
                all_candidates=candidates,
                confidence_score=top_candidate.confidence_score,
                reasoning=f"Acceptable confidence match: {top_candidate.confidence_score:.3f}"
            )
        
        # Case 5: DECENT MATCH but NO CONTEXT - Ask for more info (PROFESSIONAL APPROACH)
        if context_richness == 0 and top_candidate.confidence_score >= 0.3:
            return DisambiguationResult(
                status="needs_context",  # NEW STATUS
                all_candidates=candidates,
                confidence_score=top_candidate.confidence_score,
                reasoning="Additional context required for confident identification",
                review_reason="Please provide additional information like date of birth, address, occupation, or nationality for accurate identification"
            )
        
        # Case 6: LOW CONFIDENCE - Manual review
        return DisambiguationResult(
            status="needs_review",
            all_candidates=candidates,
            confidence_score=top_candidate.confidence_score,
            reasoning="Requires manual review",
            review_reason=f"Confidence {top_candidate.confidence_score:.3f} below threshold {self.confidence_threshold}"
        )

    # Helper methods (enhanced versions)
    def _consolidate_same_entities(self, candidates: List[EntityCandidate], 
                                    similarity_threshold: float = 0.90) -> List[EntityCandidate]:
        """IMPROVED: More conservative consolidation threshold"""
        
        if len(candidates) <= 1:
            return candidates
        
        self.logger.debug(f"🔄 ENTITY CONSOLIDATION: Starting with {len(candidates)} candidates")
        
        # Group candidates by entity similarity
        entity_groups = []
        used_indices = set()
        
        for i, candidate in enumerate(candidates):
            if i in used_indices:
                continue
                
            # Start a new group with this candidate
            current_group = [candidate]
            used_indices.add(i)
            
            # Find similar candidates
            for j, other_candidate in enumerate(candidates[i+1:], start=i+1):
                if j in used_indices:
                    continue
                    
                try:
                    similarity = self._calculate_name_similarity(candidate.name, other_candidate.name)
                    self.logger.debug(f"  Name similarity: '{candidate.name}' vs '{other_candidate.name}' = {similarity:.3f}")
                    
                    # IMPROVED: Higher threshold + additional checks
                    if (similarity >= similarity_threshold and 
                        self._sources_are_compatible(candidate, other_candidate)):
                        current_group.append(other_candidate)
                        used_indices.add(j)
                        self.logger.debug(f"    ✅ GROUPED: Added '{other_candidate.name}' to group with '{candidate.name}'")
                except Exception as e:
                    self.logger.warning(f"Similarity calculation failed: {e}")
                    continue
            
            entity_groups.append(current_group)
        
        # Consolidate each group (keep best, don't just pick by confidence)
        consolidated_candidates = []
        
        for group_idx, group in enumerate(entity_groups):
            if len(group) == 1:
                consolidated_candidates.append(group[0])
            else:
                self.logger.info(f"🔗 CONSOLIDATING GROUP {group_idx+1}: {len(group)} candidates")
                
                # IMPROVED: Pick best candidate by multiple criteria
                best_candidate = max(group, key=lambda x: (
                    x.metadata.get("search_score", 0.0),  # Tavily score first
                    1 if any(domain in x.source_url for domain in ENTITY_CREDIBLE_SOURCES) else 0,  # Domain trust
                    len(x.description),  # Content richness
                    x.confidence_score  # Final confidence
                ))
                
                # Apply source diversity boost
                original_confidence = best_candidate.confidence_score
                source_diversity_bonus = min(len(group) * 0.02, 0.1)  # Max 10% boost
                boosted_confidence = min(original_confidence + source_diversity_bonus, 1.0)
                
                # Create consolidated candidate
                consolidated_candidate = EntityCandidate(
                    name=best_candidate.name,
                    confidence_score=boosted_confidence,
                    context_match=best_candidate.context_match.copy(),
                    description=best_candidate.description,
                    source_url=best_candidate.source_url,
                    search_snippet=best_candidate.search_snippet,
                    metadata=best_candidate.metadata.copy()
                )
                
                # Add consolidation metadata
                consolidated_candidate.metadata["consolidated_from"] = len(group)
                consolidated_candidate.metadata["source_diversity_boost"] = source_diversity_bonus
                consolidated_candidate.metadata["all_sources"] = [c.source_url for c in group]
                
                consolidated_candidates.append(consolidated_candidate)
                
                self.logger.info(f"  ✅ BEST: '{best_candidate.name}' from {best_candidate.metadata.get('domain', 'unknown')}")
        
        # Sort by confidence
        consolidated_candidates.sort(key=lambda x: x.confidence_score, reverse=True)
        
        self.logger.info(f"🎯 CONSOLIDATION COMPLETE: {len(candidates)} → {len(consolidated_candidates)} candidates")
        
        return consolidated_candidates


    # HELPER: Check if sources are compatible for consolidation
    def _sources_are_compatible(self, candidate1: EntityCandidate, candidate2: EntityCandidate) -> bool:
        """Check if two candidates should be consolidated based on source compatibility"""
        
        # Don't consolidate if one is clearly more authoritative
        domain1 = candidate1.metadata.get("domain", "").lower()
        domain2 = candidate2.metadata.get("domain", "").lower()
        
        # LinkedIn profiles are unique, don't consolidate different LinkedIn profiles
        if "linkedin.com" in domain1 and "linkedin.com" in domain2:
            # Only consolidate if they're the exact same LinkedIn URL
            return candidate1.source_url == candidate2.source_url
        
        # Don't consolidate official institutional sources with general directories
        official_domains = ["edu", "gov", "org"]
        domain1_official = any(tld in domain1 for tld in official_domains)
        domain2_official = any(tld in domain2 for tld in official_domains)
        
        if domain1_official != domain2_official:
            return False
        
        return True
    def _safe_json_parse(self, json_str: str, default):
        """Safely parse JSON with fallback"""
        try:
            return json.loads(json_str)
        except:
            self.logger.debug(f"Failed to parse JSON string: {json_str}. Returning default: {default}")
            return default

    def _deduplicate_preserving_order(self, items: List[str]) -> List[str]:
        """Deduplicate list while preserving order and handling case variations"""
        seen = set()
        result = []
        for item in items:
            item_clean = item.strip()
            if item_clean:  # Skip empty strings
                item_lower = item_clean.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    result.append(item_clean)
        return result

    def _deduplicate_candidates(self, candidates: List[EntityCandidate]) -> List[EntityCandidate]:
        """Enhanced deduplication that preserves quality candidates"""
        seen_urls = set()
        unique_candidates = []
        name_groups = {}  # Group by name, then pick best
        
        for candidate in candidates:
            # Skip duplicate URLs (exact same source)
            if candidate.source_url in seen_urls:
                self.logger.debug(f"❌ Skipping duplicate URL: {candidate.source_url}")
                continue
                
            seen_urls.add(candidate.source_url)
            
            # Group by normalized name instead of rejecting
            name_key = candidate.name.lower().strip()
            
            if name_key not in name_groups:
                name_groups[name_key] = []
            
            name_groups[name_key].append(candidate)
        
        # For each name group, pick the best candidate based on quality
        for name_key, group_candidates in name_groups.items():
            if len(group_candidates) == 1:
                unique_candidates.append(group_candidates[0])
            else:

                self.logger.info(f"🔄 DEDUP GROUP '{name_key}': {len(group_candidates)} candidates")
                for i, candidate in enumerate(group_candidates):
                    self.logger.info(f"   Option {i+1}: {candidate.source_url} (Score: {candidate.metadata.get('search_score', 0.0):.3f})")
            
                # Multiple candidates with same name - pick the best one
                best_candidate = max(group_candidates, key=lambda c: (
                    c.metadata.get("search_score", 0.0),  # Tavily score first
                    c.metadata.get("quality_score", 0.0),  # Quality score second
                    len(c.description)  # Content length as tiebreaker
                ))
                self.logger.info(f"   ✅ SELECTED: {best_candidate.source_url}")
                
                self.logger.info(f"🔄 DEDUP: Found {len(group_candidates)} candidates for '{name_key}'")
                self.logger.info(f"   Selected: {best_candidate.metadata.get('domain', 'unknown')} "
                            f"(Tavily: {best_candidate.metadata.get('search_score', 0.0):.3f})")
                
                unique_candidates.append(best_candidate)
        
        self.logger.info(f"✅ DEDUPLICATION: {len(candidates)} → {len(unique_candidates)} candidates")
        return unique_candidates

    def _extract_name_from_result(self, result: Dict) -> str:
        """Extract entity name using NER + smart title parsing"""
        title = result.get("title", "")
        content = result.get("content", "")
        
        # Step 1: Try NER on title first (most reliable)
        if title:
            try:
                # Use your existing NER service to find person names in title
                person_names = self.ner_service.get_persons(title)
                if person_names:
                    # Get the longest person name (usually most complete)
                    best_name = max(person_names, key=len)
                    if len(best_name.split()) >= 2:  # At least first + last name
                        return best_name.strip()
            except Exception as e:
                self.logger.debug(f"NER failed on title: {e}")
        
        # Step 2: Fallback to smart title parsing
        if title:
            cleaned_name = self._clean_title_for_name(title)
            if cleaned_name and cleaned_name != "Unknown":
                return cleaned_name
        
        # Step 3: Try NER on content snippet (first 200 chars)
        if content:
            try:
                content_snippet = content[:200]  # Don't process entire content
                person_names = self.ner_service.get_persons(content_snippet)
                if person_names:
                    best_name = max(person_names, key=len)
                    if len(best_name.split()) >= 2:
                        return best_name.strip()
            except Exception as e:
                self.logger.debug(f"NER failed on content: {e}")
        
        # Step 4: Final fallback
        return "Unknown Entity"

    def _clean_title_for_name(self, title: str) -> str:
        """Simple but robust title cleaning - no over-engineering"""
        if not title:
            return "Unknown"
        
        cleaned = title.strip()
        
        # Remove common patterns - ONE pass, simple regex
        patterns_to_remove = [
            r' - (LinkedIn|Wikipedia|Biography|Profile|About).*$',
            r' \| (LinkedIn|Wikipedia|Biography|Profile|About).*$', 
            r' - .*\.(com|org|net).*$',  # Remove website suffixes
            r'\s*\(\d{4}.*?\)$',  # Remove years in parentheses
            r'^(Profile|Biography|About|Meet)\s+',  # Remove prefixes
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
            cleaned = cleaned.strip()
        
        # Handle separators - take first meaningful part
        if ' - ' in cleaned:
            parts = cleaned.split(' - ')
            # Pick the part that looks most like a name (shortest, proper case)
            name_part = min(parts, key=len).strip()
            if name_part and len(name_part.split()) >= 2:
                cleaned = name_part
        
        # Final cleanup
        cleaned = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', cleaned)  # Remove leading/trailing non-word chars
        
        return cleaned.strip() if cleaned.strip() else "Unknown"

    def _select_best_name_part(self, parts: List[str]) -> str:
        """Intelligently select the part that looks most like an entity name"""
        if not parts:
            return ""
        
        # Score each part
        best_part = ""
        best_score = 0
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            score = 0
            
            # Prefer shorter parts (names vs descriptions)
            if len(part) <= 50:
                score += 2
            elif len(part) <= 100:
                score += 1
            
            # Prefer parts with capitalized words (proper nouns)
            words = part.split()
            capitalized_words = sum(1 for word in words if word[0].isupper())
            if len(words) > 0:
                score += (capitalized_words / len(words)) * 3
            
            # Penalize parts with common descriptive words
            descriptive_words = [
                "biography", "profile", "about", "company", "corporation", 
                "group", "limited", "inc", "ltd", "plc", "llc",
                "news", "article", "story", "report"
            ]
            lower_part = part.lower()
            for desc_word in descriptive_words:
                if desc_word in lower_part:
                    score -= 1
            
            # Prefer parts that don't contain years
            if not re.search(r'\b(19|20)\d{2}\b', part):
                score += 1
            
            if score > best_score:
                best_score = score
                best_part = part
        
        return best_part

    def _clean_extracted_name(self, name: str) -> str:
        """Final cleanup for extracted names"""
        if not name:
            return "Unknown"
        
        # Remove common prefixes that might remain
        prefixes_to_remove = [
            "according to", "known as", "called", "named",
            "the company", "the group", "the organization"
        ]
        
        name_lower = name.lower()
        for prefix in prefixes_to_remove:
            if name_lower.startswith(prefix):
                name = name[len(prefix):].strip()
                break
        
        # Remove trailing punctuation except necessary ones
        name = re.sub(r'[,;:]+$', '', name)
        
        return name.strip() if name.strip() else "Unknown"

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate semantic similarity between names using EmbeddingService with caching"""
        if name1.lower() == name2.lower():
            return 1.0
        
        # Check cache
        cache_key = f"{name1}||{name2}"
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            # Using EmbeddingService's similarity method
            similarity = self.embedding_service.similarity(name1, name2)
            
            # Cache the result
            self._embedding_cache[cache_key] = float(similarity)
            
            # Limit cache size
            if len(self._embedding_cache) > 200:
                # Remove oldest entry
                oldest_key = next(iter(self._embedding_cache))
                del self._embedding_cache[oldest_key]
            
            return float(similarity)
        except Exception as e:
            self.logger.warning(f"Name similarity calculation failed: {e}")
            return 0.0

    def _score_context_match(self, text: str, context_items: List[str]) -> float:
        """Score semantic similarity between text and context items using EmbeddingService"""
        if not context_items:
            return 0.0
        
        try:
            # Using EmbeddingService's max_similarity_with_list method
            max_sim = self.embedding_service.max_similarity_with_list(text, context_items, text_limit=300)
            return float(max_sim)
        except Exception as e:
            self.logger.warning(f"Context match scoring failed: {e}")
            return 0.0

    def _score_keyword_match(self, text: str, keywords: List[str]) -> float:
        """Score keyword matching in text"""
        if not keywords:
            return 0.0
        
        text_lower = text.lower()
        matches = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        return min(matches / len(keywords), 1.0)

    def __del__(self):
        """Cleanup thread pool on destruction"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)