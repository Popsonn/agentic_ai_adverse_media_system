# search_strategy_agent.py

from dataclasses import dataclass
from typing import List, Dict, Optional
import time

# Import from models instead of defining locally
from models.search import SearchStrategy, SearchQuality, SearchResult
from core.state import AdverseMediaState, AgentStatus
from core.base_agent import BaseAgent
from core.exceptions import (
    AgentExecutionError,
    SearchError,
    APIError,
    ErrorSeverity
)
from config.settings import BaseAgentConfig
from services.tavily_client import TavilyService
from services.embedding_service import EmbeddingService
from config.constants import CREDIBLE_SOURCES, EXCLUDED_DOMAINS, ADVERSE_KEYWORDS
import logging
from models.classification import AdverseMediaCategory
from config.settings import SearchStrategyConfig

# ------------------- MAIN AGENT CLASS -------------------

class SearchStrategyAgent(BaseAgent):
    """
    Agent responsible for executing adaptive search strategies to find adverse media content.
    Inherits from BaseAgent and uses TavilyService for search operations.
    Uses EmbeddingService for semantic similarity operations.
    """

    def __init__(self, config: SearchStrategyConfig, logger: logging.Logger):
        """
        Initialize the SearchStrategyAgent.
        
        Args:
            config: SearchStrategyConfig instance with agent-specific settings
            logger: Logger instance for this agent
        """
        super().__init__(config, logger)
        
        # Validate config type
        if not isinstance(config, SearchStrategyConfig):
            raise AgentExecutionError(
                f"Config must be SearchStrategyConfig, got {type(config)}",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL,
                context={"expected_type": "SearchStrategyConfig", "received_type": type(config).__name__}
            )
        
        self.search_config = config
        self.similarity_threshold = config.similarity_threshold
        self.max_attempts = config.max_search_attempts
        
        # Initialize embedding service instead of direct sentence transformer
        try:
            self.embedding_service = EmbeddingService(config.sentence_transformer_model)
            self.logger.debug(f"Initialized EmbeddingService with model: {config.sentence_transformer_model}")
        except Exception as e:
            raise AgentExecutionError(
                f"Failed to initialize EmbeddingService with model '{config.sentence_transformer_model}': {e}",
                agent_name=self.agent_name,
                severity=ErrorSeverity.HIGH,
                context={"model_name": config.sentence_transformer_model}
            ) from e
        
        # TavilyService will be initialized when needed
        self.tavily_service = None

        # Use adverse keywords from constants
        self.adverse_keywords = ADVERSE_KEYWORDS

    def _run_implementation(self, state: AdverseMediaState) -> AdverseMediaState:
        agent_name = "search_agent"
        state.start_agent(agent_name)
        self._update_agent_status(state, f"Starting adaptive search process for entity: {getattr(state.resolved_entity, 'name', 'Unknown')}")

        # Initialize TavilyService if not initialized yet
        if not self.tavily_service:
            try:
                self._initialize_tavily_service()
            except Exception as e:
                error = AgentExecutionError(
                    f"Failed to initialize TavilyService: {e}",
                    agent_name=agent_name,
                    severity=ErrorSeverity.CRITICAL,
                )
                self._handle_error(state, error)
                state.fail_agent(agent_name, error)
                return state

        # Confirm resolved_entity presence
        if not state.resolved_entity:
            error = SearchError(
                "No resolved entity found from disambiguation agent",
                agent_name=agent_name,
                severity=ErrorSeverity.HIGH,
                context={"state_id": getattr(state, "id", "unknown")},
            )
            self._handle_error(state, error)
            state.fail_agent(agent_name, error)
            return state

        try:
            # Execute search strategy (includes search attempts and filtering)
            state = self._execute_search_strategy(state)

            # Always complete the agent with the results
            output_data = {
                "raw_search_results": state.raw_search_results,
                "filtered_search_results": state.filtered_search_results,
                "search_quality_metrics": state.search_quality_metrics,
                "strategies_attempted": getattr(state, "strategies_attempted", []),
                "skip_to_final_report": getattr(state, "skip_to_final_report", False)
            }
            
            state.complete_agent(agent_name, AgentStatus.COMPLETED, output_data=output_data)
            
            # Log the decision for debugging
            if getattr(state, "skip_to_final_report", False):
                self._update_agent_status(state, 
                    f"Clean entity detected - workflow will skip to final report generation")
            
            return state

        except Exception as e:
            error = AgentExecutionError(
                f"Unexpected error during search execution: {e}",
                agent_name=agent_name,
                severity=ErrorSeverity.CRITICAL,
                context={"entity_name": getattr(state.resolved_entity, "name", "unknown")},
            )
            self._handle_error(state, error)
            state.fail_agent(agent_name, error)
            return state

    def _initialize_tavily_service(self):
        """Initialize TavilyService with API key from config"""
        try:
            api_key = self.search_config.tavily_api_key
            if not api_key:
                raise APIError(
                    "Tavily API key not found in config",
                    endpoint="tavily_init",
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.CRITICAL
                )
            
            self.tavily_service = TavilyService(api_key=api_key)
            self.logger.debug("TavilyService initialized successfully")
            
        except Exception as e:
            raise APIError(
                f"Failed to initialize TavilyService: {e}",
                endpoint="tavily_init",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL,
                context={"config_has_api_key": bool(self.search_config.tavily_api_key)}
            ) from e

    def _execute_search_strategy(self, state: AdverseMediaState) -> AdverseMediaState:
        """
        Execute the main search strategy logic with multiple attempts and strategies.
        
        Args:
            state: Current AdverseMediaState
            
        Returns:
            Updated AdverseMediaState with search results
        """
        entity_name = state.resolved_entity.name
        start_time = time.time()

        # Initialize collections - these persist across all attempts
        all_raw_results = []
        best_results = []
        best_quality = SearchQuality(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        attempt = 0
        strategy = SearchStrategy.BROAD
        
        self._update_agent_status(state, f"Starting search for entity: {entity_name}")

        while attempt < self.max_attempts:
            self._update_agent_status(state, f"Attempt {attempt+1} with strategy: {strategy.name}")
            
            try:
                # Build query for current strategy
                query = self._build_query(entity_name, state.user_context, strategy)
                self._update_agent_status(state, f"Query: {query}")

                # Execute search using TavilyService
                raw_results = self._execute_single_search(query, state)
                #state.strategies_attempted.append(strategy.value)
                if not hasattr(state, 'strategies_attempted'):
                     state.strategies_attempted = []
                state.strategies_attempted.append(strategy.value)

                # Always collect raw results from this attempt
                if raw_results:
                    all_raw_results.extend(raw_results)
                    
                    # Extract full content and filter results
                    raw_results = self._extract_full_content(raw_results, state)
                    filtered = self._filter_articles(raw_results, entity_name, state)
                    self._update_agent_status(state, f"Filtered {len(raw_results)} -> {len(filtered)} articles")

                    # Check if we should continue searching
                    if not self._should_continue_search(filtered, entity_name):
                        self._update_agent_status(state, "Results appear irrelevant, stopping search early")
                        break

                    # Evaluate quality of results
                    quality = self._evaluate_quality(filtered, entity_name)
                    self._update_agent_status(state, f"Quality score: {quality.overall_quality:.3f}")

                    # Update best results if this is better
                    if quality.overall_quality > best_quality.overall_quality:
                        best_results = filtered
                        best_quality = quality
                        self._update_agent_status(state, "New best results found")

                    # Check if we've met the quality threshold
                    if quality.overall_quality >= self.search_config.search_quality_threshold:
                        self._update_agent_status(state, "Quality threshold met, stopping search")
                        break
                else:
                    self._update_agent_status(state, "No results found, trying next strategy")

            except Exception as e:
                # Handle errors in individual search attempts
                error = SearchError(
                    f"Error in search attempt {attempt + 1} with strategy {strategy.name}: {e}",
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.MEDIUM,
                    context={"attempt": attempt + 1, "strategy": strategy.name, "entity": entity_name}
                )
                self._handle_error(state, error)
            
            # Always move to next attempt/strategy
            attempt += 1
            strategy = self._next_strategy(strategy)
            if attempt < self.max_attempts:
                time.sleep(1)  # Brief pause between attempts

        # Update state with final results - always safe now
        state.raw_search_results = all_raw_results
        state.filtered_search_results = best_results
        state.search_quality_metrics = {
            "overall_quality": best_quality.overall_quality,
            "article_count": best_quality.article_count,
            "relevance_score": best_quality.relevance_score,
            "source_credibility": best_quality.source_credibility,
            "attempts_made": attempt,
            "clean_entity_likely": False
        }

        # Apply smart low-quality handling
        return self._handle_low_quality_results(state, best_results, best_quality)
    
    def _handle_low_quality_results(self, state: AdverseMediaState, best_results: List[SearchResult], best_quality: SearchQuality) -> AdverseMediaState:
        """
        Handle cases where search results are of low quality or no results found.
        Sets flags for clean entities but doesn't complete the agent - that's done by caller.
        
        Args:
            state: Current AdverseMediaState
            best_results: Best search results found
            best_quality: Quality metrics for the best results
            
        Returns:
            Updated AdverseMediaState with clean entity flags set if applicable
        """
        
        # No results found - entity is likely clean
        if not best_results:
            self._update_agent_status(state, "No adverse media articles found - entity appears clean")
            state.search_quality_metrics.update({
                "overall_quality": 0.0,
                "article_count": 0,
                "relevance_score": 0.0,
                "source_credibility": 0.0,
                "clean_entity_likely": True,
                "skipped_classification_reason": "No articles found",
                "final_assessment": "CLEAN - No adverse media found"
            })
            # Set a flag to indicate this should skip subsequent agents
            state.skip_to_final_report = True
            return state

        # Weak results - entity likely clean
        if best_quality.overall_quality < self.search_config.search_quality_threshold:
            if (best_quality.article_count <= 2 and
                best_quality.keyword_match_score < 0.1 and
                best_quality.relevance_score < 0.3):
                
                self._update_agent_status(state, "Only weak adverse signals found - entity appears clean")
                state.search_quality_metrics.update({
                    "clean_entity_likely": True,
                    "skipped_classification_reason": (
                        f"Weak signals (count: {best_quality.article_count}, "
                        f"relevance: {best_quality.relevance_score:.2f}, "
                        f"keywords: {best_quality.keyword_match_score:.2f})"
                    ),
                    "final_assessment": f"CLEAN - Only {best_quality.article_count} weak articles found"
                })
                # Set flag to skip classification and go to report
                state.skip_to_final_report = True
                return state

        # Results found and quality is decent - continue to classification
        self._update_agent_status(state, 
            f"Search completed with {len(best_results)} articles "
            f"(quality: {best_quality.overall_quality:.3f}) - proceeding to classification"
        )
        state.search_quality_metrics.update({
            "clean_entity_likely": False,
            "final_assessment": f"PROCEED - {len(best_results)} articles found for classification"
        })
        
        return state
    def _execute_single_search(self, query: str, state: AdverseMediaState) -> List[SearchResult]:
        """
        Execute a single search using the TavilyService.
        
        Args:
            query: Search query string
            state: Current AdverseMediaState
            
        Returns:
            List of SearchResult objects
        """
        def _search_operation():
            # Use TavilyService search method with constants
            results = self.tavily_service.search(
                query=query,
                topic="news",
                search_depth="advanced",
                max_results=self.search_config.max_articles,
                days=self.search_config.days_back,
                include_domains=self.search_config.include_domains,
                exclude_domains=self.search_config.exclude_domains
            )

            search_results = []
            for r in results:
                url = r.get("url", "")
                # Skip non-media sources using constants
                if url and r.get("title"):
                    search_results.append(SearchResult(
                        title=r.get("title", ""),
                        url=url,
                        content=r.get("content", ""),
                        published_date=r.get("published_date"),
                        source_domain=self._extract_domain(url),
                        search_strategy_used=query
                    ))
            return search_results

        try:
            # Use base agent's retry mechanism
            results = self._retry_with_backoff(_search_operation)
            self.metrics.api_calls_made += 1
            return results
        except Exception as e:
            raise SearchError(
                f"Search failed: {e}",
                agent_name=self.agent_name,
                severity=ErrorSeverity.HIGH,
                context={"query": query, "tavily_service_initialized": self.tavily_service is not None}
            ) from e
        
    def _build_query(self, entity_name: str, context: Optional[Dict[str, str]], strategy: SearchStrategy) -> str:
        """Build precise search query using crime-type keywords from constants and context"""
        
        # Use adverse keywords from constants based on strategy
        if strategy == SearchStrategy.BROAD:
            keywords = ADVERSE_KEYWORDS[AdverseMediaCategory.FRAUD_FINANCIAL_CRIME][:3] + \
                      ADVERSE_KEYWORDS[AdverseMediaCategory.CORRUPTION_BRIBERY][:2]
        elif strategy == SearchStrategy.TARGETED:
            keywords = ADVERSE_KEYWORDS[AdverseMediaCategory.FRAUD_FINANCIAL_CRIME][:4]
        elif strategy == SearchStrategy.DEEP_DIVE:
            keywords = ["SEC", "DOJ", "fine", "penalty", "sanctions", "investigation"]
        else:  # ALTERNATIVE
            keywords = ["lawsuit", "settlement", "court", "legal action"]
        
        # Build query parts combining entity and keywords
        adverse_queries = [f'"{entity_name}" AND {keyword}' for keyword in keywords[:3]]
        
        # Add context terms if available (role, industry, company, location)
        context_parts = []
        if context:
            for key in ["role", "industry", "company", "location"]:
                val = context.get(key)
                if val:
                    context_parts.append(f'"{val}"')
        
        full_query = " OR ".join(adverse_queries + context_parts)
        return full_query

    def _extract_full_content(self, articles: List[SearchResult], state: AdverseMediaState) -> List[SearchResult]:
        """Extract full article content using TavilyService extract API"""
        if not articles:
            return articles
        
        # Get URLs for extraction
        urls = [article.url for article in articles if article.url]
        
        if not urls:
            return articles
        
        def _extract_operation():
            # Use TavilyService extract method
            url_to_content = self.tavily_service.extract(
                urls=urls,
                extract_depth="advanced",
                format="markdown"
            )
            
            # Update articles with full content
            updated_count = 0
            for article in articles:
                if article.url in url_to_content:
                    article.content = url_to_content[article.url]
                    updated_count += 1
            
            return articles, updated_count
        
        try:
            self._update_agent_status(state, f"Extracting full content from {len(urls)} articles")
            
            # Use base agent's retry mechanism
            articles, updated_count = self._retry_with_backoff(_extract_operation)
            self.metrics.api_calls_made += 1
            
            self._update_agent_status(state, f"Successfully extracted content from {updated_count}/{len(articles)} articles")
            return articles
            
        except Exception as e:
            # Log error but don't fail - return original articles
            error = APIError(
                f"Extract failed: {e}",
                endpoint="tavily_extract",
                agent_name=self.agent_name,
                severity=ErrorSeverity.MEDIUM,
                context={"url_count": len(urls)}
            )
            self._handle_error(state, error)
            return articles

    def _filter_articles(self, articles: List[SearchResult], entity_name: str, state: AdverseMediaState) -> List[SearchResult]:
        """Filter articles by relevance and remove duplicates using EmbeddingService"""
        if not articles:
            return []
        
        entity_lower = entity_name.lower()
        relevant_articles = []
        
        # Filter articles where entity is prominently mentioned
        for article in articles:
            title_lower = article.title.lower()
            content_lower = article.content.lower()
            if (entity_lower in title_lower or content_lower.count(entity_lower) >= 2):
                relevant_articles.append(article)
        
        if not relevant_articles:
            # Return top 3 if no strong matches
            return articles[:3]
        
        if len(relevant_articles) <= 1:
            return relevant_articles
        
        try:
            # Use EmbeddingService for deduplication
            titles = [a.title for a in relevant_articles]
            unique_indices = self.embedding_service.deduplicate_by_similarity(titles, self.similarity_threshold)
            
            # Select articles based on unique indices, limited by max_filtered_articles
            selected = [relevant_articles[i] for i in unique_indices[:self.search_config.max_filtered_articles]]
            
            return selected
            
        except Exception as e:
            # Log error but don't fail - return relevant articles without deduplication
            error = AgentExecutionError(
                f"Error during article filtering: {e}",
                agent_name=self.agent_name,
                severity=ErrorSeverity.MEDIUM,
                context={"article_count": len(relevant_articles), "entity": entity_name}
            )
            self._handle_error(state, error)
            return relevant_articles[:self.search_config.max_filtered_articles]

    def _evaluate_quality(self, articles: List[SearchResult], entity_name: str) -> SearchQuality:
        """Evaluate quality of search results with advanced scoring using constants"""
        if not articles:
            return SearchQuality(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        entity_lower = entity_name.lower()
        
        # Relevance scoring with adverse context boost
        relevance_scores = []
        for article in articles:
            title_lower = article.title.lower()
            content_lower = article.content.lower()
            
            title_mentions = title_lower.count(entity_lower)
            content_mentions = content_lower.count(entity_lower)
            
            # Check adverse context near entity mentions using constants
            adverse_context_score = 0
            content_words = content_lower.split()
            for i, word in enumerate(content_words):
                if entity_lower in word:
                    window = content_words[max(0, i-5):i+5]
                    # Use adverse keywords from constants
                    all_adverse_keywords = [
                        kw for kws in ADVERSE_KEYWORDS.values() for kw in kws
                    ]
                    if any(keyword in " ".join(window) for keyword in all_adverse_keywords):
                        adverse_context_score += 1
            
            base_score = min((title_mentions * 3 + content_mentions * 0.5) / 10, 1.0)
            context_boost = min(adverse_context_score / 3, 0.5)
            relevance_scores.append(base_score + context_boost)
        
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        
        # Source credibility scoring using constants
        credibility_scores = []
        for article in articles:
            if any(src in article.source_domain.lower() for src in CREDIBLE_SOURCES):
                credibility_scores.append(1.0)
            elif article.source_domain.endswith(('.com', '.org', '.net')):
                credibility_scores.append(0.7)
            else:
                credibility_scores.append(0.3)
        avg_credibility = sum(credibility_scores) / len(credibility_scores)
        
        # Keyword matching score using constants
        all_adverse_keywords = [kw for kws in ADVERSE_KEYWORDS.values() for kw in kws]
        keyword_scores = []
        for article in articles:
            content_lower = article.content.lower()
            matches = sum(1 for keyword in all_adverse_keywords if keyword in content_lower)
            keyword_scores.append(min(matches / 5, 1.0))
        avg_keyword_match = sum(keyword_scores) / len(keyword_scores)
        
        # Time coverage - simplified fixed value for now
        time_coverage = 0.8
        
        return SearchQuality(
            article_count=len(articles),
            relevance_score=avg_relevance,
            source_credibility=avg_credibility,
            time_coverage=time_coverage,
            entity_prominence=avg_relevance,
            keyword_match_score=avg_keyword_match
        )

    def _should_continue_search(self, current_results: List[SearchResult], entity_name: str) -> bool:
        """Determine if search should continue or stop early (clean entity)"""
        if not current_results:
            return True  # Try at least two strategies
        
        irrelevant_count = 0
        entity_lower = entity_name.lower()
        for article in current_results:
            content_lower = article.content.lower()
            if (content_lower.count(entity_lower) < 2 or
                any(phrase in content_lower for phrase in ["not related", "different person", "unrelated"])):
                irrelevant_count += 1
        
        # If most results are irrelevant, stop searching
        if irrelevant_count > len(current_results) * 0.7:
            return False
        
        return True

    def _next_strategy(self, current: SearchStrategy) -> SearchStrategy:
        """Cycle to next search strategy"""
        strategy_cycle = {
            SearchStrategy.BROAD: SearchStrategy.TARGETED,
            SearchStrategy.TARGETED: SearchStrategy.DEEP_DIVE,
            SearchStrategy.DEEP_DIVE: SearchStrategy.ALTERNATIVE,
            SearchStrategy.ALTERNATIVE: SearchStrategy.BROAD
        }
        return strategy_cycle[current]

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.lower()
        except:
            return ""

    def _validate_input_state(self, state: AdverseMediaState):
        """
        Validate input state and check for required fields.
        Extends base validation with search-specific requirements.
        """
        # Call parent validation
        super()._validate_input_state(state)

        if not state.resolved_entity:
            raise AgentExecutionError(
            "No resolved entity found - disambiguation agent must complete first",
            agent_name=self.agent_name,
            severity=ErrorSeverity.HIGH,
            context={"workflow_id": getattr(state, "workflow_id", "unknown")}
        )

    # Public interface methods to maintain compatibility
    #def run(self, state: AdverseMediaState) -> AdverseMediaState:
        #"""
        #Public interface method that maintains the original API.
        #Uses BaseAgent's context manager for metrics tracking.
        #"""
        #with self:
            #return super().run(state)