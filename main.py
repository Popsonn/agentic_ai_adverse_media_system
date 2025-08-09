# Add project root to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import logging
import argparse
import json
from typing import Optional, List, Dict, Any
from langgraph.graph import StateGraph
# Your existing imports
from config.settings import BaseAgentConfig, SearchStrategyConfig
from services.brave_search_service import BraveSearchService
from services.embedding_service import EmbeddingService
from services.llm_service import create_llm_service_from_config
from services.logger import LoggerService
from services.ner_service import NERService
from services.tavily_client import TavilyService
from datetime import datetime
from workflow import create_workflow  # Your existing workflow
import uuid

def setup_logging(config: BaseAgentConfig) -> LoggerService:
    log_level = getattr(logging, config.logging_level.upper())
    logger_service = LoggerService(name="AI_Agent", level=log_level)
    return logger_service

def initialize_services(config: BaseAgentConfig, logger_service: LoggerService):
    logger = logger_service.get_logger()
    services = {}
    services['logger'] = logger_service
    try:
        services['brave_search'] = BraveSearchService(
            logger=logger,
            extract_content=True  # Set to False if you don't want content extraction
        )
        logger_service.log_info("BraveSearchService initialized successfully", "ServiceInit") 
    except ValueError as e:
        logger_service.log_error(f"Failed to initialize BraveSearchService: {e}", "ServiceInit")
        logger_service.log_error("Make sure BRAVE_SEARCH_API_KEY environment variable is set", "ServiceInit")
        raise
    except Exception as e:
        logger_service.log_error(f"Unexpected error initializing BraveSearchService: {e}", "ServiceInit")
        raise
    # Initialize EmbeddingService
    try:
        services['embedding'] = EmbeddingService(
            model_name=config.embedding_model_name
        )
        logger_service.log_info(f"EmbeddingService initialized successfully with model: {config.embedding_model_name}", "ServiceInit")
        
    except Exception as e:
        logger_service.log_error(f"Failed to initialize EmbeddingService: {e}", "ServiceInit")
        raise
    # Initialize LLMService
    try:
        services['llm'] = create_llm_service_from_config(config)
        logger_service.log_info("LLMService initialized successfully", "ServiceInit")
        logger_service.log_info(f"Available LLM models: {list(services['llm'].list_models().keys())}", "ServiceInit")   
    except ValueError as e:
        logger_service.log_error(f"Failed to initialize LLMService: {e}", "ServiceInit")
        logger_service.log_error("Make sure required API keys (GROQ_API_KEY/OPENAI_API_KEY) are set", "ServiceInit")
        raise
    except Exception as e:
        logger_service.log_error(f"Unexpected error initializing LLMService: {e}", "ServiceInit")
        raise
    # Initialize NERService
    try:
        ner_model = getattr(config, 'ner_model', None) or "dbmdz/bert-large-cased-finetuned-conll03-english"
        services['ner'] = NERService(model_name=ner_model)
        logger_service.log_info(f"NERService initialized successfully with model: {ner_model}", "ServiceInit") 
    except Exception as e:
        logger_service.log_error(f"Failed to initialize NERService: {e}", "ServiceInit")
        logger_service.log_warning("NER functionality will be unavailable", "ServiceInit")
        # Don't raise - NER might not be critical for all operations
        services['ner'] = None
    # Initialize TavilyService
    try:
        if config.tavily_api_key:
            services['tavily'] = TavilyService(api_key=config.tavily_api_key)
            logger_service.log_info("TavilyService initialized successfully", "ServiceInit")
        else:
            logger_service.log_warning("TAVILY_API_KEY not found, TavilyService will be unavailable", "ServiceInit")
            services['tavily'] = None
            
    except Exception as e:
        logger_service.log_error(f"Failed to initialize TavilyService: {e}", "ServiceInit")
        logger_service.log_warning("Tavily search functionality will be unavailable", "ServiceInit")
        services['tavily'] = None
    return services

class AdverseMediaSystem:    
    def __init__(self, config: BaseAgentConfig):
        self.config = config
        # Setup logging using your existing method
        self.logger_service = setup_logging(config)
        self.logger = self.logger_service.get_logger()
        print("DEBUG: Starting service initialization...")
        # Initialize services using your existing method
        self.services = initialize_services(config, self.logger_service) 
        print("DEBUG: Services initialized successfully")
        print(f"DEBUG: Available services: {list(self.services.keys())}")
        # Create the LangGraph workflow - READY TO USE
        print("DEBUG: Creating workflow...")
        try:
            self.workflow = self._create_workflow()
            print("DEBUG: Workflow created successfully")
            print(f"DEBUG: Workflow type: {type(self.workflow)}")
            print("DEBUG: Compiling workflow...")
            self.compiled_workflow = self.workflow.compile()
            print("DEBUG: Workflow compiled successfully")
            print(f"DEBUG: Compiled workflow type: {type(self.compiled_workflow)}")   
        except Exception as e:
            print(f"DEBUG: Error in workflow creation/compilation: {e}")
            import traceback
            traceback.print_exc()
            raise
        self.logger_service.log_info("AdverseMediaSystem initialized successfully", "SystemInit")

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with all agents and routing logic - READY TO USE"""
        return create_workflow(
            config=self.config,
            logger=self.logger,
            llm_service=self.services['llm'],
            tavily_service=self.services['tavily'],
            brave_search_service=self.services['brave_search'])

    def process_entity(self, entity_name: str, entity_context: Optional[str] = None) -> Dict[str, Any]:
        """Process a single entity through the LangGraph workflow - FIXED VERSION"""
        self.logger_service.log_info(f"Starting adverse media analysis for entity: {entity_name}", "ProcessEntity") 
        try:
            # FIXED: Create initial state as PURE DICT (no AdverseMediaState object)
            initial_state_dict = {
                'entity_name': entity_name,
                'user_input': entity_context or "",
                'workflow_id': str(uuid.uuid4()),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'processing',
                'logs': [],
                'agent_outputs': {},
                'agent_results': {},
                'human_review_required': False,
                'review_reasons': [],
                'agent_errors': [],
                # Initialize all expected fields to avoid KeyError
                'resolved_entity': None,
                'all_candidates': [],
                'disambiguation_confidence': 0.0,
                'disambiguation_result': None,
                'raw_search_results': [],
                'filtered_search_results': [],
                'search_quality_metrics': {},
                'strategies_attempted': [],
                'skip_to_final_report': False,
                'user_context': None,
                # MISSING FIELDS - ADD THESE
                'disambiguation_retry_count': 0,
                'retry_count': 0,
                'classified_articles': [],
                'resolved_articles': [],
                'classification_errors': 0,  # ← ADD THIS LINE
                'conflict_resolution_error': None,  # ← ADD THIS LINE
                'additional_context': None,  # ← ADD THIS LINE
                'context_request': None,  
    
                # Optional but recommended
                'classification_metrics': {},
                'conflict_resolution_summary': {},
            }
            
            print(f"DEBUG: Created initial state dict with keys: {list(initial_state_dict.keys())}")
            
            # Execute the workflow with dictionary
            final_state_dict = self.compiled_workflow.invoke(initial_state_dict)
            print(f"DEBUG: Workflow completed, result type: {type(final_state_dict)}")

                # ADD THE DEBUG CODE HERE - RIGHT AFTER workflow.invoke():
            print("\n=== DEBUG: WHERE ARE MY ARTICLES? ===")
            print(f"resolved_articles count: {len(final_state_dict.get('resolved_articles', []))}")
            print(f"classified_articles count: {len(final_state_dict.get('classified_articles', []))}")
            print(f"filtered_search_results count: {len(final_state_dict.get('filtered_search_results', []))}")
            # Check the actual content
            if final_state_dict.get('resolved_articles'):
                print(f"✅ Found resolved_articles: {len(final_state_dict['resolved_articles'])}")
                print(f"First article: {final_state_dict['resolved_articles'][0].get('classified_article', {}).get('article_title', 'NO TITLE')}")
            else:
                print("❌ No resolved_articles found")
            if final_state_dict.get('classified_articles'):
                print(f"✅ Found classified_articles: {len(final_state_dict['classified_articles'])}")
            else:
                print("❌ No classified_articles found")
            print("=== END DEBUG ===\n")
            
            # FIXED: Extract results directly from dict (no AdverseMediaState conversion)
            results = self._extract_results_from_dict(final_state_dict)
            self.logger_service.log_info(f"Adverse media analysis completed for {entity_name}", "ProcessEntity")
            return results     
            
        except Exception as e:
            print(f"DEBUG: Exception type: {type(e)}")
            print(f"DEBUG: Exception: {e}")
            import traceback
            traceback.print_exc()
            self.logger_service.log_error(f"Workflow failed for entity {entity_name}: {e}", "ProcessEntity")
            raise

    # NEW METHOD: Enhanced entity processing with context request handling
    def process_entity_with_context_support(self, entity_name: str, entity_context: Optional[str] = None, 
                                           additional_context: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Enhanced entity processing with context request handling"""
        self.logger_service.log_info(f"Starting adverse media analysis for entity: {entity_name}", "ProcessEntity")
        
        try:
            # Create initial state as PURE DICT
            initial_state_dict = {
                'entity_name': entity_name,
                'user_input': entity_context or "",
                'workflow_id': str(uuid.uuid4()),
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'status': 'processing',
                'logs': [],
                'agent_outputs': {},
                'agent_results': {},
                'human_review_required': False,
                'review_reasons': [],
                'agent_errors': [],
                
                # Entity disambiguation fields
                'resolved_entity': None,
                'all_candidates': [],
                'disambiguation_confidence': 0.0,
                'disambiguation_result': None,
                
                # Search fields
                'raw_search_results': [],
                'filtered_search_results': [],
                'search_quality_metrics': {},
                'strategies_attempted': [],
                'skip_to_final_report': False,
                'user_context': None,
                
                # Retry counters
                'disambiguation_retry_count': 0,
                'retry_count': 0,
                
                # Classification fields
                'classified_articles': [],
                'resolved_articles': [],
                'classification_metrics': {},
                'conflict_resolution_summary': {},
                'classification_errors': 0,  # ← ADD THIS LINE
                'conflict_resolution_error': None,  # ← ADD THIS LINE
                
                # NEW: Add additional context if provided
                'additional_context': additional_context,
                'context_request': None,
            }
            
            print(f"DEBUG: Created initial state dict with keys: {list(initial_state_dict.keys())}")
            
            # Execute the workflow
            final_state_dict = self.compiled_workflow.invoke(initial_state_dict)
            print(f"DEBUG: Workflow completed, result type: {type(final_state_dict)}")

                # ADD THE DEBUG CODE HERE - RIGHT AFTER workflow.invoke():
            print("\n=== DEBUG: WHERE ARE MY ARTICLES? ===")
            print(f"resolved_articles count: {len(final_state_dict.get('resolved_articles', []))}")
            print(f"classified_articles count: {len(final_state_dict.get('classified_articles', []))}")
            print(f"filtered_search_results count: {len(final_state_dict.get('filtered_search_results', []))}")
            # Check the actual content
            if final_state_dict.get('resolved_articles'):
                print(f"✅ Found resolved_articles: {len(final_state_dict['resolved_articles'])}")
                print(f"First article: {final_state_dict['resolved_articles'][0].get('classified_article', {}).get('article_title', 'NO TITLE')}")
            else:
                print("❌ No resolved_articles found")
            if final_state_dict.get('classified_articles'):
                print(f"✅ Found classified_articles: {len(final_state_dict['classified_articles'])}")
            else:
                print("❌ No classified_articles found")
            print("=== END DEBUG ===\n")
            
            # Extract results with context request handling
            results = self._extract_results_with_context_support(final_state_dict)
            self.logger_service.log_info(f"Adverse media analysis completed for {entity_name}", "ProcessEntity")
            return results
            
        except Exception as e:
            print(f"DEBUG: Exception type: {type(e)}")
            print(f"DEBUG: Exception: {e}")
            import traceback
            traceback.print_exc()
            self.logger_service.log_error(f"Workflow failed for entity {entity_name}: {e}", "ProcessEntity")
            raise

    # NEW METHOD: Enhanced result extraction with context request support
    def _extract_results_with_context_support(self, final_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced result extraction with context request support"""
        print("=== FINAL STATE DICT KEYS ===")
        print(list(final_state_dict.keys()))
        print("=== AGENT OUTPUTS ===")
        agent_outputs = final_state_dict.get('agent_outputs', {})
        print(list(agent_outputs.keys()))
        
        # Determine final status
        status = "UNKNOWN"
        status_value = final_state_dict.get('status', 'unknown')
        status_str = str(status_value)
        
        # Handle context request status
        if status_str == "awaiting_context":
            status = "NEEDS_CONTEXT"
        elif final_state_dict.get('skip_to_final_report', False):
            status = "CLEAN"
        elif final_state_dict.get('human_review_required', False):
            review_reasons = final_state_dict.get('review_reasons', [])
            # Check if it's specifically a context request
            if any('Additional context required' in reason for reason in review_reasons):
                status = "NEEDS_CONTEXT"
            else:
                status = "MANUAL_REVIEW"
        elif status_str == "failed":
            status = "FAILED"
        else:
            status = "COMPLETED"
        
        # Base results
        results = {
            "entity_name": final_state_dict.get('entity_name', ''),
            "status": status,
            "disambiguation_result": final_state_dict.get('disambiguation_result'),
            "search_results_count": len(final_state_dict.get('filtered_search_results', [])),
            "final_assessment": agent_outputs.get('final_assessment'),
            "classifications": final_state_dict.get('resolved_articles', []),
            "conflict_resolutions": agent_outputs.get('conflict_resolutions', []),
            "processing_logs": final_state_dict.get('logs', []),
            "workflow_metadata": {
                "total_steps": len(final_state_dict.get('logs', [])),
                "errors": final_state_dict.get('agent_errors', []),
                "retry_counts": {
                    "disambiguation": final_state_dict.get('disambiguation_retry_count', 0),
                    "search": final_state_dict.get('retry_count', 0)
                }
            }
        }
        
        # Add context request information if needed
        if status == "NEEDS_CONTEXT":
            context_request = final_state_dict.get('context_request', {})
            results["context_request"] = {
                "message": context_request.get('message', 'Additional context required'),
                "suggested_fields": context_request.get('suggested_fields', []),
                "candidates_found": context_request.get('candidates_found', 0),
                "top_candidate_confidence": context_request.get('top_candidate_confidence', 0.0),
                "workflow_id": final_state_dict.get('workflow_id'),  # For resuming
            }
        
        return results

    # NEW METHOD: For resuming with additional context
    def resume_with_context(self, workflow_id: str, additional_context: Dict[str, str], 
                           original_entity: str, original_context: str = "") -> Dict[str, Any]:
        """Resume a workflow that was waiting for additional context"""
        self.logger_service.log_info(f"Resuming workflow {workflow_id} with additional context", "ResumeContext")
        
        try:
            # Process entity with the additional context
            return self.process_entity_with_context_support(
                entity_name=original_entity,
                entity_context=original_context,
                additional_context=additional_context
            )
        except Exception as e:
            self.logger_service.log_error(f"Failed to resume workflow {workflow_id}: {e}", "ResumeContext")
            raise

    def process_multiple_entities(self, entities: List[str], contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        self.logger_service.log_info(f"Processing {len(entities)} entities", "ProcessMultiple")
        if contexts and len(contexts) != len(entities):
            raise ValueError("If contexts provided, must match number of entities")  
        results = []
        for i, entity in enumerate(entities):
            context = contexts[i] if contexts else None
            try:
                result = self.process_entity(entity, context)
                results.append(result)
            except Exception as e:
                self.logger_service.log_error(f"Failed to process entity {entity}: {e}", "ProcessMultiple")
                # Add failed result with error info
                results.append({
                    "entity_name": entity,
                    "status": "FAILED",
                    "error": str(e),
                    "final_assessment": None
                })
        return results 

    def _extract_results_from_dict(self, final_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """FIXED: Extract results directly from dict without AdverseMediaState conversion"""
        print("=== FINAL STATE DICT KEYS ===")
        print(list(final_state_dict.keys()))
        print("=== AGENT OUTPUTS ===")
        agent_outputs = final_state_dict.get('agent_outputs', {})
        print(list(agent_outputs.keys()))
        
        # Determine final status based on workflow execution
        status = "UNKNOWN"
        status_value = final_state_dict.get('status', 'unknown')
        status_str = str(status_value)
        
        # Determine final status
        if final_state_dict.get('skip_to_final_report', False):
            status = "CLEAN"
        elif final_state_dict.get('human_review_required', False):
            status = "MANUAL_REVIEW"
        elif status_str == "failed":
            status = "FAILED"
        else:
            status = "COMPLETED"
        
        # FIXED: Extract resolved articles with proper nested structure handling
        resolved_articles = final_state_dict.get('resolved_articles', [])
        
        # DEBUG: Print actual structure
        print(f"\n=== RESOLVED ARTICLES DEBUG ===")
        print(f"Total resolved articles: {len(resolved_articles)}")
        if resolved_articles:
            print(f"First article keys: {list(resolved_articles[0].keys())}")
            if 'classified_article' in resolved_articles[0]:
                print(f"Classified article keys: {list(resolved_articles[0]['classified_article'].keys())}")
        print("=== END DEBUG ===\n")
        
        # Extract key information from dict
        results = {
            "entity_name": final_state_dict.get('entity_name', ''),
            "status": status,
            "disambiguation_result": final_state_dict.get('disambiguation_result'),
            "search_results_count": len(final_state_dict.get('filtered_search_results', [])),
            "final_assessment": agent_outputs.get('final_assessment'),
            # FIXED: Use resolved_articles directly (they contain the nested structure)
            "classifications": resolved_articles,
            "conflict_resolutions": agent_outputs.get('conflict_resolutions', []),
            "processing_logs": final_state_dict.get('logs', []),
            "workflow_metadata": {
                "total_steps": len(final_state_dict.get('logs', [])),
                "errors": final_state_dict.get('agent_errors', []),
                "retry_counts": {
                    "disambiguation": final_state_dict.get('disambiguation_retry_count', 0),
                    "search": final_state_dict.get('retry_count', 0)
                }
            }
        }
        return results

    def get_workflow_visualization(self) -> str:
        try:
            return str(self.workflow)
        except Exception as e:
            return f"Workflow visualization not available: {e}"

    def test_services(self):
        if self.config.debug_mode:
            self.logger_service.log_info("Testing services...", "TestServices")
            # Test BraveSearch
            test_results = self.services['brave_search'].search("test query", count=2)
            self.logger_service.log_info(f"BraveSearch test returned {len(test_results)} results", "TestServices")  
            # Test embedding service
            similarity = self.services['embedding'].similarity("hello world", "hello earth")
            self.logger_service.log_info(f"Embedding test similarity: {similarity}", "TestServices")
            # Test LLM service
            primary_model = self.services['llm'].get_model('primary')
            self.logger_service.log_info(f"LLM primary model ready: {primary_model}", "TestServices")
            # Test NER service (if available)
            if self.services['ner']:
                test_entities = self.services['ner'].extract_entities("John Smith works at Microsoft in Seattle")
                self.logger_service.log_info(f"NER test extracted {len(test_entities)} entities", "TestServices")
            else:
                self.logger_service.log_info("NER service not available for testing", "TestServices")
            # Test Tavily service (if available)
            if self.services['tavily']:
                try:
                    test_search = self.services['tavily'].search("test news", max_results=2)
                    self.logger_service.log_info(f"Tavily test returned {len(test_search)} results", "TestServices")
                except Exception as e:
                    self.logger_service.log_warning(f"Tavily test search failed: {e}", "TestServices")
            else:
                self.logger_service.log_info("Tavily service not available for testing", "TestServices")

def create_system(config_path: Optional[str] = None) -> AdverseMediaSystem:
    if config_path:
        # TODO: Add custom config loading if needed
        config = BaseAgentConfig()
    else:
        config = BaseAgentConfig()
    return AdverseMediaSystem(config)

# UPDATED: CLI main function with context handling
def main():
    parser = argparse.ArgumentParser(description="Adverse Media Detection System")
    parser.add_argument("entity", help="Entity name to analyze")
    parser.add_argument("--context", help="Additional context about the entity")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Output file for results (JSON)")
    parser.add_argument("--multiple", nargs='+', help="Process multiple entities")
    parser.add_argument("--visualize", action='store_true', help="Show workflow structure")
    parser.add_argument("--test", action='store_true', help="Test services only")
    
    # NEW: Context provision arguments
    parser.add_argument("--additional-context", help="JSON string of additional context fields")
    parser.add_argument("--dob", help="Date of birth (YYYY-MM-DD)")
    parser.add_argument("--occupation", help="Occupation/job title")
    parser.add_argument("--location", help="Location/address")
    parser.add_argument("--organization", help="Associated organization")
    parser.add_argument("--nationality", help="Nationality")
    
    args = parser.parse_args()

    try:
        system = create_system(args.config)
        
        if args.test:
            system.test_services()
            print("Service testing completed. Check logs for results.")
            return 0

        if args.visualize:
            print("Workflow Structure:")
            print(system.get_workflow_visualization())
            return 0

        # NEW: Handle additional context from CLI arguments
        additional_context = None
        if args.additional_context:
            try:
                additional_context = json.loads(args.additional_context)
            except json.JSONDecodeError:
                print("Error: --additional-context must be valid JSON")
                return 1
        elif any([args.dob, args.occupation, args.location, args.organization, args.nationality]):
            additional_context = {}
            if args.dob: additional_context['date_of_birth'] = args.dob
            if args.occupation: additional_context['occupation'] = args.occupation
            if args.location: additional_context['location'] = args.location
            if args.organization: additional_context['organization'] = args.organization
            if args.nationality: additional_context['nationality'] = args.nationality

        # Process entities
        if args.multiple:
            # For multiple entities, use the enhanced method
            results = []
            for entity in args.multiple:
                try:
                    result = system.process_entity_with_context_support(
                        entity, args.context, additional_context
                    )
                    results.append(result)
                except Exception as e:
                    results.append({
                        "entity_name": entity,
                        "status": "FAILED",
                        "error": str(e),
                        "final_assessment": None
                    })
        else:
            # Single entity processing with context support
            results = [system.process_entity_with_context_support(
                args.entity, args.context, additional_context
            )]

        # Output handling
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            # Enhanced output with context request handling
                        # Enhanced output with context request handling
            for result in results:
                entity = result['entity_name']
                status = result['status']
                print(f"\n=== {entity} ===")
                print(f"Status: {status}")
                
                if status == "CLEAN":
                    print("✓ No adverse media found")
                elif status == "COMPLETED":
                    # FIXED: Handle the nested structure properly
                    resolved_articles = result.get('classifications', [])  # This contains resolved_articles
                    
                    # DEBUG: Show what we actually have
                    print(f"DEBUG: Found {len(resolved_articles)} resolved articles")
                    if resolved_articles:
                        print(f"DEBUG: First article structure: {type(resolved_articles[0])}")
                        print(f"DEBUG: First article keys: {list(resolved_articles[0].keys()) if isinstance(resolved_articles[0], dict) else 'Not a dict'}")
                    
                    # FIXED: Count adverse articles with proper nested structure handling
                                        # FIXED: Count adverse articles - check top level first
                    adverse_count = 0
                    for article in resolved_articles:
                        if isinstance(article, dict):
                            # Check top level first (this is where it actually is!)
                            if article.get('is_deemed_adverse', False):
                                adverse_count += 1
                            # Backup: check nested structure
                            else:
                                classified_article = article.get('classified_article', {})
                                if isinstance(classified_article, dict) and classified_article.get('is_deemed_adverse', False):
                                    adverse_count += 1
                    
                    print(f"Found {len(resolved_articles)} articles, {adverse_count} adverse")
                    
                    # OPTIONAL: Show article titles for debugging
                    if resolved_articles and len(resolved_articles) > 0:
                        print("Articles found:")
                        for i, article in enumerate(resolved_articles[:3]):  # Show first 3
                            if isinstance(article, dict):
                                classified_article = article.get('classified_article', {})
                                title = classified_article.get('article_title', 'No title')
                                
                                # FIX: Check the top-level field first, then fallback to nested
                                is_adverse = article.get('is_deemed_adverse', False)
                                if not is_adverse:  # Fallback to nested structure
                                    is_adverse = classified_article.get('is_deemed_adverse', False)
                                    
                                print(f"  {i+1}. {title} {'(adverse)' if is_adverse else '(clean)'}")
                        if len(resolved_articles) > 3:
                            print(f"  ... and {len(resolved_articles) - 3} more")
                            
                elif status == "NEEDS_CONTEXT":
                    print("📝 Additional context required")
                    context_req = result.get('context_request', {})
                    print(f"Message: {context_req.get('message', 'N/A')}")
                    suggested = context_req.get('suggested_fields', [])
                    if suggested:
                        print(f"Suggested fields: {', '.join(suggested)}")
                    candidates = context_req.get('candidates_found', 0)
                    confidence = context_req.get('top_candidate_confidence', 0)
                    print(f"Found {candidates} potential matches (top confidence: {confidence:.2f})")
                    
                    # Show CLI example for providing context
                    print("\nTo provide additional context, run:")
                    print(f"python main.py '{entity}' --dob YYYY-MM-DD --occupation 'Job Title' --location 'City, Country'")
                    
                elif status == "MANUAL_REVIEW":
                    print("⚠ Requires manual review")
                elif status == "FAILED":
                    print(f"✗ Processing failed: {result.get('error', 'Unknown error')}")
                
                if result.get('final_assessment'):
                    assessment = result['final_assessment']
                    if 'assessment_summary' in assessment:
                        print(f"Summary: {assessment['assessment_summary']}")

        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())

# USAGE EXAMPLES:
"""
# Basic usage
python main.py "John Smith"

# With initial context
python main.py "John Smith" --context "Software engineer at Google"

# With additional context fields
python main.py "John Smith" --dob "1985-03-15" --occupation "Software Engineer" --location "San Francisco, CA"

# With JSON context
python main.py "John Smith" --additional-context '{"date_of_birth": "1985-03-15", "occupation": "Engineer"}'

# Multiple entities with context
python main.py --multiple "John Smith" "Jane Doe" --occupation "Engineer"
"""