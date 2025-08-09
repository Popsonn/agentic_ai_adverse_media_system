from typing import Callable, Dict, Any
import logging
from core.state import AdverseMediaState, AgentStatus, ErrorSeverity
from agents.conflict_resolution.agent import HybridConflictResolutionAgent
from config.settings import ConflictResolutionConfig
from services.llm_service import LLMService
from services.brave_search_service import BraveSearchService
from core.exceptions import AgentExecutionError


def create_conflict_resolution_node(
    config: ConflictResolutionConfig,
    logger: logging.Logger,
    llm_service: LLMService,
    brave_search_service: BraveSearchService = None
) -> Callable[[AdverseMediaState], AdverseMediaState]:
    """
    Factory function to create the conflict resolution node for LangGraph.
    This encapsulates your agent initialization and uses unified state updates.
    """

    conflict_agent = HybridConflictResolutionAgent(
        config=config,
        logger=logger,
        llm_service=llm_service,
        brave_search_service=brave_search_service
    )

    def conflict_resolution_node(state: AdverseMediaState) -> AdverseMediaState:
        agent_name = "conflict_resolution_agent"
        state.start_agent(agent_name)
        logger.info(
        f"Starting conflict resolution for {len(state.classified_articles) if state.classified_articles else 0} articles"
    )
        try:
            if not getattr(state, 'classified_articles', None):
                logger.warning("No classified articles found in state")
                empty_output = {
                    "resolved_articles": [],
                    "summary": {
                        "total_articles": 0,
                        "resolved_articles": 0,
                        "human_review_required": 0,
                        "errors": 0,
                        "resolution_methods": {}
                    }
                }
                state.complete_agent(agent_name, AgentStatus.COMPLETED, output_data=empty_output)
                # Also set dedicated fields for backwards compatibility
                state.resolved_articles = []
                state.conflict_resolution_summary = empty_output["summary"]
                return state

            resolved_articles = []
            summary_stats = {
                "total_articles": len(state.classified_articles),
                "resolved_articles": 0,
                "human_review_required": 0,
                "errors": 0,
                "resolution_methods": {}
            }

            for article in state.classified_articles:
                try:
                    resolved_article = conflict_agent.process_article_for_conflict(article)
                    resolved_articles.append(resolved_article)

                    method = resolved_article.resolution_method.value
                    summary_stats["resolution_methods"][method] = summary_stats["resolution_methods"].get(method, 0) + 1

                    if method == "HUMAN_REVIEW_REQUIRED":
                        summary_stats["human_review_required"] += 1
                    else:
                        summary_stats["resolved_articles"] += 1
                except Exception as exc:
                    logger.error(f"Error processing article '{getattr(article, 'article_title', 'unknown')}': {exc}")
                    summary_stats["errors"] += 1

                    failed_resolution = conflict_agent._create_resolved_article(
                        classified_article=article,
                        resolution_method=conflict_agent.cr_config.resolution_method_enum.HUMAN_REVIEW_REQUIRED,
                        resolution_details={
                            "error": str(exc),
                            "reason": "Processing error occurred"
                        }
                    )
                    resolved_articles.append(failed_resolution)
                    summary_stats["human_review_required"] += 1

            # After the for loop in conflict_resolution_node:

            if summary_stats["human_review_required"] > 0:
                state.set_review_required(
                    f"{summary_stats['human_review_required']} articles require human review", 
                    agent_name
                )

            # Mark agent complete with outputs centralized under agent_results
            state.complete_agent(
                agent_name,
                AgentStatus.COMPLETED,
                output_data={
                    "resolved_articles": resolved_articles,
                    "summary": summary_stats
                }
            )
            # Also update dedicated fields
            state.resolved_articles = resolved_articles
            state.conflict_resolution_summary = summary_stats

            logger.info(f"Conflict resolution completed: {summary_stats['resolved_articles']} resolved, "
                        f"{summary_stats['human_review_required']} require human review, {summary_stats['errors']} errors")

            return state

        except Exception as e:
            logger.error(f"Critical error in conflict resolution node: {e}")
            wrapped_error = AgentExecutionError(
                str(e),
                agent_name=agent_name,
                severity=ErrorSeverity.CRITICAL
            )
            state.fail_agent(agent_name, wrapped_error)
            return state

    return conflict_resolution_node

def create_conflict_resolution_wrapper(conflict_resolution_node_func, logger):
    """NEW wrapper - converts dict to AdverseMediaState for conflict resolution"""
    def conflict_resolution_wrapper(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print(f"DEBUG: Conflict resolution wrapper input keys: {list(state_dict.keys())}")
            
            # Convert dict to AdverseMediaState object
            from core.state import AdverseMediaState, AgentResult, AgentStatus, AgentLogEntry, AgentErrorEntry
            from models.entity import EntityCandidate, DisambiguationResult
            from models.search import SearchResult
            from models.results import ClassifiedArticleResult, ResolvedArticleResult
            from core.exceptions import BaseAdverseMediaException, ErrorSeverity
            from datetime import datetime
            
            state_obj = AdverseMediaState(
                entity_name=state_dict.get('entity_name', ''),
                user_input=state_dict.get('user_input', ''),
                workflow_id=state_dict.get('workflow_id', str(__import__('uuid').uuid4()))
            )
            
            # Copy relevant fields from dict to object
            for key, value in state_dict.items():
                if hasattr(state_obj, key) and key not in ['created_at', 'updated_at', 'processing_start_time', 'processing_end_time', 'agent_results', 'agent_logs', 'agent_errors']:
                    try:
                        # Handle special object fields that need reconstruction
                        if key == 'resolved_entity' and value and isinstance(value, dict):
                            if hasattr(EntityCandidate, 'from_dict'):
                                state_obj.resolved_entity = EntityCandidate.from_dict(value)
                        elif key == 'all_candidates' and value:
                            if hasattr(EntityCandidate, 'from_dict'):
                                state_obj.all_candidates = [EntityCandidate.from_dict(c) if isinstance(c, dict) else c for c in value]
                        elif key == 'disambiguation_result' and value and isinstance(value, dict):
                            if hasattr(DisambiguationResult, 'from_dict'):
                                state_obj.disambiguation_result = DisambiguationResult.from_dict(value)
                        elif key == 'raw_search_results' and value:
                            if hasattr(SearchResult, 'from_dict'):
                                state_obj.raw_search_results = [SearchResult.from_dict(r) if isinstance(r, dict) else r for r in value]
                        elif key == 'filtered_search_results' and value:
                            if hasattr(SearchResult, 'from_dict'):
                                state_obj.filtered_search_results = [SearchResult.from_dict(r) if isinstance(r, dict) else r for r in value]
                        elif key == 'classified_articles' and value:
                            if hasattr(ClassifiedArticleResult, 'from_dict'):
                                state_obj.classified_articles = [ClassifiedArticleResult.from_dict(a) if isinstance(a, dict) else a for a in value]
                        elif key == 'resolved_articles' and value:
                            if hasattr(ResolvedArticleResult, 'from_dict'):
                                state_obj.resolved_articles = [ResolvedArticleResult.from_dict(a) if isinstance(a, dict) else a for a in value]
                        elif key == 'completed_agents' and isinstance(value, list):  # ADD THIS LINE
                            setattr(state_obj, key, set(value)) 
                        else:
                            setattr(state_obj, key, value)
                    except Exception as field_error:
                        print(f"WARNING: Could not set field {key}: {field_error}")
                        # Continue without this field
            
            # Reconstruct agent_results
            agent_results_dict = state_dict.get('agent_results', {})
            if agent_results_dict:
                state_obj.agent_results = {}
                for agent_name, result_data in agent_results_dict.items():
                    if isinstance(result_data, dict):
                        try:
                            start_time = None
                            if result_data.get('start_time'):
                                if isinstance(result_data['start_time'], str):
                                    start_time = datetime.fromisoformat(result_data['start_time'])
                                else:
                                    start_time = result_data['start_time']
                            
                            end_time = None
                            if result_data.get('end_time'):
                                if isinstance(result_data['end_time'], str):
                                    end_time = datetime.fromisoformat(result_data['end_time'])
                                else:
                                    end_time = result_data['end_time']
                            
                            agent_result = AgentResult(
                                agent_name=result_data.get('agent_name', agent_name),
                                status=AgentStatus(result_data.get('status', 'pending')),
                                start_time=start_time,
                                end_time=end_time,
                                error=result_data.get('error'),
                                metrics=result_data.get('metrics'),
                                output_data=result_data.get('output_data'),
                                retry_count=result_data.get('retry_count', 0)
                            )
                            state_obj.agent_results[agent_name] = agent_result
                        except Exception as e:
                            print(f"WARNING: Could not reconstruct AgentResult for {agent_name}: {e}")
                            state_obj.agent_results[agent_name] = AgentResult(
                                agent_name=agent_name,
                                status=AgentStatus.PENDING
                            )
                    else:
                        state_obj.agent_results[agent_name] = result_data
            
            # Reconstruct agent_logs
            agent_logs_list = state_dict.get('agent_logs', [])
            if agent_logs_list:
                state_obj.agent_logs = []
                for log_data in agent_logs_list:
                    if isinstance(log_data, dict):
                        try:
                            timestamp = log_data.get('timestamp')
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp)
                            elif timestamp is None:
                                timestamp = datetime.now()
                            
                            log_entry = AgentLogEntry(
                                agent_name=log_data.get('agent_name', ''),
                                message=log_data.get('message', ''),
                                timestamp=timestamp,
                                level=log_data.get('level', 'INFO'),
                                context=log_data.get('context', {})
                            )
                            state_obj.agent_logs.append(log_entry)
                        except Exception as e:
                            print(f"WARNING: Could not reconstruct AgentLogEntry: {e}")
                    else:
                        state_obj.agent_logs.append(log_data)

            # Reconstruct agent_errors
            agent_errors_list = state_dict.get('agent_errors', [])
            if agent_errors_list:
                state_obj.agent_errors = []
                for error_data in agent_errors_list:
                    if isinstance(error_data, dict):
                        try:
                            timestamp = error_data.get('timestamp')
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp)
                            elif timestamp is None:
                                timestamp = datetime.now()
                            
                            error_msg = error_data.get('message', 'Unknown error')
                            severity_str = error_data.get('severity', 'UNKNOWN')
                            try:
                                severity = ErrorSeverity(severity_str) if hasattr(ErrorSeverity, severity_str) else None
                            except:
                                severity = None
                            
                            base_error = BaseAdverseMediaException(error_msg, severity)
                            
                            error_entry = AgentErrorEntry(
                                agent_name=error_data.get('agent_name', ''),
                                error=base_error,
                                timestamp=timestamp,
                                item_id=error_data.get('item_id')
                            )
                            state_obj.agent_errors.append(error_entry)
                        except Exception as e:
                            print(f"WARNING: Could not reconstruct AgentErrorEntry: {e}")
                    else:
                        state_obj.agent_errors.append(error_data)
            
            # Set proper datetime fields
            state_obj.created_at = datetime.now()
            state_obj.updated_at = datetime.now()
            
            print("DEBUG: Running conflict resolution agent...")
            
            # Call the actual node function (which expects AdverseMediaState)
            result_obj = conflict_resolution_node_func(state_obj)
            
            print("DEBUG: Conflict resolution completed, converting back to dict...")
            
            # Convert result back to dict
            if isinstance(result_obj, dict):
                result_dict = result_obj
            else:
                result_dict = result_obj.to_dict()
            
            print("DEBUG: Successfully converted conflict resolution results back to dict")
            return result_dict
            
        except Exception as e:
            print(f"DEBUG: Exception in conflict resolution wrapper: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"Conflict resolution wrapper error: {e}"
            if hasattr(logger, 'log_error'):
                logger.log_error(error_msg, "workflow")
            elif hasattr(logger, 'error'):
                logger.error(f"[workflow] {error_msg}")
            else:
                print(f"ERROR: {error_msg}")
            
            # Return error state as dict
            error_dict = state_dict.copy()
            error_dict.update({
                'status': 'failed',
                'updated_at': datetime.now().isoformat(),
                'logs': error_dict.get('logs', []) + [f"[workflow] ERROR: {error_msg}"],
                'agent_errors': error_dict.get('agent_errors', []) + [{
                    'agent_name': 'conflict_resolution_agent',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'CRITICAL'
                }]
            })
            return error_dict
    
    return conflict_resolution_wrapper
