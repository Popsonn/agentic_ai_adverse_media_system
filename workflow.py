from langgraph.graph import StateGraph, END
from typing import Dict, Any, List
from agents.classification.agent_wrapper import create_classification_wrapper
from agents.search_strategy.agent import SearchStrategyAgent
from agents.entity_disambiguation.agent import EntityDisambiguationAgent
from agents.classification.agent import ClassificationAgent
from agents.conflict_resolution.agent import HybridConflictResolutionAgent
from agents.conflict_resolution.agent_wrapper import create_conflict_resolution_node, create_conflict_resolution_wrapper
from core.exceptions import ErrorSeverity 
from services.brave_search_service import BraveSearchService
from datetime import datetime

# IMPORT THE SPECIFIC CONFIG CLASSES
from config.settings import (
    EntityDisambiguationAgentConfig, 
    SearchStrategyConfig, 
    ClassificationAgentConfig,
    ConflictResolutionConfig
)

def should_continue_after_disambiguation(state: Dict[str, Any]) -> str:
    """Routing function for disambiguation results - UPDATED to handle needs_context"""
    
    disambiguation_result = state.get('disambiguation_result')
    
    if not disambiguation_result:
        return "failed"

    # disambiguation_result is a dict (from DisambiguationResult.to_dict())
    status = disambiguation_result.get('status')
    confidence_score = disambiguation_result.get('confidence_score', 0.0)

    if status == "resolved":
        if confidence_score >= 0.6:
            return "search_strategy"
        else:
            return "manual_review"

    elif status == "needs_review":
        return "manual_review"

    elif status == "needs_context":  # NEW: Handle context requests
        return "context_required"

    elif status == "no_matches":
        retry_count = state.get('disambiguation_retry_count', 0)
        if retry_count < 2:
            state['disambiguation_retry_count'] = retry_count + 1
            return "retry_disambiguation"
        else:
            return "failed"

    elif status == "error":
        errors = state.get('agent_errors', [])
        if errors:
            # Get the latest error
            latest_error = errors[-1]
            error_severity = latest_error.get('severity', 'UNKNOWN')
            
            #if error_severity == 'CRITICAL':
            if error_severity == ErrorSeverity.CRITICAL.value:
                return "failed"
            else:
                retry_count = state.get('disambiguation_retry_count', 0)
                if retry_count < 1:
                    state['disambiguation_retry_count'] = retry_count + 1
                    return "retry_disambiguation"
                else:
                    return "manual_review"
        return "failed"

    return "failed"

def context_required_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """UPDATED: Handle cases where additional context is needed - with waiting state"""
    
    # Add log entry
    logs = state.get('logs', [])
    logs.append("[workflow] Additional context required for entity identification")
    state['logs'] = logs
    
    # Set waiting for context status
    state['status'] = 'awaiting_context'  # NEW: Specific status for waiting
    state['updated_at'] = datetime.now().isoformat()
    
    # Get the context request reason from disambiguation result
    disambiguation_result = state.get('disambiguation_result', {})
    context_reason = disambiguation_result.get('review_reason', 'Additional context required')
    
    # Add to review reasons
    review_reasons = state.get('review_reasons', [])
    review_reasons.append(f"[entity_disambiguation] {context_reason}")
    state['review_reasons'] = review_reasons
    
    # Mark as requiring human input (context provision)
    state['human_review_required'] = True
    
    # Add context-specific metadata for UI/API consumers
    state['context_request'] = {
        'type': 'additional_context',
        'message': context_reason,
        'suggested_fields': [
            'date_of_birth', 
            'address', 
            'occupation', 
            'nationality',
            'organization',
            'additional_identifiers'
        ],
        'entity_name': state.get('entity_name', ''),
        'candidates_found': len(state.get('all_candidates', [])),
        'top_candidate_confidence': disambiguation_result.get('confidence_score', 0.0),
        'awaiting_retry': True  # NEW: Flag indicating this can be retried
    }
    
    return state

def retry_with_context_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """NEW: Handle context provision and prepare for retry"""
    
    # Validate that additional context was provided
    additional_context = state.get('additional_context')
    if not additional_context:
        # No context provided - return to context_required
        logs = state.get('logs', [])
        logs.append("[workflow] No additional context provided - context still required")
        state['logs'] = logs
        return state
    
    # Log context received
    logs = state.get('logs', [])
    context_fields = list(additional_context.keys()) if isinstance(additional_context, dict) else ['provided']
    logs.append(f"[workflow] Additional context received: {', '.join(context_fields)}")
    state['logs'] = logs
    
    # Enhance user_input with additional context
    original_input = state.get('user_input', '')
    
    # Format additional context nicely
    if isinstance(additional_context, dict):
        context_parts = []
        for key, value in additional_context.items():
            if value:  # Only include non-empty values
                formatted_key = key.replace('_', ' ').title()
                context_parts.append(f"{formatted_key}: {value}")
        
        context_text = ". ".join(context_parts)
        enhanced_input = f"{original_input}. Additional context: {context_text}"
    else:
        enhanced_input = f"{original_input}. Additional context: {additional_context}"
    
    # Update user input with enhanced context
    state['user_input'] = enhanced_input
    
    # Clear previous disambiguation results to force fresh analysis
    state['resolved_entity'] = None
    state['all_candidates'] = []
    state['disambiguation_confidence'] = 0.0
    state['disambiguation_result'] = None
    
    # Clear context request state
    state['context_request'] = None
    state['human_review_required'] = False
    
    # Reset retry counters to give fresh attempts
    state['disambiguation_retry_count'] = 0
    
    # Set status back to in progress
    state['status'] = 'in_progress'
    state['updated_at'] = datetime.now().isoformat()
    
    # Log that we're retrying with context
    logs.append("[workflow] Retrying disambiguation with additional context")
    state['logs'] = logs
    
    return state

def should_retry_with_context(state: Dict[str, Any]) -> str:
    """NEW: Routing function to determine if context was provided for retry"""
    
    # Check if additional context was provided
    additional_context = state.get('additional_context')
    
    if additional_context:
        return "retry_disambiguation"
    else:
        return "still_waiting"
    
def should_continue_after_search(state: Dict[str, Any]) -> str:
    """FIXED: Routing function for search results that handles dict input"""
    
    # Check if search agent detected a clean entity
    if state.get("skip_to_final_report", False):
        return "clean_entity"
    
    # Legacy support: Also check the old way
    search_quality_metrics = state.get('search_quality_metrics', {})
    if search_quality_metrics.get('clean_entity_likely'):
        return "clean_entity"

    # Check for search failures and retry logic
    agent_results = state.get('agent_results', {})
    search_result = agent_results.get('search_agent', {})
    
    if search_result.get('status') == 'failed':
        retry_count = state.get('retry_count', 0)
        if retry_count < 1:
            state['retry_count'] = retry_count + 1
            return "retry"

    # Default: continue to classification
    return "continue"

def clean_entity_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """FIXED: Handle clean entities - works with dict state"""
    
    # Add log entry to logs list
    logs = state.get('logs', [])
    logs.append("[workflow] Entity determined to be clean - finalizing assessment")
    state['logs'] = logs
    
    # Set final processing status
    state['status'] = 'completed'
    state['processing_end_time'] = datetime.now().isoformat()
    state['updated_at'] = datetime.now().isoformat()
    
    # Add clean entity assessment to agent outputs
    search_quality_metrics = state.get('search_quality_metrics', {})
    filtered_search_results = state.get('filtered_search_results', [])
    
    clean_assessment = {
        "final_risk_level": "CLEAN",
        "assessment_summary": search_quality_metrics.get(
            'final_assessment', 
            "No adverse media found - entity appears clean"
        ),
        "articles_found": len(filtered_search_results),
        "search_quality": search_quality_metrics.get('overall_quality', 0.0),
        "reason": search_quality_metrics.get('skipped_classification_reason', 'No adverse content detected')
    }
    
    agent_outputs = state.get('agent_outputs', {})
    agent_outputs["final_assessment"] = clean_assessment
    state['agent_outputs'] = agent_outputs
    
    return state

def manual_review_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """FIXED: Handle manual review cases - works with dict state"""
    
    # Add log entry
    logs = state.get('logs', [])
    logs.append("[workflow] Routing to manual review")
    state['logs'] = logs
    
    # Set review required
    state['human_review_required'] = True
    
    review_reasons = state.get('review_reasons', [])
    review_reasons.append("[workflow] Manual review required")
    state['review_reasons'] = review_reasons
    
    state['status'] = 'needs_review'
    state['updated_at'] = datetime.now().isoformat()
    
    return state

def failed_processing_handler(state: Dict[str, Any]) -> Dict[str, Any]:
    """FIXED: Handle failed processing cases - works with dict state"""
    
    # Add log entry
    logs = state.get('logs', [])
    logs.append("[workflow] Processing failed")
    state['logs'] = logs
    
    # Set failed status
    state['status'] = 'failed'
    state['processing_end_time'] = datetime.now().isoformat()
    state['updated_at'] = datetime.now().isoformat()
    
    return state

def create_entity_disambiguation_wrapper(entity_agent, logger):
    """FIXED wrapper - properly handles dict to AdverseMediaState conversion"""
    def disambiguation_wrapper(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print(f"DEBUG: Disambiguation wrapper input keys: {list(state_dict.keys())}")
            
            # Convert dict to AdverseMediaState object
            from core.state import AdverseMediaState, AgentResult, AgentStatus
            from models.entity import EntityCandidate, DisambiguationResult
            from datetime import datetime
            
            state_obj = AdverseMediaState(
                entity_name=state_dict.get('entity_name', ''),
                user_input=state_dict.get('user_input', ''),
                workflow_id=state_dict.get('workflow_id', str(__import__('uuid').uuid4()))
            )
            
            # Copy relevant fields from dict to object
            for key, value in state_dict.items():
                if hasattr(state_obj, key) and key not in ['created_at', 'updated_at', 'processing_start_time', 'processing_end_time', 'agent_results']:
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
                        elif key == 'completed_agents' and isinstance(value, list):  # ADD THIS
                            setattr(state_obj, key, set(value))  # Convert list back to set
                        else:
                            setattr(state_obj, key, value)
                    except Exception as field_error:
                        print(f"WARNING: Could not set field {key}: {field_error}")
                        # Continue without this field
            
            # FIX: Properly reconstruct agent_results as AgentResult objects
            agent_results_dict = state_dict.get('agent_results', {})
            if agent_results_dict:
                state_obj.agent_results = {}
                for agent_name, result_data in agent_results_dict.items():
                    if isinstance(result_data, dict):
                        # Convert dict back to AgentResult object with FIXED datetime handling
                        try:
                            # FIX: Convert string datetimes back to datetime objects
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
                            # Create a minimal AgentResult
                            state_obj.agent_results[agent_name] = AgentResult(
                                agent_name=agent_name,
                                status=AgentStatus.PENDING
                            )
                    else:
                        # Already an AgentResult object
                        state_obj.agent_results[agent_name] = result_data
            
            # FIX: Properly reconstruct agent_logs as AgentLogEntry objects
            agent_logs_list = state_dict.get('agent_logs', [])
            if agent_logs_list:
                from core.state import AgentLogEntry
                state_obj.agent_logs = []
                for log_data in agent_logs_list:
                    if isinstance(log_data, dict):
                        try:
                            # Convert string timestamp back to datetime
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
                            # Skip this log entry
                    else:
                        # Already an AgentLogEntry object
                        state_obj.agent_logs.append(log_data)
            
            # FIX: Properly reconstruct agent_errors as AgentErrorEntry objects  
            agent_errors_list = state_dict.get('agent_errors', [])
            if agent_errors_list:
                from core.state import AgentErrorEntry
                from core.exceptions import BaseAdverseMediaException, ErrorSeverity
                state_obj.agent_errors = []
                for error_data in agent_errors_list:
                    if isinstance(error_data, dict):
                        try:
                            # Convert string timestamp back to datetime
                            timestamp = error_data.get('timestamp')
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp)
                            elif timestamp is None:
                                timestamp = datetime.now()
                            
                            # Create a basic exception object from dict data
                            # This is a simplified reconstruction - you might need to adjust based on your exception structure
                            error_msg = error_data.get('message', 'Unknown error')
                            severity_str = error_data.get('severity', 'UNKNOWN')
                            try:
                                severity = ErrorSeverity(severity_str) if hasattr(ErrorSeverity, severity_str) else ErrorSeverity.UNKNOWN
                            except:
                                severity = ErrorSeverity.UNKNOWN if hasattr(ErrorSeverity, 'UNKNOWN') else None
                            
                            # Create a basic exception - adjust this based on your exception structure
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
                            # Skip this error entry
                    else:
                        # Already an AgentErrorEntry object
                        state_obj.agent_errors.append(error_data)
            
            # Set proper datetime fields
            state_obj.created_at = datetime.now()
            state_obj.updated_at = datetime.now()
            
            print("DEBUG: Running entity disambiguation agent...")
            
            # Call the actual agent method
            result_obj = entity_agent._run_implementation(state_obj)
            
            print("DEBUG: Agent completed, converting back to dict...")
            
            # FIX: Check if result is already a dict or needs conversion
            if isinstance(result_obj, dict):
                result_dict = result_obj
            else:
                result_dict = result_obj.to_dict()
            
            print("DEBUG: Successfully converted back to dict")
            return result_dict
            
        except Exception as e:
            print(f"DEBUG: Exception in disambiguation wrapper: {e}")
            import traceback
            traceback.print_exc()
            
            # Log error
            error_msg = f"Entity disambiguation wrapper error: {e}"
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
                    'agent_name': 'disambiguation_agent',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'CRITICAL'
                }]
            })
            return error_dict
    
    return disambiguation_wrapper

def create_search_agent_wrapper(search_agent, logger):
    """FIXED wrapper - properly handles dict to AdverseMediaState conversion"""
    def search_wrapper(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print(f"DEBUG: Search wrapper input keys: {list(state_dict.keys())}")
            
            # Convert dict to AdverseMediaState object
            from core.state import AdverseMediaState, AgentResult, AgentStatus
            from models.entity import EntityCandidate, DisambiguationResult
            from models.search import SearchResult
            from datetime import datetime
            
            state_obj = AdverseMediaState(
                entity_name=state_dict.get('entity_name', ''),
                user_input=state_dict.get('user_input', ''),
                workflow_id=state_dict.get('workflow_id', str(__import__('uuid').uuid4()))
            )
            
            # Copy relevant fields from dict to object
            for key, value in state_dict.items():
                if hasattr(state_obj, key) and key not in ['created_at', 'updated_at', 'processing_start_time', 'processing_end_time', 'agent_results']:
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
                        elif key == 'completed_agents' and isinstance(value, list):  # ADD THIS LINE
                            setattr(state_obj, key, set(value))  # Convert list back to set
                        else:
                            setattr(state_obj, key, value)
                    except Exception as field_error:
                        print(f"WARNING: Could not set field {key}: {field_error}")
                        # Continue without this field
            
            # FIX: Properly reconstruct agent_results as AgentResult objects
            agent_results_dict = state_dict.get('agent_results', {})
            if agent_results_dict:
                state_obj.agent_results = {}
                for agent_name, result_data in agent_results_dict.items():
                    if isinstance(result_data, dict):
                        # Convert dict back to AgentResult object with FIXED datetime handling
                        try:
                            # FIX: Convert string datetimes back to datetime objects
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
                            # Create a minimal AgentResult
                            state_obj.agent_results[agent_name] = AgentResult(
                                agent_name=agent_name,
                                status=AgentStatus.PENDING
                            )
                    else:
                        # Already an AgentResult object
                        state_obj.agent_results[agent_name] = result_data
            
            # FIX: Properly reconstruct agent_logs as AgentLogEntry objects
            agent_logs_list = state_dict.get('agent_logs', [])
            if agent_logs_list:
                from core.state import AgentLogEntry
                state_obj.agent_logs = []
                for log_data in agent_logs_list:
                    if isinstance(log_data, dict):
                        try:
                            # Convert string timestamp back to datetime
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
                            # Skip this log entry
                    else:
                        # Already an AgentLogEntry object
                        state_obj.agent_logs.append(log_data)
            
            # FIX: Properly reconstruct agent_errors as AgentErrorEntry objects  
            agent_errors_list = state_dict.get('agent_errors', [])
            if agent_errors_list:
                from core.state import AgentErrorEntry
                from core.exceptions import BaseAdverseMediaException, ErrorSeverity
                state_obj.agent_errors = []
                for error_data in agent_errors_list:
                    if isinstance(error_data, dict):
                        try:
                            # Convert string timestamp back to datetime
                            timestamp = error_data.get('timestamp')
                            if isinstance(timestamp, str):
                                timestamp = datetime.fromisoformat(timestamp)
                            elif timestamp is None:
                                timestamp = datetime.now()
                            
                            # Create a basic exception object from dict data
                            # This is a simplified reconstruction - you might need to adjust based on your exception structure
                            error_msg = error_data.get('message', 'Unknown error')
                            severity_str = error_data.get('severity', 'UNKNOWN')
                            try:
                                severity = ErrorSeverity(severity_str) if hasattr(ErrorSeverity, severity_str) else ErrorSeverity.UNKNOWN
                            except:
                                severity = ErrorSeverity.UNKNOWN if hasattr(ErrorSeverity, 'UNKNOWN') else None
                            
                            # Create a basic exception - adjust this based on your exception structure
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
                            # Skip this error entry
                    else:
                        # Already an AgentErrorEntry object
                        state_obj.agent_errors.append(error_data)
            
            # Set proper datetime fields
            state_obj.created_at = datetime.now()
            state_obj.updated_at = datetime.now()
            
            print("DEBUG: Running search agent...")
            
            # Call the actual agent method
            result_obj = search_agent._run_implementation(state_obj)
            
            print("DEBUG: Search agent completed, converting back to dict...")
            
            # FIX: Check if result is already a dict or needs conversion
            if isinstance(result_obj, dict):
                result_dict = result_obj
            else:
                result_dict = result_obj.to_dict()
            
            print("DEBUG: Successfully converted search results back to dict")
            return result_dict
            
        except Exception as e:
            print(f"DEBUG: Exception in search wrapper: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"Search agent wrapper error: {e}"
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
                    'agent_name': 'search_agent',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'CRITICAL'
                }]
            })
            return error_dict
    
    return search_wrapper
def create_workflow(config, logger, llm_service, tavily_service, brave_search_service: BraveSearchService):
    """Complete workflow setup with routing - FIXED for dict-based state"""

    workflow = StateGraph(dict)  # Using dict state throughout

    # Create EntityDisambiguationAgentConfig
    entity_config = EntityDisambiguationAgentConfig(
        # Copy base config attributes
        tavily_api_key=config.tavily_api_key,
        brave_api_key=config.brave_api_key,
        groq_api_key=config.groq_api_key,
        openai_api_key=config.openai_api_key,
        primary_llm_model=config.primary_llm_model,
        secondary_llm_model=config.secondary_llm_model,
        arbitration_llm_model=config.arbitration_llm_model,
        embedding_model_name=config.embedding_model_name,
        ner_model=config.ner_model,
        batch_size=config.batch_size,
        data_storage_path=config.data_storage_path,
        confidence_threshold=config.confidence_threshold,
        max_retries=config.max_retries,
        timeout=config.timeout,
        logging_level=config.logging_level,
        enable_metrics=config.enable_metrics,
        debug_mode=config.debug_mode,
        human_review_confidence_threshold=config.human_review_confidence_threshold,
        
        # EntityDisambiguationAgentConfig specific attributes
        review_threshold=0.85,
        max_candidates=5,
        max_context_terms=3
    )
    
    # Create SearchStrategyConfig
    search_config = SearchStrategyConfig(
        # Copy base config attributes
        tavily_api_key=config.tavily_api_key,
        brave_api_key=config.brave_api_key,
        groq_api_key=config.groq_api_key,
        openai_api_key=config.openai_api_key,
        primary_llm_model=config.primary_llm_model,
        secondary_llm_model=config.secondary_llm_model,
        arbitration_llm_model=config.arbitration_llm_model,
        embedding_model_name=config.embedding_model_name,
        ner_model=config.ner_model,
        batch_size=config.batch_size,
        data_storage_path=config.data_storage_path,
        confidence_threshold=config.confidence_threshold,
        max_retries=config.max_retries,
        timeout=config.timeout,
        logging_level=config.logging_level,
        enable_metrics=config.enable_metrics,
        debug_mode=config.debug_mode,
        human_review_confidence_threshold=config.human_review_confidence_threshold,
        
        # SearchStrategyConfig specific attributes
        similarity_threshold=0.85,
        max_search_attempts=3,
        sentence_transformer_model="all-MiniLM-L6-v2",
        search_quality_threshold=0.5,
        max_articles=10,
        max_filtered_articles=5,
        days_back=730
    )

    # Create ClassificationAgentConfig
    classification_config = ClassificationAgentConfig(
        # Copy base config attributes
        tavily_api_key=config.tavily_api_key,
        brave_api_key=config.brave_api_key,
        groq_api_key=config.groq_api_key,
        openai_api_key=config.openai_api_key,
        primary_llm_model=config.primary_llm_model,
        secondary_llm_model=config.secondary_llm_model,
        arbitration_llm_model=config.arbitration_llm_model,
        embedding_model_name=config.embedding_model_name,
        ner_model=config.ner_model,
        batch_size=config.batch_size,
        data_storage_path=config.data_storage_path,
        confidence_threshold=config.confidence_threshold,
        max_retries=config.max_retries,
        timeout=config.timeout,
        logging_level=config.logging_level,
        enable_metrics=config.enable_metrics,
        debug_mode=config.debug_mode,
        human_review_confidence_threshold=config.human_review_confidence_threshold,
        
        # ClassificationAgentConfig specific attributes
        max_classification_errors=3,
        skip_articles_without_content=True,
        require_secondary_llm=True,
        primary_llm_role="primary",
        secondary_llm_role="secondary"
    )

    # Create ConflictResolutionConfig
    conflict_config = ConflictResolutionConfig(
        # Copy base config attributes
        tavily_api_key=config.tavily_api_key,
        brave_api_key=config.brave_api_key,
        groq_api_key=config.groq_api_key,
        openai_api_key=config.openai_api_key,
        primary_llm_model=config.primary_llm_model,
        secondary_llm_model=config.secondary_llm_model,
        arbitration_llm_model=config.arbitration_llm_model,
        embedding_model_name=config.embedding_model_name,
        ner_model=config.ner_model,
        batch_size=config.batch_size,
        data_storage_path=config.data_storage_path,
        confidence_threshold=config.confidence_threshold,
        max_retries=config.max_retries,
        timeout=config.timeout,
        logging_level=config.logging_level,
        enable_metrics=config.enable_metrics,
        debug_mode=config.debug_mode,
        human_review_confidence_threshold=config.human_review_confidence_threshold,
        
        # ConflictResolutionConfig specific attributes
        high_confidence_threshold=0.85,
        confidence_gap_for_conservative=0.2,
        llm_arbitration_min_confidence=0.75,
        minimal_absolute_confidence_for_rule=0.6,
        escalate_critical=True,
        enable_reasoning=True,
        enable_external_search=True,
        brave_search_api_key=config.brave_api_key
    )

    # Initialize agents with the correct config types
    entity_agent = EntityDisambiguationAgent(
        config=entity_config,
        logger=logger, 
        llm_service=llm_service, 
        tavily_service=tavily_service
    )
    
    search_agent = SearchStrategyAgent(
        config=search_config,
        logger=logger
    )
    
    classification_agent = ClassificationAgent(
        config=classification_config,
        logger=logger, 
        llm_service=llm_service
    )

    # Create FULLY DICT-BASED wrappers
    entity_wrapper = create_entity_disambiguation_wrapper(entity_agent, logger)
    search_wrapper = create_search_agent_wrapper(search_agent, logger)
    
    # Use existing wrappers for classification and conflict resolution (ensure they're dict-compatible)
    classification_node = create_classification_wrapper(classification_agent, logger)
    # Your existing code (keep this)
    conflict_resolution_node = create_conflict_resolution_node(
    config=conflict_config,
    logger=logger,
    llm_service=llm_service,
    brave_search_service=brave_search_service
)

# ADD this single line to wrap it
    conflict_resolution_node = create_conflict_resolution_wrapper(
    conflict_resolution_node, 
    logger
)

    # Add agent nodes
    workflow.add_node("disambiguate", entity_wrapper)  
    workflow.add_node("search", search_wrapper)   
    workflow.add_node("classify", classification_node)
    workflow.add_node("conflict_resolution", conflict_resolution_node)

    # Add support nodes
    workflow.add_node("manual_review", manual_review_handler)
    workflow.add_node("failed", failed_processing_handler)
    workflow.add_node("clean_entity_final", clean_entity_handler)
    workflow.add_node("context_required", context_required_handler)
    workflow.add_node("retry_with_context", retry_with_context_handler)  # NEW NODE
        
    # Set entry point
    workflow.set_entry_point("disambiguate")

    # Conditional edge for disambiguation
    workflow.add_conditional_edges(
        "disambiguate",
        should_continue_after_disambiguation,  
        {
            "search_strategy": "search",
            "manual_review": "manual_review",
            "context_required": "context_required",
            "retry_disambiguation": "disambiguate",
            "failed": "failed"
        }
    )

    # NEW: Conditional edge for context provision
    workflow.add_conditional_edges(
        "context_required",
        should_retry_with_context,
        {
            "retry_disambiguation": "retry_with_context",
            "still_waiting": END  # Stay in waiting state
        }
    )

    # NEW: Edge from retry_with_context back to disambiguation
    workflow.add_edge("retry_with_context", "disambiguate")

    workflow.add_conditional_edges(
        "search",
        should_continue_after_search, 
        {
            "clean_entity": "clean_entity_final",
            "retry": "disambiguate",
            "continue": "classify"
        }
    )

    workflow.add_edge("classify", "conflict_resolution")
    workflow.add_edge("conflict_resolution", END)

    # Terminal edges
    workflow.add_edge("manual_review", END)
    workflow.add_edge("failed", END)
    workflow.add_edge("clean_entity_final", END)
    # NOTe: context_required now has conditional routing instead of direct END

    return workflow

# USAGE EXAMPLE: How to resume workflow with additional context
"""
# When user provides additional context, update the state:
state['additional_context'] = {
    'date_of_birth': '1985-03-15',
    'occupation': 'Software Engineer',
    'location': 'San Francisco, CA'
}

# Then resume the workflow from context_required node:
result = workflow.invoke(state, {"configurable": {"thread_id": "resume_123"}})
"""