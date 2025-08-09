from typing import Callable
import logging
from typing import Dict, Any
from datetime import datetime
import traceback

def create_classification_wrapper(classification_agent, logger):
    """NEW wrapper - converts dict to AdverseMediaState for classification agent"""
    def classification_wrapper(state_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            print(f"DEBUG: Classification wrapper input keys: {list(state_dict.keys())}")
            
            # Convert dict to AdverseMediaState object
            from core.state import AdverseMediaState, AgentResult, AgentStatus
            from models.entity import EntityCandidate, DisambiguationResult
            from models.search import SearchResult
            from models.results import ClassifiedArticleResult
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
                        elif key == 'classified_articles' and value:
                            if hasattr(ClassifiedArticleResult, 'from_dict'):
                                state_obj.classified_articles = [ClassifiedArticleResult.from_dict(a) if isinstance(a, dict) else a for a in value]
                        elif key == 'completed_agents' and isinstance(value, list):  # ADD THIS LINE
                            setattr(state_obj, key, set(value))
                        else:
                            setattr(state_obj, key, value)
                    except Exception as field_error:
                        print(f"WARNING: Could not set field {key}: {field_error}")
                        # Continue without this field
            
            # Add this right after the field copying loop in classification_wrapper
            print(f"DEBUG: Classification input - filtered_search_results count: {len(state_dict.get('filtered_search_results', []))}")
            print(f"DEBUG: Classification state_obj - filtered_search_results count: {len(getattr(state_obj, 'filtered_search_results', []))}")
            if state_dict.get('filtered_search_results'):
                print(f"DEBUG: First filtered result type: {type(state_dict['filtered_search_results'][0])}")
            
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
            
            print("DEBUG: Running classification agent...")
            
            # Call the actual agent method
            result_obj = classification_agent._run_implementation(state_obj)
            
            print("DEBUG: Classification agent completed, converting back to dict...")
            
            # FIX: Check if result is already a dict or needs conversion
            if isinstance(result_obj, dict):
                result_dict = result_obj
            else:
                result_dict = result_obj.to_dict()
            
            print("DEBUG: Successfully converted classification results back to dict")
            return result_dict
            
        except Exception as e:
            print(f"DEBUG: Exception in classification wrapper: {e}")
            import traceback
            traceback.print_exc()
            
            error_msg = f"Classification agent wrapper error: {e}"
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
                    'agent_name': 'classification_agent',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'CRITICAL'
                }]
            })
            return error_dict
    
    return classification_wrapper