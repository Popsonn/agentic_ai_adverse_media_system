from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import uuid
from enum import Enum

# Import exceptions and models (make sure these resolve correctly in your project)
from core.exceptions import BaseAdverseMediaException, ErrorSeverity
from models.entity import DisambiguationResult, EntityCandidate
from models.search import SearchResult
from models.results import ResolvedArticleResult
# Add this import at the top
from models.results import ClassifiedArticleResult


class AgentStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    NEEDS_CONTEXT = "needs_context"
    #ERROR = "error"
    #SKIPPED = "skipped"


class ProcessingStatus(Enum):
    INITIALIZED = "initialized"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_review"
    AWAITING_CONTEXT = "awaiting_context"  
    CANCELLED = "cancelled"
    PAUSED = "paused"

@dataclass
class AgentLogEntry:
    agent_name: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    level: str = "INFO"
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # Safe timestamp conversion
        if isinstance(self.timestamp, datetime):
            timestamp_str = self.timestamp.isoformat()
        elif isinstance(self.timestamp, str):
            timestamp_str = self.timestamp
        else:
            timestamp_str = str(self.timestamp)
            
        return {
            "timestamp": timestamp_str,
            "agent_name": self.agent_name,
            "message": self.message,
            "level": self.level,
            "context": self.context,
        }

@dataclass
class AgentErrorEntry:
    agent_name: str
    error: BaseAdverseMediaException
    timestamp: datetime = field(default_factory=datetime.now)
    item_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        # Safe timestamp conversion
        if isinstance(self.timestamp, datetime):
            timestamp_str = self.timestamp.isoformat()
        elif isinstance(self.timestamp, str):
            timestamp_str = self.timestamp
        else:
            timestamp_str = str(self.timestamp)
            
        return {
            "timestamp": timestamp_str,
            "agent_name": self.agent_name,
            "error_type": self.error.__class__.__name__,
            "message": self.error.message,
            "severity": getattr(self.error.severity, "value", "UNKNOWN"),
            "item_id": self.item_id,
            "context": getattr(self.error, "context", {}),
        }

@dataclass
class AgentResult:
    agent_name: str
    status: AgentStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[BaseAdverseMediaException] = None
    metrics: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    
    @property
    def execution_time(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
        
    def mark_completed(self, output_data: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, Any]] = None):
        self.status = AgentStatus.COMPLETED
        self.end_time = datetime.now()
        if output_data:
            self.output_data = output_data
        if metrics:
            self.metrics = metrics
            
    def mark_failed(self, error: BaseAdverseMediaException):
        self.status = AgentStatus.FAILED
        self.end_time = datetime.now()
        self.error = error

@dataclass
class AdverseMediaState:
    # Core identification
    workflow_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_name: str = ""
    user_input: str = ""
    
    # Overall processing state
    status: ProcessingStatus = ProcessingStatus.INITIALIZED
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    
    # Entity disambiguation results
    resolved_entity: Optional[EntityCandidate] = None
    all_candidates: List[EntityCandidate] = field(default_factory=list)
    disambiguation_confidence: float = 0.0
    disambiguation_result: Optional[DisambiguationResult] = None
    
    # Search results
    raw_search_results: List[SearchResult] = field(default_factory=list)
    filtered_search_results: List[SearchResult] = field(default_factory=list)
    search_quality_metrics: Dict[str, Any] = field(default_factory=dict)  # Changed to Any
    user_context: Optional[Dict[str, str]] = None
    strategies_attempted: List[str] = field(default_factory=list)
    
    # Classification results
    classified_articles: List[ClassifiedArticleResult] = field(default_factory=list)
    classification_metrics: Dict[str, Any] = field(default_factory=dict)
    classification_errors: int = 0
    
    # Conflict resolution results
    resolved_articles: List[ResolvedArticleResult] = field(default_factory=list)
    conflict_resolution_summary: Dict[str, Any] = field(default_factory=dict)
    conflict_resolution_error: Optional[str] = None
    
    # Agent execution tracking
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    current_agent: Optional[str] = None
    completed_agents: Set[str] = field(default_factory=set)
    
    # Logging and error tracking
    logs: List[str] = field(default_factory=list)  # simple logs for backwards compatibility
    agent_logs: List[AgentLogEntry] = field(default_factory=list)
    agent_errors: List[AgentErrorEntry] = field(default_factory=list)
    
    # Review and validation
    human_review_required: bool = False
    review_reasons: List[str] = field(default_factory=list)
    
    # Additional data storage
    metrics: Dict[str, Any] = field(default_factory=dict)
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    
    # FIX: Add workflow control flags
    skip_to_final_report: bool = False  # For clean entities

    #context handling
    additional_context: Optional[Dict[str, str]] = None
    context_request: Optional[Dict[str, Any]] = None
    
    def start_processing(self):
        """Initialize processing state"""
        self.status = ProcessingStatus.IN_PROGRESS
        self.processing_start_time = datetime.now()
        self.update_timestamp()
    
    def complete_processing(self, status: ProcessingStatus = ProcessingStatus.COMPLETED):
        """Mark overall processing as complete"""
        self.status = status
        self.processing_end_time = datetime.now()
        self.current_agent = None
        self.update_timestamp()
    
    def start_agent(self, agent_name: str):
        """Start execution of a specific agent"""
        self.current_agent = agent_name
        if agent_name not in self.agent_results:
            self.agent_results[agent_name] = AgentResult(
                agent_name=agent_name,
                status=AgentStatus.IN_PROGRESS,
                start_time=datetime.now()
            )
        else:
            # Retry case
            result = self.agent_results[agent_name]
            result.status = AgentStatus.IN_PROGRESS
            result.start_time = datetime.now()
            result.retry_count += 1

        if agent_name not in self.execution_order:
            self.execution_order.append(agent_name)
        
        # Start overall processing if not started
        if self.status == ProcessingStatus.INITIALIZED:
            self.status = ProcessingStatus.IN_PROGRESS
        
        self.update_timestamp()
    
    def complete_agent(self, agent_name: str, status: AgentStatus = AgentStatus.COMPLETED,
                       output_data: Optional[Dict[str, Any]] = None,
                       metrics: Optional[Dict[str, Any]] = None):
        """Mark agent as completed with optional output data"""
        if agent_name in self.agent_results:
            result = self.agent_results[agent_name]
            result.status = status
            result.end_time = datetime.now()
            if output_data:
                result.output_data = output_data
                # Also store in agent_outputs for backward compatibility
                self.agent_outputs[agent_name] = output_data
            if metrics:
                result.metrics = metrics
            
            if status == AgentStatus.COMPLETED:
                self.completed_agents.add(agent_name)
        
        if self.current_agent == agent_name:
            self.current_agent = None
        
        self.update_timestamp()
    
    def fail_agent(self, agent_name: str, error: BaseAdverseMediaException):
        """Mark agent as failed with error information"""
        if agent_name in self.agent_results:
            self.agent_results[agent_name].mark_failed(error)

        self.add_error(agent_name, error)
        
        # Set overall status to failed for critical errors
        if hasattr(error, 'severity') and error.severity.value == 'CRITICAL':
            self.status = ProcessingStatus.FAILED
        
        self.update_timestamp()
    
    def is_agent_completed(self, agent_name: str) -> bool:
        """Check if a specific agent has completed successfully"""
        return (agent_name in self.agent_results and
                self.agent_results[agent_name].status == AgentStatus.COMPLETED)
    
    def get_failed_agents(self) -> List[str]:
        """Get list of agent names that have failed"""
        return [name for name, result in self.agent_results.items()
                if result.status == AgentStatus.FAILED]
    
    def can_proceed_to_agent(self, agent_name: str, required_agents: Optional[List[str]] = None) -> bool:
        """Check if workflow can proceed to a specific agent"""
        if required_agents:
            return all(self.is_agent_completed(req_agent) for req_agent in required_agents)
        return True
    
    def update_timestamp(self):
        """Update the last modified timestamp"""
        self.updated_at = datetime.now()
    
    def is_ready_for_next_agent(self, required_fields: List[str]) -> bool:
        """Check if state has required fields populated for next agent"""
        for field_name in required_fields:
            if not hasattr(self, field_name) or getattr(self, field_name) is None:
                return False
        return True
    
    def set_review_required(self, reason: str, agent_name: Optional[str] = None):
        """Mark state as requiring human review"""
        self.human_review_required = True
        full_reason = f"[{agent_name}] {reason}" if agent_name else reason
        self.review_reasons.append(full_reason)
        
        if self.status == ProcessingStatus.IN_PROGRESS:
            self.status = ProcessingStatus.NEEDS_REVIEW
        
        if agent_name:
            self.add_log(agent_name, f"Review required: {reason}", level="WARNING")
        
        self.update_timestamp()
    
    def add_log(self, agent_name: str, message: str, level: str = "INFO",
                item_id: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """Add a log entry from an agent"""
        log_message = f"[{agent_name}] {message}"
        self.logs.append(log_message)
        
        entry = AgentLogEntry(
            agent_name=agent_name,
            message=message,
            timestamp=datetime.now(),
            level=level,
            context=context or {}
        )
        self.agent_logs.append(entry)
        self.update_timestamp()
    
    def add_error(self, agent_name: str, error: BaseAdverseMediaException, item_id: Optional[str] = None):
        """Add an error entry from an agent"""
        entry = AgentErrorEntry(
            agent_name=agent_name,
            error=error,
            timestamp=datetime.now(),
            item_id=item_id
        )
        self.agent_errors.append(entry)
        
        # Also add to simple logs for backward compatibility
        error_log = f"[{agent_name}] ERROR: {error.message}"
        self.logs.append(error_log)
        
        self.update_timestamp()
    
    def get_processing_duration(self) -> Optional[float]:
        """Get total processing duration in seconds"""
        if self.processing_start_time is None:
            return None
        end_time = self.processing_end_time or datetime.now()
        return (end_time - self.processing_start_time).total_seconds()
    
    def has_errors(self) -> bool:
        """Check if any errors have occurred"""
        return len(self.agent_errors) > 0
    
    def has_critical_errors(self) -> bool:
        """Check if any critical errors have occurred"""
        return any(
            hasattr(error.error, 'severity') and error.error.severity == ErrorSeverity.CRITICAL
            for error in self.agent_errors
        )
    
    def get_agent_execution_summary(self) -> Dict[str, Any]:
        """Get summary of agent execution statistics"""
        summary = {
            "total_agents": len(self.agent_results),
            "completed": len([r for r in self.agent_results.values() if r.status == AgentStatus.COMPLETED]),
            "failed": len([r for r in self.agent_results.values() if r.status == AgentStatus.FAILED]),
            "in_progress": len([r for r in self.agent_results.values() if r.status == AgentStatus.IN_PROGRESS]),
            "execution_times": {
                name: result.execution_time
                for name, result in self.agent_results.items()
                if result.execution_time is not None
            },
            "execution_order": self.execution_order.copy()
        }
        return summary
    
    def reset_for_retry(self):
        """Reset state for a complete retry of the workflow"""
        self.status = ProcessingStatus.INITIALIZED
        self.processing_start_time = None
        self.processing_end_time = None
        self.current_agent = None
        self.completed_agents.clear()
        self.agent_results.clear()
        self.execution_order.clear()
        self.agent_outputs.clear()
        self.logs.clear()
        self.agent_logs.clear()
        self.agent_errors.clear()
        self.human_review_required = False
        self.review_reasons.clear()
        self.skip_to_final_report = False
        self.update_timestamp()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        def safe_enum_value(enum_value):
            """Safely get enum value or string representation"""
            if hasattr(enum_value, 'value'):
                return enum_value.value
            return str(enum_value)
        
        return {
            "workflow_id": self.workflow_id,
            "entity_name": self.entity_name,
            "user_input": self.user_input,
            "status": safe_enum_value(self.status),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "processing_start_time": self.processing_start_time.isoformat() if self.processing_start_time else None,
            "processing_end_time": self.processing_end_time.isoformat() if self.processing_end_time else None,
            
            # Entity disambiguation
            "resolved_entity": self.resolved_entity.to_dict() if self.resolved_entity else None,
            "all_candidates": [c.to_dict() for c in self.all_candidates],
            "disambiguation_confidence": self.disambiguation_confidence,
            "disambiguation_result": self.disambiguation_result.to_dict() if self.disambiguation_result else None,
            
            # Search results
            "raw_search_results": [r.to_dict() for r in self.raw_search_results],
            "filtered_search_results": [r.to_dict() for r in self.filtered_search_results],
            "search_quality_metrics": self.search_quality_metrics,
            "user_context": self.user_context,
            "strategies_attempted": self.strategies_attempted,
            
            # Classification
            "classified_articles": [article.to_dict() for article in self.classified_articles],
            "classification_metrics": self.classification_metrics,
            "classification_errors": self.classification_errors,
            
            # Conflict resolution
            "resolved_articles": [article.to_dict() for article in self.resolved_articles],
            "conflict_resolution_summary": self.conflict_resolution_summary,
            "conflict_resolution_error": self.conflict_resolution_error,

            # Agent execution
            "agent_results": {
                k: {
                    "status": safe_enum_value(v.status),
                    "start_time": v.start_time.isoformat() if v.start_time else None,
                    "end_time": v.end_time.isoformat() if v.end_time else None,
                    "retry_count": v.retry_count,
                    "error": v.error.to_dict() if v.error and hasattr(v.error, 'to_dict') else str(v.error) if v.error else None,
                    "output_data": v.output_data,
                    "metrics": v.metrics,
                }
                for k, v in self.agent_results.items()
            },
            "execution_order": self.execution_order,
            "current_agent": self.current_agent,
            "completed_agents": list(self.completed_agents),
            
            # Logging
            "logs": self.logs,
            "agent_logs": [log.to_dict() for log in self.agent_logs],
            "agent_errors": [error.to_dict() for error in self.agent_errors],
            
            # Review
            "human_review_required": self.human_review_required,
            "review_reasons": self.review_reasons,
            
            # Additional data
            "metrics": self.metrics,
            "agent_outputs": self.agent_outputs,
            "skip_to_final_report": self.skip_to_final_report,
        }
    
    def __str__(self) -> str:
        duration = self.get_processing_duration()
        duration_str = f"{duration:.2f}s" if duration else "N/A"
        status_str = self.status.value if hasattr(self.status, 'value') else str(self.status)
        return (f"AdverseMediaState(entity='{self.entity_name}', "
                f"status={status_str}, "
                f"duration={duration_str}, "
                f"agents_completed={len(self.completed_agents)}, "
                f"errors={len(self.agent_errors)})")
    
    def __repr__(self) -> str:
        return self.__str__()
    
    @property
    def search_status(self) -> AgentStatus:
        return self.agent_results.get("search_agent",
                                  AgentResult("search_agent", AgentStatus.PENDING)).status

    @property
    def classification_status(self) -> AgentStatus:
        return self.agent_results.get("classification_agent",
                                  AgentResult("classification_agent", AgentStatus.PENDING)).status

    @property
    def disambiguation_status(self) -> AgentStatus:
        return self.agent_results.get("entity_disambiguation_agent",
                                  AgentResult("entity_disambiguation_agent", AgentStatus.PENDING)).status

    @property
    def conflict_resolution_status(self) -> AgentStatus:
        return self.agent_results.get("conflict_resolution_agent",
                                  AgentResult("conflict_resolution_agent", AgentStatus.PENDING)).status


     