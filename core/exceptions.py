# adverse_media_system/core/exceptions.py

"""
Custom exceptions for the adverse media system.
Provides a structured error hierarchy, consistent error handling,
and includes utility functions for error management and reporting.
"""

from typing import Optional, Dict, Any, List, Tuple
from enum import Enum


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BaseAdverseMediaException(Exception):
    """
    Base exception class for all adverse media system errors.
    All custom exceptions in the system should inherit from this class.
    """

    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 agent_name: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.

        Args:
            message: Error message
            severity: Severity level of the error
            agent_name: Name of the agent where error occurred
            context: Additional context information (e.g., entity_id, query, article_id)
        """
        self.message = message
        self.severity = severity
        self.agent_name = agent_name
        self.context = context or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        agent_info = f"[{self.agent_name}] " if self.agent_name else ""
        context_str = ", ".join(f"{k}={v}" for k, v in self.context.items()) if self.context else ""
        if context_str:
            return f"{agent_info}{self.message} (Context: {context_str})"
        return f"{agent_info}{self.message}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            'message': self.message,
            'severity': self.severity.value,
            'agent_name': self.agent_name,
            'context': self.context,
            'exception_type': self.__class__.__name__
        }

    @staticmethod
    def _filter_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Filter out parameters that conflict with BaseAdverseMediaException.__init__"""
        return {k: v for k, v in kwargs.items() if k not in ['severity', 'context', 'agent_name']}
    
    @staticmethod
    def _merge_context(existing_context: Dict[str, Any], kwargs_context: Dict[str, Any]) -> Dict[str, Any]:
        """Safely merge context dictionaries"""
        merged = kwargs_context.copy() if kwargs_context else {}
        merged.update(existing_context)
        return merged


# =============================================================================
# System-wide Core Exceptions (previously in base_agents.py context)
# =============================================================================

class AgentConfigurationError(BaseAdverseMediaException):
    """Raised when an agent's configuration is invalid or missing.
    This directly replaces the generic AgentConfigurationError from base_agents.py.
    """
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        self.config_key = config_key
        context = self._merge_context({"config_key": config_key}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.CRITICAL), context=context, **self._filter_kwargs(kwargs))

class AgentInitializationError(BaseAdverseMediaException):
    """Raised when an agent or core component initialization fails."""
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        self.component = component
        context = self._merge_context({"component": component}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))

class AgentExecutionError(BaseAdverseMediaException):
    """
    A general error indicating a failure during an agent's run cycle.
    Most specific operational errors will inherit from more specific classes below.
    This serves as a high-level catch-all for unexpected agent run failures.
    """
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), **self._filter_kwargs(kwargs))


# =============================================================================
# Configuration and Initialization Exceptions
# (Unified from both files, preferring second file's structure)
# =============================================================================

# ConfigurationError from second file is more robust, AgentConfigurationError now directly inherits from it.
# We'll use AgentConfigurationError as the primary name for consistency with base_agents.py.
# We keep ModelLoadError from the first file.

class ConfigurationError(AgentConfigurationError): # AgentConfigurationError is now the specific one
    """Raised when configuration is invalid or missing. Alias for AgentConfigurationError."""
    # This class exists for clarity and potential future different types of config errors
    pass

class ModelLoadError(BaseAdverseMediaException):
    """Raised when ML model loading fails (e.g., embedding model, NER model)."""
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        self.model_name = model_name
        context = self._merge_context({"model_name": model_name}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.CRITICAL), context=context, **self._filter_kwargs(kwargs))


# =============================================================================
# Data Validation Exceptions
# (Unified from both files)
# =============================================================================

class ValidationError(BaseAdverseMediaException):
    """Base class for data validation failures."""
    def __init__(self, message: str, data_type: Optional[str] = None,
                 missing_fields: Optional[List[str]] = None, **kwargs):
        self.data_type = data_type
        self.missing_fields = missing_fields or []
        context = self._merge_context({"data_type": data_type, "missing_fields": missing_fields}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.MEDIUM), context=context, **self._filter_kwargs(kwargs))


class InputValidationError(ValidationError):
    """Raised when input data validation fails."""
    pass


class OutputValidationError(ValidationError):
    """Raised when output data validation fails."""
    pass


class DataFormatError(ValidationError):
    """Raised when data format is incorrect."""
    def __init__(self, message: str, expected_format: Optional[str] = None,
                 actual_format: Optional[str] = None, **kwargs):
        self.expected_format = expected_format
        self.actual_format = actual_format
        context = self._merge_context({"expected_format": expected_format, "actual_format": actual_format}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class SchemaValidationError(ValidationError):
    """Raised when data doesn't match expected schema."""
    pass


class StateValidationError(ValidationError):
    """Raised when `AdverseMediaState` validation fails."""
    def __init__(self, message: str, state_field: Optional[str] = None, **kwargs):
        self.state_field = state_field
        context = self._merge_context({"state_field": state_field}, kwargs.get("context", {}))
        super().__init__(message, data_type="state", context=context, **self._filter_kwargs(kwargs))


class ArticleValidationError(ValidationError):
    """Raised when article data validation fails."""
    def __init__(self, message: str, article_title: Optional[str] = None, **kwargs):
        self.article_title = article_title
        context = self._merge_context({"article_title": article_title}, kwargs.get("context", {}))
        super().__init__(message, data_type="article", context=context, **self._filter_kwargs(kwargs))


# =============================================================================
# External Service & API Exceptions
# (Unified from both files)
# =============================================================================

class ExternalServiceError(BaseAdverseMediaException):
    """Base class for errors interacting with external APIs or services."""
    def __init__(self, message: str, service_name: Optional[str] = None, **kwargs):
        self.service_name = service_name
        context = self._merge_context({"service_name": service_name}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class APIError(ExternalServiceError):
    """Raised when a generic API call fails."""
    def __init__(self, message: str, endpoint: Optional[str] = None,
                 status_code: Optional[int] = None, api_response: Optional[Dict] = None, **kwargs):
        self.endpoint = endpoint
        self.status_code = status_code
        self.api_response = api_response
        context = self._merge_context({"endpoint": endpoint, "status_code": status_code, "api_response": api_response}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class RateLimitError(APIError):
    """Raised when an API call is rate-limited."""
    def __init__(self, message: str, retry_after: Optional[int] = None, **kwargs):
        self.retry_after = retry_after
        context = self._merge_context({"retry_after": retry_after}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))


class SearchError(ExternalServiceError):
    """Base class for search-related errors."""
    def __init__(self, message: str, query: Optional[str] = None, **kwargs):
        self.query = query
        context = self._merge_context({"query": query}, kwargs.get("context", {}))
        super().__init__(message, service_name="web_search", context=context, **self._filter_kwargs(kwargs))

class SearchTimeoutError(SearchError):
    """Raised when a search operation times out."""
    pass

class SearchAPIError(SearchError):
    """Raised when a search API returns an error."""
    # This takes the additional attributes from the first file's SearchAPIError
    def __init__(self, message: str, status_code: Optional[int] = None,
                 api_response: Optional[Dict] = None, **kwargs):
        self.status_code = status_code
        self.api_response = api_response
        context = self._merge_context({"status_code": status_code, "api_response": api_response}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class SearchQuotaExceededError(SearchError):
    """Raised when search API quota is exceeded."""
    pass


class NoSearchResultsError(SearchError):
    """Raised when no search results are found for a query."""
    pass

class ContentExtractionError(ExternalServiceError):
    """Base class for content extraction errors (e.g., from URLs)."""
    pass

class URLExtractionError(ContentExtractionError):
    """Raised when URL content extraction fails."""
    def __init__(self, message: str, url: Optional[str] = None, **kwargs):
        self.url = url
        context = self._merge_context({"url": url}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class ContentParsingError(ContentExtractionError):
    """Raised when extracted content parsing fails."""
    pass


class ContentFilteringError(ContentExtractionError):
    """Raised when content filtering (e.g., boilerplate removal) fails."""
    pass


# =============================================================================
# Processing Errors (General Agent Operations)
# (Unified from both files, enhancing with more specific types)
# =============================================================================

class ProcessingError(BaseAdverseMediaException):
    """Base class for general processing-related errors within an agent."""
    pass

class MaxRetriesExceededError(ProcessingError):
    """Raised when an operation fails after exhausting all retries."""
    def __init__(self, message: str, max_attempts: int, current_attempts: int, **kwargs):
        self.max_attempts = max_attempts
        self.current_attempts = current_attempts
        context = self._merge_context({"max_attempts": max_attempts, "current_attempts": current_attempts}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))


class BatchProcessingError(ProcessingError):
    """Raised when batch processing fails for a subset or entire batch."""
    def __init__(self, message: str, batch_size: Optional[int] = None,
                 processed_count: Optional[int] = None, **kwargs):
        self.batch_size = batch_size
        self.processed_count = processed_count
        context = self._merge_context({"batch_size": batch_size, "processed_count": processed_count}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class MaxErrorsExceededError(ProcessingError):
    """Raised when a maximum cumulative error threshold is exceeded during processing."""
    def __init__(self, message: str, max_errors: int, current_errors: int, **kwargs):
        self.max_errors = max_errors
        self.current_errors = current_errors
        context = self._merge_context({"max_errors": max_errors, "current_errors": current_errors}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.CRITICAL), context=context, **self._filter_kwargs(kwargs))


class HumanReviewRequiredError(ProcessingError):
    """Raised when human intervention is deemed necessary for an item or case."""
    def __init__(self, message: str, review_reason: Optional[str] = None, **kwargs):
        self.review_reason = review_reason
        context = self._merge_context({"review_reason": review_reason}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))


class FilteringError(ProcessingError):
    """Raised when a filtering operation fails."""
    pass


class ScoringError(ProcessingError):
    """Raised when a scoring operation fails."""
    pass


class EmbeddingError(ProcessingError):
    """Raised when text embedding calculation fails."""
    def __init__(self, message: str, text_chunk: Optional[str] = None, **kwargs):
        context = self._merge_context({"text_chunk": text_chunk}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class SimilarityCalculationError(ProcessingError):
    """Raised when similarity calculation fails."""
    pass


# =============================================================================
# Disambiguation Specific Exceptions
# (From first file)
# =============================================================================

class DisambiguationError(ProcessingError): # Inherit from ProcessingError for clearer hierarchy
    """Base class for entity disambiguation errors."""
    pass


class EntityExtractionError(DisambiguationError):
    """Raised when named entity extraction fails."""
    pass


class ContextExtractionError(DisambiguationError):
    """Raised when context extraction for disambiguation fails."""
    pass


class CandidateValidationError(DisambiguationError):
    """Raised when candidate entities for disambiguation fail validation."""
    pass


class AmbiguousEntityError(DisambiguationError):
    """Raised when an entity cannot be unambiguously disambiguated."""
    def __init__(self, message: str, entity_name: Optional[str] = None, candidates: Optional[List[Dict]] = None, **kwargs):
        self.entity_name = entity_name
        self.candidates = candidates or []
        context = self._merge_context({"entity_name": entity_name, "candidate_count": len(candidates) if candidates else 0}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.MEDIUM), context=context, **self._filter_kwargs(kwargs))


# =============================================================================
# Quality Assessment Specific Exceptions
# (From first file)
# =============================================================================

class QualityAssessmentError(ProcessingError): # Inherit from ProcessingError
    """Base class for quality assessment errors (e.g., relevance, credibility)."""
    pass


class RelevanceCalculationError(QualityAssessmentError):
    """Raised when relevance calculation fails."""
    pass


class CredibilityAssessmentError(QualityAssessmentError):
    """Raised when credibility assessment fails."""
    pass


class QualityThresholdError(QualityAssessmentError):
    """Raised when a quality score does not meet a required threshold."""
    def __init__(self, message: str, actual_quality: float,
                 required_quality: float, **kwargs):
        self.actual_quality = actual_quality
        self.required_quality = required_quality
        context = self._merge_context({"actual_quality": actual_quality, "required_quality": required_quality}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.MEDIUM), context=context, **self._filter_kwargs(kwargs))


# =============================================================================
# Strategy Specific Exceptions
# (From first file)
# =============================================================================

class StrategyError(ProcessingError): # Inherit from ProcessingError
    """Base class for strategy-related errors (e.g., search strategy selection)."""
    pass


class StrategySelectionError(StrategyError):
    """Raised when dynamic strategy selection fails."""
    pass


class StrategyExecutionError(StrategyError):
    """Raised when a selected strategy fails during execution."""
    pass


# =============================================================================
# DSPy / LLM Specific Exceptions
# (Unified from both files, favoring explicit DSPy/LLM focus)
# =============================================================================

class LLMError(BaseAdverseMediaException):
    """Base class for all LLM-related operational failures."""
    def __init__(self, message: str, llm_name: Optional[str] = None, **kwargs):
        self.llm_name = llm_name
        context = self._merge_context({"llm_name": llm_name}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))


class PromptError(LLMError):
    """Raised when prompt generation or processing (e.g., template rendering) fails."""
    pass


class DSPyExecutionError(LLMError): # Inherits from LLMError, better than ClassificationError
    """Raised when a DSPy module or program execution fails."""
    def __init__(self, message: str, module_name: Optional[str] = None, **kwargs):
        self.module_name = module_name
        context = self._merge_context({"module_name": module_name}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))

class SignatureError(DSPyExecutionError):
    """Raised when DSPy signature validation fails."""
    pass


# =============================================================================
# Classification Specific Exceptions
# (From second file)
# =============================================================================

class ClassificationError(ProcessingError): # Inherit from ProcessingError
    """Base class for classification-related errors."""
    def __init__(self, message: str, article_title: Optional[str] = None,
                 entity_name: Optional[str] = None, **kwargs):
        self.article_title = article_title
        self.entity_name = entity_name
        context = self._merge_context({"article_title": article_title, "entity_name": entity_name}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class LLMClassificationError(ClassificationError, LLMError): # Can inherit from both if needed, or primarily ClassificationError
    """Raised when LLM-based classification fails."""
    # We combine the attributes and inherit from ClassificationError for hierarchy.
    # LLMError can be mixed in if its methods are needed, or simply duplicate attributes if not.
    # For now, prioritizing the ClassificationError hierarchy.
    def __init__(self, message: str, llm_name: Optional[str] = None, **kwargs):
        self.llm_name = llm_name
        # Pass kwargs for ClassificationError parent, and llm_name for LLMError context
        context = self._merge_context({"llm_name": llm_name}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))


class ClassificationTimeoutError(ClassificationError):
    """Raised when classification takes too long."""
    def __init__(self, message: str, timeout_seconds: Optional[float] = None, **kwargs):
        self.timeout_seconds = timeout_seconds
        context = self._merge_context({"timeout_seconds": timeout_seconds}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.MEDIUM), context=context, **self._filter_kwargs(kwargs))


class MultipleClassificationFailuresError(ClassificationError):
    """Raised when multiple classification attempts for a single item fail."""
    def __init__(self, message: str, failure_count: int, **kwargs):
        self.failure_count = failure_count
        context = self._merge_context({"failure_count": failure_count}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.CRITICAL), context=context, **self._filter_kwargs(kwargs))


# =============================================================================
# Conflict Resolution Specific Exceptions
# (From second file)
# =============================================================================

class ConflictResolutionError(ProcessingError): # Inherit from ProcessingError
    """Base class for conflict resolution errors."""
    def __init__(self, message: str, article_title: Optional[str] = None,
                 conflict_type: Optional[str] = None, **kwargs):
        self.article_title = article_title
        self.conflict_type = conflict_type
        context = self._merge_context({"article_title": article_title, "conflict_type": conflict_type}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class ConflictDetectionError(ConflictResolutionError):
    """Raised when conflict detection fails."""
    pass


class ResolutionMethodError(ConflictResolutionError):
    """Raised when a specific resolution method fails."""
    def __init__(self, message: str, method_name: Optional[str] = None, **kwargs):
        self.method_name = method_name
        context = self._merge_context({"method_name": method_name}, kwargs.get("context", {}))
        super().__init__(message, context=context, **self._filter_kwargs(kwargs))


class ArbitrationError(ConflictResolutionError):
    """Raised when LLM arbitration fails to resolve a conflict."""
    def __init__(self, message: str, arbitration_confidence: Optional[float] = None, **kwargs):
        self.arbitration_confidence = arbitration_confidence
        context = self._merge_context({"arbitration_confidence": arbitration_confidence}, kwargs.get("context", {}))
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.HIGH), context=context, **self._filter_kwargs(kwargs))


class CriticalConflictError(ConflictResolutionError):
    """Raised when a critical conflict cannot be resolved automatically and requires immediate attention."""
    def __init__(self, message: str, **kwargs):
        super().__init__(message, severity=kwargs.get('severity', ErrorSeverity.CRITICAL), **self._filter_kwargs(kwargs))

class MissingDataError(ValidationError):
    """Raised when required data is missing."""
    def __init__(self, message: str, missing_fields: Optional[List[str]] = None, **kwargs):
        super().__init__(message, missing_fields=missing_fields, **kwargs)


# =============================================================================
# Helper Functions (Unified and adapted)
# =============================================================================

def create_error_context(entity_name: str = None, query: str = None,
                         strategy: str = None, article_id: str = None,
                         **kwargs) -> Dict[str, Any]:
    """
    Helper function to create a standardized error context dictionary.
    Combines fields from both previous versions.
    """
    context = {}
    if entity_name:
        context["entity_name"] = entity_name
    if query:
        context["query"] = query
    if strategy:
        context["strategy"] = strategy
    if article_id:
        context["article_id"] = article_id # Added for consistency with ArticleValidationError etc.
    context.update(kwargs)
    return context


def handle_api_error(response: Dict, operation: str, service_name: str = "unknown_service") -> None:
    """
    Helper function to parse common API error responses and raise appropriate exceptions.
    Adapted to use the new APIError and RateLimitError hierarchy.
    Assumes 'response' dict might contain an 'error' key with 'code' and 'message'.
    """
    if "error" in response:
        error_info = response["error"]
        if isinstance(error_info, dict):
            error_code = error_info.get("code", "unknown").lower()
            error_message = error_info.get("message", f"Unknown API error for {operation}")
            status_code = error_info.get("status_code", None)

            common_context = create_error_context(
                operation=operation,
                service_name=service_name,
                raw_response=response
            )

            if "rate_limit" in error_code or status_code == 429:
                retry_after = error_info.get("retry_after", None) # Some APIs send this
                raise RateLimitError(
                    f"{service_name} rate limit exceeded for {operation}: {error_message}",
                    retry_after=retry_after,
                    context=common_context
                )
            elif "timeout" in error_code:
                # Could be SearchTimeoutError or a generic API timeout
                if service_name == "web_search":
                    raise SearchTimeoutError(f"{service_name} timeout for {operation}: {error_message}", context=common_context)
                else:
                    raise APIError(f"{service_name} timeout for {operation}: {error_message}", endpoint=operation, status_code=status_code, context=common_context)
            elif "quota" in error_code:
                 if service_name == "web_search":
                    raise SearchQuotaExceededError(f"{service_name} quota exceeded for {operation}: {error_message}", context=common_context)
                 else: # Generic API quota
                    raise APIError(f"{service_name} quota exceeded for {operation}: {error_message}", endpoint=operation, status_code=status_code, context=common_context)
            elif status_code is not None and status_code >= 400: # Generic HTTP client/server error
                 raise APIError(
                    f"{service_name} API error (Status: {status_code}) for {operation}: {error_message}",
                    endpoint=operation, status_code=status_code, api_response=response, context=common_context
                 )
            else:
                raise APIError(
                    f"{service_name} API error for {operation}: {error_message}",
                    endpoint=operation, api_response=response, context=common_context
                )
        else:
            raise APIError(f"{service_name} API error for {operation}: {error_info}", api_response=response)

def should_escalate_to_human_review(exceptions: List[BaseAdverseMediaException]) -> Tuple[bool, str]:
    """
    Determine if a list of exceptions should trigger human review based on severity
    thresholds or specific error types.
    """
    if not exceptions:
        return False, "No errors to evaluate"

    # Check for critical errors
    critical_errors = [
        exc for exc in exceptions
        if isinstance(exc, BaseAdverseMediaException) and exc.severity == ErrorSeverity.CRITICAL
    ]

    if critical_errors:
        return True, f"Critical errors detected: {len(critical_errors)} critical error(s)"

    # Check for multiple high-severity errors
    high_errors = [
        exc for exc in exceptions
        if isinstance(exc, BaseAdverseMediaException) and exc.severity == ErrorSeverity.HIGH
    ]

    if len(high_errors) >= 3: # Threshold of 3 high-severity errors
        return True, f"Multiple high-severity errors: {len(high_errors)} high-severity error(s)"

    # Check for specific error types that inherently require human review
    escalation_types = [
        CriticalConflictError,
        MultipleClassificationFailuresError,
        HumanReviewRequiredError
    ]

    for exc in exceptions:
        if any(isinstance(exc, exc_type) for exc_type in escalation_types):
            return True, f"Specific error type requires human review: {exc.__class__.__name__}"

    return False, "No escalation required"

def validate_required_fields(data: Dict, required_fields: List[str],
                             operation: str = "operation") -> None:
    """
    Helper function to validate required fields in a dictionary.
    Raises InputValidationError if any field is missing or empty.
    """
    missing_fields = [field for field in required_fields if field not in data or not data[field]]

    if missing_fields:
        raise InputValidationError(
            f"Missing required fields for {operation}: {', '.join(missing_fields)}",
            missing_fields=missing_fields,
            context={"operation": operation}
        )


# =============================================================================
# Error Mapping Utility
# =============================================================================

# Map common error patterns (keywords) to specific exception classes
# This allows for a more generic way to raise specific exceptions based on a string identifier.
ERROR_MAPPING: Dict[str, type[BaseAdverseMediaException]] = {
    "timeout": SearchTimeoutError, # Specific search timeout
    "rate_limit": RateLimitError,
    "quota": SearchQuotaExceededError, # Specific search quota
    "api_key": AgentConfigurationError, # Configuration error for API keys
    "no_results": NoSearchResultsError,
    "ambiguous": AmbiguousEntityError,
    "validation": ValidationError,
    "model_load": ModelLoadError,
    "embedding": EmbeddingError,
    "similarity": SimilarityCalculationError,
    "classification_timeout": ClassificationTimeoutError,
    "dspy_execution": DSPyExecutionError,
    "arbitration": ArbitrationError,
    "human_review": HumanReviewRequiredError,
    "max_errors": MaxErrorsExceededError,
    "max_retries": MaxRetriesExceededError,
    "critical_conflict": CriticalConflictError,
    "data_format": DataFormatError,
    "missing_data": MissingDataError,
    "content_parsing": ContentParsingError,
    "url_extraction": URLExtractionError,
    "strategy_execution": StrategyExecutionError,
}


def map_error(error_type_key: str, message: str, **kwargs) -> BaseAdverseMediaException:
    """
    Maps an error type key (from ERROR_MAPPING) to the appropriate exception class
    and instantiates it with the given message and keyword arguments.
    """
    exception_class = ERROR_MAPPING.get(error_type_key, BaseAdverseMediaException)
    # Ensure context is properly merged if passed in kwargs
    if 'context' in kwargs:
        kwargs['context'] = create_error_context(**kwargs['context']) # Use helper to standardize context
    return exception_class(message, **kwargs)


def create_error_summary(exceptions: List[BaseAdverseMediaException]) -> Dict[str, Any]:
    """
    Create a summary of multiple exceptions, providing breakdowns by severity,
    agent, and exception type.
    """
    if not exceptions:
        return {"total_errors": 0, "summary": "No errors"}

    severity_counts = {}
    agent_counts = {}
    error_types = {}

    for exc in exceptions:
        if isinstance(exc, BaseAdverseMediaException): # Ensure it's our custom exception
            severity = exc.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            if exc.agent_name:
                agent = exc.agent_name
                agent_counts[agent] = agent_counts.get(agent, 0) + 1

            error_type = exc.__class__.__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
        else:
            # Handle non-BaseAdverseMediaException if they somehow get here
            error_type = exc.__class__.__name__
            error_types[error_type] = error_types.get(error_type, 0) + 1
            severity_counts["unknown"] = severity_counts.get("unknown", 0) + 1

    return {
        "total_errors": len(exceptions),
        "severity_breakdown": severity_counts,
        "agent_breakdown": agent_counts,
        "error_type_breakdown": error_types,
        "has_critical_errors": any(
            isinstance(exc, BaseAdverseMediaException) and exc.severity == ErrorSeverity.CRITICAL
            for exc in exceptions
        )
    }