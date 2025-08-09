# adverse_media_system/core/base_agents.py
import dspy
import logging
import time
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Tuple
from services.llm_service import LLMService

# Import the configuration dataclasses
from config.settings import BaseAgentConfig
from core.state import AdverseMediaState
# Import custom exceptions from the new consolidated exceptions.py
from core.exceptions import (
    BaseAdverseMediaException, # General base for type checking
    AgentConfigurationError,
    AgentInitializationError,
    AgentExecutionError,
    MaxRetriesExceededError,
    LLMError,
    DSPyExecutionError,
    SearchError,
    APIError,
    ErrorSeverity # To use for explicit severity setting
)

# For DSPy specific agents
try:
    import dspy
    from dspy import LM
    DSPY_AVAILABLE = True
except ImportError:
    dspy = None
    LM = None
    DSPY_AVAILABLE = False

@dataclass
class AgentMetrics:
    """
    Dataclass to hold performance metrics for an agent.
    """
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    errors: List[str] = field(default_factory=list)
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    items_processed: int = 0
    items_failed: int = 0
    api_calls_made: int = 0

    def reset(self):
        """Resets all metrics to their initial state."""
        self.total_runs = 0
        self.successful_runs = 0
        self.failed_runs = 0
        self.errors = []
        self.total_processing_time = 0.0
        self.average_processing_time = 0.0
        self.items_processed = 0
        self.items_failed = 0
        self.api_calls_made = 0

    def update_average_processing_time(self):
        """Calculates and updates the average processing time."""
        if self.successful_runs > 0:
            self.average_processing_time = self.total_processing_time / self.successful_runs
        else:
            self.average_processing_time = 0.0

    def add_error(self, error_message: str, max_errors: int = 1000):
        """Adds an error message to the list of errors with bounds checking."""
        if len(self.errors) >= max_errors:
            self.errors.pop(0)  # Remove oldest error to prevent unbounded growth
        self.errors.append(error_message)

class BaseAgent(ABC):
    """
    Abstract Base Class for all agents in the Adverse Media System.
    Provides common functionalities like configuration, logging, error handling,
    metrics tracking, and interaction with the AdverseMediaState.
    """

    def __init__(self, config: BaseAgentConfig, logger: logging.Logger):
        """
        Initializes the BaseAgent with a configuration and a logger.

        Args:
            config: An instance of BaseAgentConfig or a subclass thereof.
            logger: A pre-configured logging.Logger instance.
        """
        if not isinstance(config, BaseAgentConfig):
            raise AgentConfigurationError(
                f"Config must be an instance of BaseAgentConfig or its subclass, got {type(config)}.",
                config_key="base_agent_config_type",
                severity=ErrorSeverity.CRITICAL # Explicitly set severity
            )
        if not isinstance(logger, logging.Logger):
            raise AgentConfigurationError(
                f"Logger must be a logging.Logger instance, got {type(logger)}.",
                config_key="base_agent_logger_type",
                severity=ErrorSeverity.CRITICAL # Explicitly set severity
            )

        self.config: BaseAgentConfig = config
        self.logger: logging.Logger = logger
        self.metrics: AgentMetrics = AgentMetrics()
        self.agent_name: str = self.__class__.__name__

        self.logger.debug(f"BaseAgent '{self.agent_name}' initialized with config: {config}")
        self._current_state = None

    
    @abstractmethod
    def _run_implementation(self, state: AdverseMediaState) -> AdverseMediaState:
        """Abstract method to be implemented by concrete agents."""
        raise NotImplementedError("Each agent must implement its own '_run_implementation' method.")
    
    def run(self, state: AdverseMediaState) -> AdverseMediaState:
        """Main entry point for agent execution."""
        self._current_state = state
        try:
            return self._run_implementation(state)
        finally:
            self._current_state = None

    def _update_agent_status(self, state: AdverseMediaState, status_message: str, level: str = "INFO"):
        """
        Updates the status of the current agent in the AdverseMediaState.
        """
        if hasattr(state, 'add_log'):
            state.add_log(self.agent_name, status_message, level=level)
        self.logger.log(getattr(logging, level.upper()), f"[{self.agent_name}] {status_message}")

    def _handle_error(self, state: AdverseMediaState, error: BaseAdverseMediaException, item_id: Optional[str] = None):
        """
        Logs an error and updates the AdverseMediaState with the structured error object.
        Increments failed runs/items.
        Now takes a BaseAdverseMediaException object directly.
        """
        self.logger.error(f"[{self.agent_name}] Error: {error.message} (Type: {error.__class__.__name__})")
        if hasattr(state, 'add_error'):
            # Pass the full exception object for richer state logging
            state.add_error(self.agent_name, error, item_id=item_id)
        self.metrics.failed_runs += 1
        if item_id:
            self.metrics.items_failed += 1
        self.metrics.add_error(error.message) # Keep old behavior for metrics' simple error list

    def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Executes a function with exponential backoff and retries.
        Uses the max_retries and timeout from self.config.

        Args:
            func: The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            The result of the function execution.

        Raises:
            MaxRetriesExceededError: If the function fails after all retries.
            AgentExecutionError: For other unexpected errors during execution.
        """
        self.logger.debug(f"=== RETRY DEBUG: Calling function {func.__name__} with {len(args)} args ===")
        retries = 0
        base_delay = 1.0 # seconds
        max_delay = self.config.timeout # Cap individual delay at overall timeout

        while retries < self.config.max_retries:
            try:
                self.logger.debug(f"Attempt {retries + 1}/{self.config.max_retries} for {func.__name__}...")
                result = func(*args, **kwargs)
                self.logger.debug(f"Successfully executed {func.__name__} on attempt {retries + 1}.")
                return result
            except Exception as e:
                self.logger.warning(
                    f"[{self.agent_name}] Attempt {retries + 1} for {func.__name__} failed: {e}"
                )
                retries += 1
                if retries >= self.config.max_retries:
                    raise MaxRetriesExceededError(
                        f"Max retries ({self.config.max_retries}) exceeded for {func.__name__}.",
                        max_attempts=self.config.max_retries,
                        current_attempts=retries,
                        agent_name=self.agent_name,
                        context={"function": func.__name__}
                    ) from e

                delay = min(base_delay * (2 ** retries) + random.uniform(0, 1), max_delay)
                self.logger.info(f"[{self.agent_name}] Retrying {func.__name__} in {delay:.2f} seconds...")
                time.sleep(delay)
        raise AgentExecutionError(
            f"Function {func.__name__} failed unexpectedly after retry loop.",
            agent_name=self.agent_name,
            context={"function": func.__name__}
        )

    def _safe_process_item(self, func: Callable, state: AdverseMediaState, item: Any, item_id: Optional[str] = None) -> Optional[Any]:
        """
        Processes a single item safely, catching exceptions and updating metrics.
        Errors are logged to the provided AdverseMediaState object.
        Now uses the new structured exception handling.

        Args:
            func: The processing function for the item.
            state: The current AdverseMediaState object to log errors to.
            item: The item to process.
            item_id: An optional identifier for the item for logging/error tracking.

        Returns:
            The result of the processing, or None if an error occurred.
        """
        try:
            result = func(item)
            self.metrics.items_processed += 1
            return result
        except BaseAdverseMediaException as e: # Catch custom exceptions
            self._handle_error(state, e, item_id=item_id)
            self.metrics.items_failed += 1
            return None
        except Exception as e: # Catch any other unexpected exceptions
            error_msg = f"An unexpected error occurred processing item '{item_id if item_id else item}': {e}"
            # Wrap generic exceptions into our AgentExecutionError
            wrapped_error = AgentExecutionError(
                error_msg,
                agent_name=self.agent_name,
                context={"item_id": item_id, "original_error_type": type(e).__name__}
            )
            self._handle_error(state, wrapped_error, item_id=item_id)
            self.metrics.items_failed += 1
            return None

    def _batch_process(self, func: Callable[[List[Any]], List[Any]], state: AdverseMediaState, items: List[Any], batch_size: Optional[int] = None) -> List[Any]:
        """
        Processes items in batches, applying a function to each batch.
        Assumes the function 'func' can handle a list of items and returns a list.
        Errors are logged to the provided AdverseMediaState object.
        Now uses the new structured exception handling.

        Args:
            func: The function to apply to each batch.
            state: The current AdverseMediaState object to log errors to.
            items: The list of items to process.
            batch_size: Optional batch size. Uses config.batch_size if not provided.

        Returns:
            A flattened list of results from all batches.
        """
        if not items:
            return []

        actual_batch_size = batch_size if batch_size is not None else self.config.batch_size
        results = []
        for i in range(0, len(items), actual_batch_size):
            batch = items[i:i + actual_batch_size]
            try:
                batch_results = func(batch)
                results.extend(batch_results)
                self.metrics.items_processed += len(batch)
            except BaseAdverseMediaException as e: # Catch custom exceptions
                error_msg = f"Failed to process batch {i//actual_batch_size + 1} ({len(batch)} items): {e.message}"
                # Update original exception with batch context or create a new one
                if not e.agent_name: # If agent_name not already set by specific exception
                    e.agent_name = self.agent_name
                e.context.update({"batch_index": i//actual_batch_size, "batch_size": len(batch)})
                self._handle_error(state, e, item_id=f"batch_{i//actual_batch_size + 1}")
                self.metrics.items_failed += len(batch) # Count all items in batch as failed
            except Exception as e: # Catch any other unexpected exceptions
                error_msg = f"An unexpected error occurred processing batch {i//actual_batch_size + 1} ({len(batch)} items): {e}"
                wrapped_error = AgentExecutionError(
                    error_msg,
                    agent_name=self.agent_name,
                    context={"batch_index": i//actual_batch_size, "batch_size": len(batch), "original_error_type": type(e).__name__}
                )
                self._handle_error(state, wrapped_error, item_id=f"batch_{i//actual_batch_size + 1}")
                self.metrics.items_failed += len(batch) # Count all items in batch as failed
        return results

    def _validate_input_state(self, state: AdverseMediaState):
        """
        Performs basic validation on the incoming AdverseMediaState.
        Can be extended by subclasses for specific input requirements.
        """
        if not isinstance(state, AdverseMediaState):
            raise AgentExecutionError( # Using AgentExecutionError for this fundamental check
                f"Input must be an AdverseMediaState object, got {type(state)}.",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL, # Critical because input is fundamental
                context={"expected_type": "AdverseMediaState", "received_type": type(state).__name__}
            )
        self.logger.debug(f"Input state validated for {self.agent_name}.")

    def get_metrics(self) -> AgentMetrics:
        """Returns the current metrics for the agent."""
        self.metrics.update_average_processing_time()
        return self.metrics

    def reset_metrics(self):
        """Resets the agent's performance metrics."""
        self.metrics.reset()
        self.logger.info(f"Metrics for {self.agent_name} reset.")

    def __enter__(self):
        """Context manager entry: records start time for metrics."""
        self.start_time = time.time()
        self.metrics.total_runs += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: records end time and updates metrics."""
        end_time = time.time()
        start_time = getattr(self, 'start_time', end_time)  # Safety check for missing start_time
        elapsed_time = end_time - start_time
        self.metrics.total_processing_time += elapsed_time

        if exc_type is None:
            self.metrics.successful_runs += 1
            self.logger.info(f"[{self.agent_name}] Run completed successfully in {elapsed_time:.2f} seconds.")
        else:
            # If an exception occurred, log it via _handle_error
            self.metrics.failed_runs += 1
            error_message = f"Run failed due to {exc_type.__name__}: {exc_val}"
            # Determine if it's already a BaseAdverseMediaException or needs wrapping
            if isinstance(exc_val, BaseAdverseMediaException):
                handled_error = exc_val
                if not handled_error.agent_name: # Ensure agent_name is set
                    handled_error.agent_name = self.agent_name
            else:
                handled_error = AgentExecutionError(
                    error_message,
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.CRITICAL, # Treat unhandled top-level errors as critical
                    context={"original_exception_type": exc_type.__name__, "traceback_info": str(exc_tb)}
                )
            
            current_state = getattr(self, '_current_state', None)
            if current_state is not None:
                self._handle_error(current_state, handled_error)  # FIXED: Use existing method
            else:
                # Log to metrics only if no state is available
                self.metrics.add_error(handled_error.message)
                self.logger.error(f"[{self.agent_name}] {error_message} (no state available for error tracking)")

class BaseDSPyAgent(BaseAgent):
    def __init__(self, config: BaseAgentConfig, logger: logging.Logger, llm_service: Optional['LLMService'] = None, **kwargs):
        super().__init__(config, logger, **kwargs)
        if not DSPY_AVAILABLE:
            raise AgentInitializationError(
                "DSPy library not found. Please install it to use BaseDSPyAgent.",
                component="dspy_library",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL
            )
        
        # Use injected LLMService or create one from config
        if llm_service is not None:
            self.llm_service = llm_service
        else:
            from services.llm_service import create_llm_service_from_config
            self.llm_service = create_llm_service_from_config(config)
        
        # For backward compatibility, set self.lm to primary model
        # Also validate that primary model exists
        try:
            self.lm = self.llm_service.get_model('primary')
        except ValueError as e:
            raise AgentInitializationError(
                f"Primary model not available in LLMService: {e}",
                component="llm_service",
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL
            ) from e
        
        self.logger.debug(f"BaseDSPyAgent '{self.agent_name}' initialized with LLMService.")

    def _execute_with_llm(self, dspy_program: Any, model_name: str = "primary", *args, **kwargs) -> Any:
        """
        Executes a DSPy program with error handling and retry logic using the specified model.

        Args:
            dspy_program: An instance of a DSPy Module (e.g., dspy.Predict, dspy.ChainOfThought).
            model_name: Name of the model to use ('primary', 'secondary', 'arbitration', or custom name).
            *args: Arguments for the dspy_program's forward method.
            **kwargs: Keyword arguments for the dspy_program's forward method.

        Returns:
            The result of the DSPy program execution.

        Raises:
            DSPyExecutionError: If the DSPy program fails after retries or due to other issues.
        """
        try:
            # Validate model exists before attempting to use it
            available_models = self.llm_service.list_models()
            if model_name not in available_models:
                raise DSPyExecutionError(
                    f"Model '{model_name}' not available in LLMService. Available models: {list(available_models.keys())}",
                    module_name=dspy_program.__class__.__name__,
                    agent_name=self.agent_name,
                    severity=ErrorSeverity.HIGH,
                    context={"model_name": model_name, "available_models": list(available_models.keys())}
                )
            
            with self.llm_service.use_model(model_name):
                result = self._retry_with_backoff(dspy_program.forward, *args, **kwargs)
            self.metrics.api_calls_made += 1
            return result
            
        except MaxRetriesExceededError as e:
            raise DSPyExecutionError(
                f"DSPy program '{dspy_program.__class__.__name__}' failed after max retries: {e.message}",
                module_name=dspy_program.__class__.__name__,
                agent_name=self.agent_name,
                severity=ErrorSeverity.HIGH,
                context={"original_error": e.to_dict(), "model_name": model_name}
            ) from e
        except DSPyExecutionError:
            # Re-raise DSPyExecutionError without wrapping
            raise
        except ValueError as e:
            # Handle model not found errors from LLMService
            raise DSPyExecutionError(
                f"Model '{model_name}' not available in LLMService: {e}",
                module_name=dspy_program.__class__.__name__,
                agent_name=self.agent_name,
                severity=ErrorSeverity.HIGH,
                context={"model_name": model_name, "available_models": list(self.llm_service.list_models().keys())}
            ) from e
        except BaseAdverseMediaException as e:
            e.agent_name = self.agent_name
            raise e
        except Exception as e:
            raise DSPyExecutionError(
                f"An unexpected error occurred during DSPy program '{dspy_program.__class__.__name__}' execution: {e}",
                module_name=dspy_program.__class__.__name__,
                agent_name=self.agent_name,
                severity=ErrorSeverity.CRITICAL,
                context={"original_error_type": type(e).__name__, "model_name": model_name}
            ) from e

    def _compare_llm_results(self, result1: Any, result2: Any, prompt: str) -> bool:
        """
        Uses the arbitration model to compare two results and determine consistency.

        Args:
            result1: The first result to compare.
            result2: The second result to compare.
            prompt: The original prompt or context that led to the results.

        Returns:
            True if results are consistent, False otherwise.
        """
        try:
            # Check if arbitration model is available
            available_models = self.llm_service.list_models()
            if 'arbitration' not in available_models:
                self.logger.warning("Arbitration model not available in LLMService. Cannot compare results reliably.")
                return False

            class ArbitrationSignature(dspy.Signature):
                """
                Given an original prompt and two different results, determine if the results are consistent.
                Output 'Yes' if consistent, 'No' if inconsistent. Also provide a brief reason.
                """
                prompt: str = dspy.InputField(desc="The original context or prompt that generated the results.")
                result1: str = dspy.InputField(desc="The first result from an LLM.")
                result2: str = dspy.InputField(desc="The second result from an LLM.")
                consistency: str = dspy.OutputField(desc="Is it 'Yes' or 'No'?")
                reason: str = dspy.OutputField(desc="Brief explanation for consistency assessment.")

            arbitration_program = dspy.Predict(ArbitrationSignature)

            with self.llm_service.use_model('arbitration'):
                response = self._retry_with_backoff(
                    arbitration_program.forward,
                    prompt=prompt,
                    result1=str(result1),
                    result2=str(result2)
                )
            
            self.logger.debug(f"Arbitration response: Consistency: {response.consistency}, Reason: {response.reason}")
            return response.consistency.strip().lower() == 'yes'
            
        except MaxRetriesExceededError as e:
            self.logger.error(f"Arbitration LLM failed after max retries: {e.message}")
            return False
        except ValueError as e:
            self.logger.error(f"Arbitration model not available: {e}")
            return False
        except BaseAdverseMediaException as e:
            e.agent_name = self.agent_name
            self.logger.error(f"Error during LLM result arbitration: {e.message}")
            return False
        except Exception as e:
            self.logger.error(f"An unexpected error occurred during LLM result arbitration: {e}")
            return False

    def get_available_models(self) -> Dict[str, str]:
        """
        Returns a dictionary of available models from the LLMService.
        
        Returns:
            Dictionary mapping model names to their string representations.
        """
        return self.llm_service.list_models()

    def switch_primary_model(self, model_name: str) -> None:
        """
        Switches the primary model for this agent.
        
        Args:
            model_name: Name of the model to set as primary.
            
        Raises:
            DSPyExecutionError: If the model is not available.
        """
        try:
            self.lm = self.llm_service.get_model(model_name)
            self.logger.info(f"Primary model switched to '{model_name}' for agent '{self.agent_name}'")
        except ValueError as e:
            raise DSPyExecutionError(
                f"Cannot switch to model '{model_name}': {e}",
                module_name="model_switch",
                agent_name=self.agent_name,
                severity=ErrorSeverity.HIGH,
                context={"requested_model": model_name, "available_models": list(self.llm_service.list_models().keys())}
            ) from e