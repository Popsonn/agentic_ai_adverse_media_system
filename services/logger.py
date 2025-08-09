# services/logger.py
import logging
from datetime import datetime
from typing import Optional, Any

class LoggerService:
    """Centralized logging service for agents"""
    
    def __init__(self, name: str = __name__, level: int = logging.INFO):
        self.name = name
        self.level = level
        self._logger = None
    
    def get_logger(self) -> logging.Logger:
        """Get or create a configured logger instance"""
        if self._logger is None:
            self._logger = logging.getLogger(self.name)
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    "[%(asctime)s] %(levelname)s - %(name)s - %(message)s", 
                    "%Y-%m-%d %H:%M:%S"
                )
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(self.level)
        return self._logger
    
    def log_debug(self, message: str, agent_name: Optional[str] = None):
        """Log debug messages"""
        if agent_name:
            message = f"[{agent_name}] {message}"
        self.get_logger().debug(message)
    
    def log_info(self, message: str, agent_name: Optional[str] = None):
        """Log info messages"""
        if agent_name:
            message = f"[{agent_name}] {message}"
        self.get_logger().info(message)
    
    def log_warning(self, message: str, agent_name: Optional[str] = None):
        """Log warning messages"""
        if agent_name:
            message = f"[{agent_name}] {message}"
        self.get_logger().warning(message)
    
    def log_error(self, message: str, agent_name: Optional[str] = None):
        """Log error messages"""
        if agent_name:
            message = f"[{agent_name}] {message}"
        self.get_logger().error(message)
    
    def log_critical(self, message: str, agent_name: Optional[str] = None):
        """Log critical messages"""
        if agent_name:
            message = f"[{agent_name}] {message}"
        self.get_logger().critical(message)
    
    def log_event(self, state: Any, agent: str, message: str):
        """Log event to state (for state-based logging)"""
        if hasattr(state, 'add_log'):
            state.add_log(agent, message)
        else:
            # Fallback to regular logging if state doesn't have add_log
            self.log_info(f"Event: {message}", agent)
    
    def log_state_error(self, state: Any, message: str):
        """Log error to state (for state-based logging)"""
        if hasattr(state, 'add_error'):
            state.add_error(message)
        else:
            # Fallback to regular logging if state doesn't have add_error
            self.log_error(f"State Error: {message}")
    
    @staticmethod
    def create_timestamped_message(message: str, agent_name: Optional[str] = None) -> str:
        """Create a timestamped message for console output"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        agent_prefix = f"[{agent_name}] " if agent_name else ""
        return f"[{timestamp}]: {agent_prefix}{message}"


# Global logger instance for convenience
_default_logger = LoggerService()

# Static methods for backward compatibility (Agent 3 style)
def log_debug(message: str, agent_name: Optional[str] = None):
    """Log debug messages (static method for backward compatibility)"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    agent_prefix = f"[{agent_name}] " if agent_name else ""
    print(f"DEBUG [{timestamp}]: {agent_prefix}{message}")

def log_info(message: str, agent_name: Optional[str] = None):
    """Log info messages (static method for backward compatibility)"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    agent_prefix = f"[{agent_name}] " if agent_name else ""
    print(f"INFO [{timestamp}]: {agent_prefix}{message}")

def log_error(message: str, agent_name: Optional[str] = None, state: Any = None):
    """Log error messages (static method for backward compatibility)"""
    if state is not None:
        # State-based logging (Agent 2 style)
        if hasattr(state, 'add_error'):
            state.add_error(message)
        else:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            agent_prefix = f"[{agent_name}] " if agent_name else ""
            print(f"ERROR [{timestamp}]: {agent_prefix}{message}")
    else:
        # Regular logging (Agent 3 style)
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        agent_prefix = f"[{agent_name}] " if agent_name else ""
        print(f"ERROR [{timestamp}]: {agent_prefix}{message}")

def log_event(state: Any, agent: str, message: str):
    """Log event to state (Agent 2 style for backward compatibility)"""
    if hasattr(state, 'add_log'):
        state.add_log(agent, message)
    else:
        # Fallback to regular logging
        log_info(f"Event: {message}", agent)

def get_logger(name: str = __name__) -> logging.Logger:
    """Get logger instance (Agent 1 style for backward compatibility)"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s - %(message)s", 
            "%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def get_logger_service(name: str = __name__, level: int = logging.INFO) -> LoggerService:
    """Get a configured LoggerService instance"""
    return LoggerService(name, level)