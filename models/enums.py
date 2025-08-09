from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum

class ConflictSeverity(Enum):
    """
    Categorizes the severity of a classification conflict.
    Directly from your HybridConflictResolutionAgent.
    """
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class ResolutionMethod(Enum):
    """
    Describes the method used to resolve a classification conflict or determine final classification.
    Directly from your HybridConflictResolutionAgent.
    """
    NO_CONFLICT = "no_conflict"
    CONFIDENCE_BASED = "confidence_based"
    REASONING_ARBITRATION = "reasoning_arbitration"
    CONSERVATIVE_DEFAULT = "conservative_default"
    EXTERNAL_SEARCH_CONTEXT = "external_search_context"
    HUMAN_REVIEW_REQUIRED = "human_review_required" # Changed to match your agent's enum value
    LOW_CONFIDENCE_AGREEMENT = "low_confidence_agreement"
    EXTERNAL_VALIDATION = "external_validation"
    EXTERNAL_CONTEXT_ANALYSIS = "external_context_analysis"

