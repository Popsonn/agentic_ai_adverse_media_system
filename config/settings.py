# adverse_media_system/config/settings.py
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import os
from config.constants import CREDIBLE_SOURCES, EXCLUDED_DOMAINS
from models.enums import ResolutionMethod


@dataclass
class BaseAgentConfig:
    """Base configuration for all agents and system-wide parameters."""
    # --- API Keys ---
    tavily_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    brave_api_key: Optional[str] = field(default_factory=lambda: os.getenv("BRAVE_SEARCH_API_KEY"))
    groq_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    together_api_key: Optional[str] = field(default_factory=lambda: os.getenv("TOGETHER_API_KEY"))  # Added TogetherAI

    # --- LLM Model Names (Standardized) ---
    primary_llm_model: str = "groq/llama-3.3-70b-versatile"        # Main workhorse model
    #secondary_llm_model: Optional[str] = "groq/llama-3.1-8b-instant"  # Faster model for simple tasks
    secondary_llm_model: Optional[str] = "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"  # Changed to TogetherAI
    arbitration_llm_model: Optional[str] = "groq/deepseek-r1-distill-llama-70b"  # Conflict resolution
    embedding_model_name: str = "all-MiniLM-L6-v2"
    ner_model: Optional[str] = "dbmdz/bert-large-cased-finetuned-conll03-english"

    # --- Model Role Mapping (for clarity) ---
    @property
    def model_roles(self) -> Dict[str, str]:
        """Map logical roles to actual model names"""
        return {
            "entity_disambiguation": self.primary_llm_model,
            "search_strategy": self.primary_llm_model,
            "classification": self.primary_llm_model,
            "conflict_resolution": self.arbitration_llm_model or self.primary_llm_model,
            "ner": self.primary_llm_model,  # If using LLM for NER instead of transformer
            "quality_check": self.secondary_llm_model or self.primary_llm_model,
        }

    # --- Global System Parameters ---
    batch_size: int = 5
    data_storage_path: str = "data/"
    
    # --- Common Agent Parameters ---
    confidence_threshold: float = 0.65
    max_retries: int = 3
    timeout: int = 60
    logging_level: str = "INFO"
    enable_metrics: bool = True
    debug_mode: bool = False
    
    # --- Human Review Thresholds ---
    human_review_confidence_threshold: float = 0.5

    def __post_init__(self):
            """Perform validation after initialization."""
            # Validate that we have at least one LLM API key
            if not self.groq_api_key and not self.openai_api_key and not self.together_api_key:
                if self.debug_mode:
                    print("WARNING: No LLM API key (Groq/OpenAI/Together) set.")
            
            # Validate logging level
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if self.logging_level.upper() not in valid_levels:
                raise ValueError(f"Invalid logging_level: {self.logging_level}. Must be one of {valid_levels}")


# --- Agent-Specific Configuration (unchanged) ---
#@dataclass
#class EntityDisambiguationConfig(BaseAgentConfig):
    #disambiguation_confidence_threshold: float = 0.7
    #disambiguation_review_threshold: float = 0.85

@dataclass
class EntityDisambiguationAgentConfig(BaseAgentConfig):
    """Configuration specific to the EntityDisambiguationAgent."""
    confidence_threshold: float = 0.65
    review_threshold: float = 0.80
    max_candidates: int = 5
    max_context_terms: int = 3
    # No embedding model name config needed if we always use the EmbeddingService default
    # Add NER model name config if you want to allow overriding the default

#@dataclass
#class SearchStrategyConfig(BaseAgentConfig):
    #search_similarity_threshold: float = 0.85
    #search_quality_threshold: float = 0.6
    #max_search_attempts: int = 3
    #days_back: int = 3650
    #max_articles_per_search: int = 7
    #max_filtered_articles: int = 10

@dataclass
class SearchStrategyConfig(BaseAgentConfig):
    """Configuration specific to SearchStrategyAgent"""
    similarity_threshold: float = 0.85
    max_search_attempts: int = 3
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    search_quality_threshold: float = 0.5
    max_articles: int = 10
    max_filtered_articles: int = 5
    days_back: int = 730
    include_domains: List[str] = field(default_factory=lambda: CREDIBLE_SOURCES.copy())
    exclude_domains: List[str] = field(default_factory=lambda: EXCLUDED_DOMAINS.copy())
    
    def __post_init__(self):
        super().__post_init__()
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.max_search_attempts < 1:
            raise ValueError("max_search_attempts must be at least 1")

# ------------------- MAIN AGENT CLASS ---------

@dataclass
class ClassificationAgentConfig(BaseAgentConfig):
    #classification_min_confidence: float = 0.7
    max_classification_errors: int = 3
    skip_articles_without_content: bool = True
    require_secondary_llm: bool = True # This flag still belongs here if specific to this agent
    confidence_threshold: float = 0.7
    primary_llm_role: str = "primary"
    secondary_llm_role: str = "secondary"


#@dataclass
#class ConflictResolutionConfig(BaseAgentConfig):
    #conflict_high_confidence_threshold: float = 0.85
    #conflict_conservative_gap: float = 0.2
    #conflict_arbitration_min_confidence: float = 0.75
    #conflict_min_absolute_confidence: float = 0.6
    #enable_reasoning_arbitration: bool = True
    #enable_external_search_context: bool = True
    #escalate_critical_conflicts: bool = True

@dataclass
class ConflictResolutionConfig(BaseAgentConfig):
    """
    Configuration specific to the Conflict Resolution Agent.
    """
    high_confidence_threshold: float = 0.85
    confidence_gap_for_conservative: float = 0.2
    llm_arbitration_min_confidence: float = 0.75
    minimal_absolute_confidence_for_rule: float = 0.6
    escalate_critical: bool = True
    enable_reasoning: bool = True
    enable_external_search: bool = True
    brave_search_api_key: Optional[str] = None

    @property
    def resolution_method_enum(self) -> type:
        """Return the ResolutionMethod enum for use in conflict resolution."""
        return ResolutionMethod
