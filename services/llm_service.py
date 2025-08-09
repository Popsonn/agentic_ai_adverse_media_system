# services/llm_service.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
from contextlib import contextmanager
import dspy
import os
from dotenv import load_dotenv

load_dotenv()

class LLMConfigProvider(ABC):
    """Abstract interface for LLM configuration"""
    
    @abstractmethod
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider (groq, openai, together, etc.)"""
        pass
    
    @abstractmethod
    def get_model_name(self, role: str) -> str:
        """Get model name for a role (primary, secondary, arbitration)"""
        pass

class EnvironmentLLMConfig(LLMConfigProvider):
    """Configuration provider that reads from environment variables"""
    
    def __init__(self):
        self._model_defaults = {
            'primary': 'groq/llama-3.3-70b-versatile',
            'secondary': 'together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',  # Updated to TogetherAI
            'arbitration': 'groq/deepseek-r1-distill-llama-70b'
        }
    
    def get_api_key(self, provider: str) -> Optional[str]:
        env_keys = {
            'groq': 'GROQ_API_KEY',
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'together': 'TOGETHER_API_KEY'  # Added TogetherAI
        }
        return os.getenv(env_keys.get(provider.lower()))
    
    def get_model_name(self, role: str) -> str:
        # Try environment first, fall back to defaults
        env_key = f"{role.upper()}_LLM_MODEL"
        return os.getenv(env_key, self._model_defaults.get(role, self._model_defaults['primary']))

class ConfigBasedLLMConfig(LLMConfigProvider):
    """Configuration provider that uses your settings config"""
    
    def __init__(self, config):
        self.config = config
    
    def get_api_key(self, provider: str) -> Optional[str]:
        key_map = {
            'groq': self.config.groq_api_key,
            'openai': self.config.openai_api_key,
            'together': getattr(self.config, 'together_api_key', None)  # Added TogetherAI
        }
        return key_map.get(provider.lower())
    
    def get_model_name(self, role: str) -> str:
        model_map = {
            'primary': self.config.primary_llm_model,
            'secondary': self.config.secondary_llm_model,
            'arbitration': self.config.arbitration_llm_model
        }
        return model_map.get(role) or self.config.primary_llm_model

class LLMService:
    """
    Production-ready LLM service following dependency injection principles.
    Testable, configurable, and maintainable.
    """
    
    def __init__(self, config_provider: LLMConfigProvider):
        self.config_provider = config_provider
        self.models: Dict[str, dspy.LM] = {}
        self.default_name: Optional[str] = None
    
    def register_model(self, name: str, model_name: str, api_key: Optional[str] = None, **kwargs) -> None:
        """Register a model with explicit parameters"""
        if api_key is None:
            provider = self._extract_provider(model_name)
            api_key = self.config_provider.get_api_key(provider)
        
        if not api_key:
            raise ValueError(f"No API key found for provider: {self._extract_provider(model_name)}")
        
        # Handle TogetherAI specific configuration
        if self._extract_provider(model_name) == 'together':
            # Based on Together AI docs: together_ai/model-name format
            actual_model_name = model_name.replace('together/', 'together_ai/')
            # Add the required api_base for Together AI
            self.models[name] = dspy.LM(
                model=actual_model_name, 
                api_key=api_key,
                api_base="https://api.together.xyz/v1",
                **kwargs
            )
        else:
            self.models[name] = dspy.LM(model=model_name, api_key=api_key, **kwargs)
        
        if self.default_name is None:
            self.default_name = name

    def test_models(self) -> Dict[str, bool]:
        """Test all registered models to ensure they work"""
        results = {}
        for name, model in self.models.items():
            try:
                # Simple test query
                with dspy.context(lm=model):
                    response = dspy.Predict("question -> answer")("What is 2+2?")
                results[name] = True
                print(f"✅ {name}: Working")
            except Exception as e:
                results[name] = False
                print(f"❌ {name}: Failed - {str(e)}")
        return results
    
    def register_role(self, role: str, **kwargs) -> None:
        """Register a model by role (primary, secondary, arbitration)"""
        model_name = self.config_provider.get_model_name(role)
        self.register_model(role, model_name, **kwargs)
    
    def setup_standard_models(self) -> None:
        """Setup the standard model roles"""
        registered_models = []
        
        # Always try to register all three models
        for role in ['primary', 'secondary', 'arbitration']:
            try:
                self.register_role(role)
                registered_models.append(role)
                print(f"✅ Registered {role} model: {self.config_provider.get_model_name(role)}")
            except ValueError as e:
                print(f"❌ Could not register {role} model: {e}")
        
        # Set default in order of preference
        if 'primary' in registered_models:
            self.set_global_default('primary')
        elif 'secondary' in registered_models:
            self.set_global_default('secondary')
            print("Warning: Primary model not available, using secondary as default")
        elif 'arbitration' in registered_models:
            self.set_global_default('arbitration')
            print("Warning: Primary/secondary not available, using arbitration as default")
        else:
            raise RuntimeError("No LLM models could be registered! Check your API keys and model configurations.")
        
        print(f"Available models: {list(self.models.keys())}")
        return registered_models
    
    def get_model(self, name: str = "default") -> dspy.LM:
        """Get a registered model"""
        if name == "default":
            if self.default_name is None:
                raise ValueError("No default model set")
            name = self.default_name
        
        if name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        
        return self.models[name]
    
    def set_global_default(self, name: str) -> None:
        """Set global DSPy default model"""
        dspy.configure(lm=self.get_model(name))
        self.default_name = name
    
    @contextmanager
    def use_model(self, name: str):
        """Context manager for temporarily switching models"""
        yield dspy.context(lm=self.get_model(name))
    
    def list_models(self) -> Dict[str, str]:
        """List all registered models"""
        return {name: str(model) for name, model in self.models.items()}
    
    def has_model(self, name: str) -> bool:
        """Check if a model is available"""
        return name in self.models
    
    def get_fallback_model(self, preferred: str, fallback: str) -> str:
        """Get preferred model if available, otherwise fallback"""
        return preferred if self.has_model(preferred) else fallback if self.has_model(fallback) else "default"
    
    def _extract_provider(self, model_name: str) -> str:
        """Extract provider from model name"""
        if '/' in model_name:
            provider = model_name.split('/')[0]
            # Map provider names to standardized names
            provider_map = {
                'together': 'together',
                'groq': 'groq',
                'openai': 'openai',
                'anthropic': 'anthropic'
            }
            return provider_map.get(provider, provider)
        return 'groq'  # Default fallback

# Factory functions for different use cases
def create_llm_service_from_config(config) -> LLMService:
    """Create LLM service using your settings config"""
    config_provider = ConfigBasedLLMConfig(config)
    service = LLMService(config_provider)
    service.setup_standard_models()
    return service

def create_llm_service_from_env() -> LLMService:
    """Create LLM service using environment variables only"""
    config_provider = EnvironmentLLMConfig()
    service = LLMService(config_provider)
    service.setup_standard_models()
    return service

# Utility function for your classification agent
def get_classification_model(llm_service: LLMService) -> str:
    """Get the best available model for classification"""
    if llm_service.has_model('primary'):
        return 'primary'
    elif llm_service.has_model('secondary'):
        return 'secondary'
    else:
        return 'default'

# Usage examples:
# 
# # Production usage with your config
# from config.settings import BaseAgentConfig
# config = BaseAgentConfig()
# llms = create_llm_service_from_config(config)
# 
# # Development/testing usage
# llms = create_llm_service_from_env()
# 
# # Usage with fallback logic
# classification_model = get_classification_model(llms)
# with llms.use_model(classification_model):
#     # Your classification code here
#     pass
# 
# # Check what models are available
# print(llms.list_models())
# 
# # Use fallback logic
# model_to_use = llms.get_fallback_model('primary', 'secondary')
# with llms.use_model(model_to_use):
#     # Your code here
#     pass