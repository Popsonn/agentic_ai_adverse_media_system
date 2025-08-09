# services/ner_service.py
from typing import List, Dict, Any, Optional
from transformers import pipeline
import logging
from dataclasses import dataclass

@dataclass
class NEREntity:
    """Structured representation of a named entity."""
    text: str
    label: str
    confidence: float
    start: int
    end: int

class NERService:
    """Named Entity Recognition service with caching and error handling."""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self._pipeline = None
        self._cache = {}  # Simple in-memory cache
        
        # Initialize pipeline lazily
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize the NER pipeline with error handling."""
        try:
            self._pipeline = pipeline(
                "ner", 
                model=self.model_name, 
                aggregation_strategy="simple"
            )
            self.logger.info(f"NER pipeline initialized with model: {self.model_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize NER model {self.model_name}: {str(e)}")
            raise RuntimeError(f"NER service initialization failed: {str(e)}")

    def extract_entities(self, text: str, min_confidence: float = 0.5) -> List[NEREntity]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text to process
            min_confidence: Minimum confidence threshold for entities
            
        Returns:
            List of NEREntity objects
        """
        if not text or not text.strip():
            return []
            
        # Check cache first
        cache_key = f"{text[:100]}_{min_confidence}"  # Use first 100 chars as key
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            raw_entities = self._pipeline(text)
            entities = []
            
            for entity in raw_entities:
                if entity.get('score', 0) >= min_confidence:
                    ner_entity = NEREntity(
                        text=entity['word'],
                        label=entity['entity_group'],
                        confidence=entity['score'],
                        start=entity.get('start', 0),
                        end=entity.get('end', 0)
                    )
                    entities.append(ner_entity)
            
            # Cache the result
            self._cache[cache_key] = entities
            
            # Simple cache size management
            if len(self._cache) > 100:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                
            return entities
            
        except Exception as e:
            self.logger.warning(f"NER extraction failed for text: {text[:50]}... Error: {str(e)}")
            return []

    def extract_entities_by_type(self, text: str, entity_types: List[str] = None) -> Dict[str, List[NEREntity]]:
        """
        Extract entities grouped by type.
        
        Args:
            text: Input text
            entity_types: List of entity types to extract (PER, ORG, LOC, MISC)
            
        Returns:
            Dictionary mapping entity types to lists of entities
        """
        all_entities = self.extract_entities(text)
        
        if entity_types is None:
            entity_types = ['PER', 'ORG', 'LOC', 'MISC']
        
        grouped = {entity_type: [] for entity_type in entity_types}
        
        for entity in all_entities:
            if entity.label in grouped:
                grouped[entity.label].append(entity)
        
        return grouped

    def get_persons(self, text: str) -> List[str]:
        """Extract person names from text."""
        entities = self.extract_entities_by_type(text, ['PER'])
        return [entity.text for entity in entities['PER']]

    def get_organizations(self, text: str) -> List[str]:
        """Extract organization names from text."""
        entities = self.extract_entities_by_type(text, ['ORG'])
        return [entity.text for entity in entities['ORG']]

    def get_locations(self, text: str) -> List[str]:
        """Extract location names from text."""
        entities = self.extract_entities_by_type(text, ['LOC'])
        return [entity.text for entity in entities['LOC']]

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
        self.logger.info("NER cache cleared")

    def get_cache_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)