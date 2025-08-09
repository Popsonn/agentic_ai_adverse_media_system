# services/embedding.py
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Optional

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def get_model(self) -> SentenceTransformer:
        """Returns the sentence transformer model"""
        return self.model
    
    def encode(self, texts: List[str], convert_to_numpy: bool = False) -> np.ndarray:
        """
        Encode texts into embeddings
        
        Args:
            texts: List of texts to encode
            convert_to_numpy: Whether to convert to numpy array
            
        Returns:
            Encoded embeddings
        """
        return self.model.encode(texts, convert_to_numpy=convert_to_numpy)
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            embeddings = self.model.encode([text1, text2])
            return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        except:
            return 0.0
    
    def max_similarity_with_list(self, base_text: str, items: List[str], 
                                text_limit: int = 300) -> float:
        """
        Find maximum similarity between base text and a list of items
        
        Args:
            base_text: Text to compare against
            items: List of texts to compare with
            text_limit: Character limit for base text (default: 300)
            
        Returns:
            Maximum similarity score
        """
        try:
            if not items:
                return 0.0
            
            # Truncate base text if specified
            truncated_base = base_text[:text_limit] if text_limit else base_text
            base_embedding = self.model.encode([truncated_base])
            item_embeddings = self.model.encode(items)
            similarities = cosine_similarity(base_embedding, item_embeddings)[0]
            return float(max(similarities))
        except:
            return 0.0
    
    def deduplicate_by_similarity(self, texts: List[str], threshold: float) -> List[int]:
        """
        Deduplicate texts by similarity, returning indices of unique items
        
        Args:
            texts: List of texts to deduplicate
            threshold: Similarity threshold above which items are considered duplicates
            
        Returns:
            List of indices of unique items
        """
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            sim_matrix = cosine_similarity(embeddings)
            
            selected, seen = [], set()
            for i in range(len(texts)):
                if i in seen:
                    continue
                selected.append(i)
                seen.add(i)
                for j in range(i + 1, len(texts)):
                    if sim_matrix[i][j] > threshold:
                        seen.add(j)
            return selected
        except:
            return list(range(len(texts)))  # Return all indices if error
    
    def batch_similarity(self, texts1: List[str], texts2: List[str]) -> List[List[float]]:
        """
        Calculate similarity matrix between two lists of texts
        
        Args:
            texts1: First list of texts
            texts2: Second list of texts
            
        Returns:
            2D list of similarity scores
        """
        try:
            embeddings1 = self.model.encode(texts1)
            embeddings2 = self.model.encode(texts2)
            sim_matrix = cosine_similarity(embeddings1, embeddings2)
            return sim_matrix.tolist()
        except:
            return [[0.0] * len(texts2) for _ in texts1]
        
        # Add this method to your existing EmbeddingService class
    # Keep all other methods exactly as they are!

    def max_similarity_with_list_robust(self, base_text: str, items: List[str], 
                                    text_limit: int = 300) -> tuple[float, bool]:
        """
        Robust version of max_similarity_with_list that returns success status.
        
        Args:
            base_text: Text to compare against
            items: List of texts to compare with  
            text_limit: Character limit for base text (default: 300)
            
        Returns:
            Tuple of (similarity_score, calculation_succeeded)
            - If succeeded=True: similarity_score is reliable
            - If succeeded=False: similarity_score is 0.0 and should be ignored
        """
        try:
            if not items or not base_text.strip():
                return 0.0, True  # Valid result: no items to compare or empty text
            
            # Smart truncation at sentence boundary when possible
            if text_limit and len(base_text) > text_limit:
                truncated = base_text[:text_limit]
                # Try to end at sentence boundary
                last_period = truncated.rfind('.')
                last_space = truncated.rfind(' ')
                if last_period > text_limit * 0.7:  # If period is in last 30%
                    truncated = truncated[:last_period + 1]
                elif last_space > text_limit * 0.8:  # If space is in last 20%
                    truncated = truncated[:last_space]
            else:
                truncated = base_text
            
            base_embedding = self.model.encode([truncated])
            item_embeddings = self.model.encode(items)
            similarities = cosine_similarity(base_embedding, item_embeddings)[0]
            max_sim = float(max(similarities))
            
            # Clamp to valid range
            return max(0.0, min(1.0, max_sim)), True
            
        except Exception as e:
            # Log the error but don't break
            if hasattr(self, 'logger'):
                self.logger.warning(f"Embedding similarity calculation failed: {e}")
            return 0.0, False  # Explicitly indicate failure

    # Optional: Add smart truncation as separate utility
    def smart_truncate_text(self, text: str, limit: int = 300) -> str:
        """
        Truncate text intelligently at sentence or word boundaries.
        
        Args:
            text: Text to truncate
            limit: Character limit
            
        Returns:
            Truncated text
        """
        if len(text) <= limit:
            return text
        
        truncated = text[:limit]
        
        # Try to end at sentence boundary first
        last_period = truncated.rfind('.')
        if last_period > limit * 0.7:  # If period is in last 30%
            return truncated[:last_period + 1]
        
        # Fall back to word boundary
        last_space = truncated.rfind(' ')
        if last_space > limit * 0.8:  # If space is in last 20%
            return truncated[:last_space]
        
        # Last resort: character truncation
        return truncated
        
# Convenience functions for backward compatibility
def get_sentence_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Legacy function wrapper for backward compatibility"""
    service = EmbeddingService(model_name)
    return service.get_model()

def deduplicate_articles_by_similarity(titles: List[str], model: SentenceTransformer, 
                                     threshold: float) -> List[int]:
    """Legacy function wrapper for backward compatibility"""
    # Create service from existing model
    service = EmbeddingService()
    service.model = model  # Use the provided model
    return service.deduplicate_by_similarity(titles, threshold)