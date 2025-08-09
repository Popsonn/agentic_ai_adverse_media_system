# adverse_media_system/utils/similarity.py

from sklearn.metrics.pairwise import cosine_similarity
from services.embedding_service import embedding_service

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """Calculates cosine similarity between two texts."""
    try:
        embeddings = embedding_service.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except Exception:
        return 0.0