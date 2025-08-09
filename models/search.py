# adverse_media_system/models/search.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

# --- Enums ---
class SearchStrategy(Enum):
    """
    Defines different strategies for conducting web searches.
    Matches the enum used in SearchStrategyAgent.
    """
    BROAD = "broad"
    TARGETED = "targeted"
    DEEP_DIVE = "deep_dive" # Corrected from deep_dIVE as per your code
    ALTERNATIVE = "alternative"

# --- Data Models ---
@dataclass
class SearchQuality:
    """
    Represents the calculated quality metrics for a set of search results.
    Used by the SearchStrategyAgent to evaluate search performance.
    """
    article_count: int
    relevance_score: float
    source_credibility: float
    time_coverage: float
    entity_prominence: float
    keyword_match_score: float

    @property
    def overall_quality(self) -> float:
        """Calculates an aggregated overall quality score based on weighted metrics."""
        if self.article_count == 0:
            return 0.0

        weights = {
            'relevance': 0.25,
            'credibility': 0.25,
            'coverage': 0.15,
            'prominence': 0.15,
            'keyword_match': 0.1,
            'quantity': 0.1
        }
        quantity_score = min(self.article_count / 10, 1.0) # Cap quantity score at 1.0

        return (
            weights['relevance'] * self.relevance_score +
            weights['credibility'] * self.source_credibility +
            weights['coverage'] * self.time_coverage +
            weights['prominence'] * self.entity_prominence +
            weights['keyword_match'] * self.keyword_match_score +
            weights['quantity'] * quantity_score
        )


@dataclass
class SearchResult:
    """
    Represents a single item returned from a web search engine.
    This dataclass is named 'SearchResult' to align with its usage in your SearchStrategyAgent.
    """
    title: str
    url: str
    content: str # Full content of the article (as extracted by Tavily extract API)
    published_date: Optional[str] # Date article was published
    source_domain: str # The domain name of the source (e.g., "reuters.com")
    search_strategy_used: str # The actual query string or strategy name used to get this result item (as per your code)
    snippet: Optional[str] = None # A brief summary of the search result content (often from initial search API)
    relevance_score: Optional[float] = None # Relevance score from the search engine
    metadata: Dict[str, Any] = field(default_factory=dict) # Flexible field for any other metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "published_date": self.published_date,
            "source_domain": self.source_domain,
            "search_strategy_used": self.search_strategy_used,
            "snippet": self.snippet,
            "relevance_score": self.relevance_score,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SearchResult':
        """Create SearchResult instance from dictionary"""
        return cls(
            title=data.get('title', ''),
            url=data.get('url', ''),
            content=data.get('content', ''),
            published_date=data.get('published_date'),
            source_domain=data.get('source_domain', ''),
            search_strategy_used=data.get('search_strategy_used', ''),
            snippet=data.get('snippet'),
            relevance_score=data.get('relevance_score'),
            metadata=data.get('metadata', {})
        )
