# adverse_media_system/services/tavily_client.py

# services/tavily_client.py
from tavily import TavilyClient
from typing import List, Dict, Optional

class TavilyService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = TavilyClient(api_key=api_key)
    
    def get_client(self) -> TavilyClient:
        """Returns the raw Tavily client for direct usage"""
        return self.client
    
    def search(self, query: str, max_results: int = 10, days: Optional[int] = None,
            include_domains: Optional[List[str]] = None, 
            exclude_domains: Optional[List[str]] = None,
            topic: str = "news", search_depth: str = "advanced",
            include_raw_content: bool = False,  # ← ADD THIS PARAMETER
            include_answer: bool = False) -> List[Dict]:  # ← ADD THIS TOO
        """
        Perform a Tavily search with configurable parameters
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            days: Number of days to search back
            include_domains: List of domains to include in search
            exclude_domains: List of domains to exclude from search
            topic: Search topic (default: "news")
            search_depth: Search depth (default: "advanced")
            include_raw_content: Whether to include raw content (default: False)
            include_answer: Whether to include answer (default: False)
            
        Returns:
            List of search results
        """
        search_params = {
            "query": query,
            "topic": topic,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_raw_content": include_raw_content,  # ← USE THE PARAMETER
        }
        
        if include_answer:  # ← ADD THIS
            search_params["include_answer"] = include_answer
        
        if days is not None:
            search_params["days"] = days
        if include_domains is not None:
            search_params["include_domains"] = include_domains
        if exclude_domains is not None:
            search_params["exclude_domains"] = exclude_domains
            
        return self.client.search(**search_params).get("results", [])
    
        # Fixed extract method for TavilyService
    def extract(self, urls: List[str], extract_depth: str = "advanced", 
                format: str = "markdown", include_images: bool = False) -> Dict[str, str]:
        """
        Extract content from URLs using Tavily
        
        Args:
            urls: List of URLs to extract content from
            extract_depth: Extraction depth ("basic" or "advanced")
            format: Output format ("markdown" or "text")
            include_images: Whether to include images in response
            
        Returns:
            Dictionary mapping URLs to their extracted content
        """
        # Call with proper parameters matching official API
        response = self.client.extract(
            urls=urls,
            extract_depth=extract_depth,
            format=format,
            include_images=include_images
        )
        
        # Handle the response structure from official docs
        results = response.get("results", [])
        
        # Map URLs to their raw_content (confirmed field name)
        url_to_content = {}
        for result in results:
            url = result.get("url", "")
            raw_content = result.get("raw_content", "")
            if url and raw_content:
                url_to_content[url] = raw_content
        
        return url_to_content


# Convenience functions for backward compatibility
def tavily_search(query: str, api_key: str, max_results: int, days: int,
                  include_domains: List[str], exclude_domains: List[str]) -> List[Dict]:
    """Legacy function wrapper for backward compatibility"""
    service = TavilyService(api_key)
    return service.search(query, max_results, days, include_domains, exclude_domains)

def tavily_extract(urls: List[str], api_key: str) -> Dict[str, str]:
    """Legacy function wrapper for backward compatibility"""
    service = TavilyService(api_key)
    return service.extract(urls)

def get_tavily_client(api_key: str) -> TavilyClient:
    """Legacy function wrapper for backward compatibility"""
    service = TavilyService(api_key)
    return service.get_client()