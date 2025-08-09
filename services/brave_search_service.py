import os
import requests
from typing import List, Optional
import logging
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import time
from models.search import SearchResult
from core.exceptions import SearchError, ErrorSeverity

class BraveSearchService:
    """Enhanced service for performing web searches using Brave Search API with content extraction"""
    
    def __init__(self, logger: Optional[logging.Logger] = None, extract_content: bool = True):
        self.api_key = os.getenv("BRAVE_SEARCH_API_KEY")
        if not self.api_key:
            raise ValueError("BRAVE_SEARCH_API_KEY environment variable not set.")
        self.endpoint = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        # Headers for content extraction requests
        self.content_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        self.logger = logger
        self.extract_content = extract_content
        # Rate limiting for content extraction
        self.request_delay = 1.0  # seconds between requests
        self.max_content_length = 50000  # characters limit for content

    def search(self, query: str, count: int = 5) -> List[SearchResult]:
        """Perform web search and return SearchResult objects"""
        if not query:
            if self.logger:
                self.logger.warning("Empty search query provided. Skipping search.")
            return []

        params = {
            "q": query,
            "count": count,
            "freshness": "pw",
            "search_lang": "en",
            "country": "US",
            "safesearch": "moderate"
        }

        try:
            if self.logger:
                self.logger.debug(f"Performing Brave Search for query: '{query}'")
            response = requests.get(self.endpoint, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            if 'web' in data and 'results' in data['web']:
                for item in data['web']['results']:
                    # Extract content if enabled
                    content = ""
                    if self.extract_content:
                        content = self._extract_content(item.get("url", ""))
                        time.sleep(self.request_delay)  # Rate limiting
                    
                    results.append(SearchResult(
                        title=item.get("title", "No Title"),
                        url=item.get("url", "#"),
                        content=content,  # Now populated with extracted content
                        published_date=item.get("published_date"),
                        source_domain=self._extract_domain(item.get("url", "")),
                        search_strategy_used=query,
                        snippet=item.get("snippet", "No Snippet"),
                        relevance_score=None,
                        metadata={
                            "brave_source": item.get("source", "Brave Search"),
                            "content_extracted": bool(content),
                            "content_length": len(content) if content else 0,
                            "raw_item": item
                        }
                    ))
            return results
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error during Brave Search API call for '{query}': {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise SearchError(
                error_msg,
                query=query,
                severity=ErrorSeverity.MEDIUM,
                context={"api_endpoint": self.endpoint}
            ) from e
        except ValueError as e:
            error_msg = f"JSON decoding error from Brave Search API for '{query}': {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise SearchError(
                error_msg,
                query=query,
                severity=ErrorSeverity.MEDIUM,
                context={"api_endpoint": self.endpoint}
            ) from e
        except Exception as e:
            error_msg = f"Unexpected error in BraveSearchService for '{query}': {e}"
            if self.logger:
                self.logger.error(error_msg)
            raise SearchError(
                error_msg,
                query=query,
                severity=ErrorSeverity.HIGH,
                context={"api_endpoint": self.endpoint}
            ) from e

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return "unknown"

    def _extract_content(self, url: str) -> str:
        """
        Extract main content from a webpage URL.
        Uses multiple strategies to get clean, readable text.
        """
        if not url or url == "#":
            return ""
        
        try:
            if self.logger:
                self.logger.debug(f"Extracting content from: {url}")
            
            response = requests.get(
                url, 
                headers=self.content_headers, 
                timeout=10,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            self._remove_unwanted_elements(soup)
            
            # Try multiple extraction strategies
            content = self._extract_main_content(soup)
            
            # Clean and truncate content
            content = self._clean_content(content)
            
            if self.logger:
                self.logger.debug(f"Extracted {len(content)} characters from {url}")
            
            return content[:self.max_content_length] if content else ""
            
        except requests.exceptions.RequestException as e:
            if self.logger:
                self.logger.warning(f"Failed to fetch content from {url}: {e}")
            return ""
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to extract content from {url}: {e}")
            return ""

    def _remove_unwanted_elements(self, soup: BeautifulSoup) -> None:
        """Remove unwanted HTML elements that don't contain main content"""
        unwanted_tags = [
            'script', 'style', 'nav', 'header', 'footer', 'aside', 
            'menu', 'form', 'button', 'input', 'select', 'textarea',
            'iframe', 'embed', 'object', 'applet', 'canvas', 'audio', 'video'
        ]
        
        unwanted_classes = [
            'advertisement', 'ads', 'ad', 'sidebar', 'navigation', 
            'nav', 'menu', 'footer', 'header', 'comments', 'comment',
            'social', 'share', 'related', 'popup', 'modal', 'cookie'
        ]
        
        unwanted_ids = [
            'advertisement', 'ads', 'sidebar', 'navigation', 'footer', 
            'header', 'comments', 'social', 'share'
        ]
        
        # Remove by tag
        for tag in unwanted_tags:
            for element in soup.find_all(tag):
                element.decompose()
        
        # Remove by class
        for class_name in unwanted_classes:
            for element in soup.find_all(class_=lambda x: x and any(cls in str(x).lower() for cls in [class_name])):
                element.decompose()
        
        # Remove by id
        for id_name in unwanted_ids:
            for element in soup.find_all(id=lambda x: x and id_name in str(x).lower()):
                element.decompose()

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main content using multiple strategies, prioritizing article content.
        """
        content = ""
        
        # Strategy 1: Look for article/main content tags
        main_content_selectors = [
            'article',
            'main',
            '[role="main"]',
            '.article-content',
            '.post-content', 
            '.entry-content',
            '.content',
            '.story-body',
            '.article-body'
        ]
        
        for selector in main_content_selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text(separator=' ', strip=True) for elem in elements])
                if len(content) > 200:  # Reasonable content length
                    return content
        
        # Strategy 2: Find largest text block
        all_paragraphs = soup.find_all(['p', 'div'])
        if all_paragraphs:
            # Get text from all paragraphs and find the container with most content
            text_blocks = []
            for p in all_paragraphs:
                text = p.get_text(separator=' ', strip=True)
                if len(text) > 50:  # Filter out short text blocks
                    text_blocks.append(text)
            
            if text_blocks:
                content = ' '.join(text_blocks)
                return content
        
        # Strategy 3: Fallback to body text
        body = soup.find('body')
        if body:
            content = body.get_text(separator=' ', strip=True)
            return content
        
        # Strategy 4: Last resort - all text
        return soup.get_text(separator=' ', strip=True)

    def _clean_content(self, content: str) -> str:
        """Clean extracted content"""
        if not content:
            return ""
        
        import re
        
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common noise patterns
        noise_patterns = [
            r'Cookie Policy.*?(?=\s[A-Z])',  # Cookie notices
            r'Privacy Policy.*?(?=\s[A-Z])',  # Privacy notices
            r'Subscribe.*?newsletter.*?(?=\s[A-Z])',  # Newsletter prompts
            r'Sign up.*?(?=\s[A-Z])',  # Sign up prompts
            r'\b(Advertisement|Sponsored|Ad)\b.*?(?=\s[A-Z])',  # Ad labels
        ]
        
        for pattern in noise_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        content = re.sub(r'[.]{3,}', '...', content)
        content = re.sub(r'[-]{3,}', '---', content)
        
        return content.strip()