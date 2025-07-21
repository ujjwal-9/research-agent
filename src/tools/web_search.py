"""Web search tool using DuckDuckGo."""

import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from ddgs import DDGS
from loguru import logger

from src.config import settings


class WebSearchTool:
    """Tool for searching the web using DuckDuckGo."""
    
    def __init__(self):
        self.max_results = settings.duckduckgo_max_results
    
    async def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search the web for the given query."""
        max_results = max_results or self.max_results
        
        try:
            logger.info(f"Searching web for: {query}")
            
            # Use DuckDuckGo search
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            
            # Format results
            formatted_results = []
            for result in results:
                formatted_result = {
                    'title': result.get('title', ''),
                    'url': result.get('href', ''),
                    'snippet': result.get('body', ''),
                    'source': 'duckduckgo'
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} web search results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Web search failed for query '{query}': {e}")
            return []
    
    async def search_news(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search for news articles."""
        max_results = max_results or self.max_results
        
        try:
            logger.info(f"Searching news for: {query}")
            
            with DDGS() as ddgs:
                results = list(ddgs.news(query, max_results=max_results))
            
            formatted_results = []
            for result in results:
                formatted_result = {
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'snippet': result.get('body', ''),
                    'date': result.get('date', ''),
                    'source': result.get('source', 'news')
                }
                formatted_results.append(formatted_result)
            
            logger.info(f"Found {len(formatted_results)} news results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"News search failed for query '{query}': {e}")
            return []
    
    async def get_page_content(self, url: str) -> Optional[str]:
        """Fetch and extract text content from a web page."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        
                        # Simple text extraction (could be enhanced with BeautifulSoup)
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(html, 'html.parser')
                        
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        
                        # Get text content
                        text = soup.get_text()
                        
                        # Clean up text
                        lines = (line.strip() for line in text.splitlines())
                        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                        text = ' '.join(chunk for chunk in chunks if chunk)
                        
                        return text[:5000]  # Limit content length
                    else:
                        logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Error fetching content from {url}: {e}")
            return None