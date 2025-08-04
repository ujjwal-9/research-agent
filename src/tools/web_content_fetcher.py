"""
Web content fetcher tool for retrieving and processing content from external URLs.
"""

import logging
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import time
import re
from bs4 import BeautifulSoup


@dataclass
class WebContent:
    """Represents content retrieved from a web page."""

    url: str
    title: str
    content: str
    status_code: int
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    fetch_time: float = 0.0


class WebContentFetcher:
    """Tool for fetching and processing web content."""

    def __init__(
        self,
        timeout: int = 30,
        max_content_length: int = 100000,
        concurrent_requests: int = 5,
    ):
        """Initialize the web content fetcher.

        Args:
            timeout: Request timeout in seconds
            max_content_length: Maximum content length to process
            concurrent_requests: Maximum concurrent requests
        """
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.max_content_length = max_content_length
        self.concurrent_requests = concurrent_requests

        # Common headers to avoid blocking
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    async def fetch_content(self, url: str) -> WebContent:
        """Fetch content from a single URL.

        Args:
            url: URL to fetch content from

        Returns:
            WebContent object with fetched data
        """
        start_time = time.time()

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout), headers=self.headers
            ) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()

                        # Parse and clean content
                        cleaned_content, title, metadata = self._process_html_content(
                            content, url
                        )

                        # Truncate if too long
                        if len(cleaned_content) > self.max_content_length:
                            cleaned_content = (
                                cleaned_content[: self.max_content_length] + "..."
                            )

                        fetch_time = time.time() - start_time

                        return WebContent(
                            url=url,
                            title=title,
                            content=cleaned_content,
                            status_code=response.status,
                            metadata=metadata,
                            fetch_time=fetch_time,
                        )
                    else:
                        return WebContent(
                            url=url,
                            title="",
                            content="",
                            status_code=response.status,
                            error=f"HTTP {response.status}",
                            fetch_time=time.time() - start_time,
                        )

        except asyncio.TimeoutError:
            return WebContent(
                url=url,
                title="",
                content="",
                status_code=0,
                error="Timeout",
                fetch_time=time.time() - start_time,
            )
        except Exception as e:
            return WebContent(
                url=url,
                title="",
                content="",
                status_code=0,
                error=str(e),
                fetch_time=time.time() - start_time,
            )

    async def fetch_multiple_urls(self, urls: List[str]) -> List[WebContent]:
        """Fetch content from multiple URLs concurrently.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of WebContent objects
        """
        semaphore = asyncio.Semaphore(self.concurrent_requests)

        async def bounded_fetch(url):
            async with semaphore:
                return await self.fetch_content(url)

        self.logger.info(f"🌐 Fetching content from {len(urls)} URLs")

        tasks = [bounded_fetch(url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Log results
        successful = len([r for r in results if r.error is None])
        self.logger.info(f"✅ Successfully fetched {successful}/{len(urls)} URLs")

        return results

    def fetch_urls_sync(self, urls: List[str]) -> List[WebContent]:
        """Synchronous wrapper for fetching multiple URLs.

        Args:
            urls: List of URLs to fetch

        Returns:
            List of WebContent objects
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.fetch_multiple_urls(urls))

    def _process_html_content(self, html_content: str, url: str) -> tuple:
        """Process HTML content to extract clean text and metadata.

        Args:
            html_content: Raw HTML content
            url: Source URL

        Returns:
            Tuple of (cleaned_content, title, metadata)
        """
        try:
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else ""

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.extract()

            # Get main content areas (prefer article, main, or content divs)
            main_content = None
            content_selectors = [
                "article",
                "main",
                '[role="main"]',
                ".content",
                ".main-content",
                ".post-content",
                ".entry-content",
                ".article-content",
            ]

            for selector in content_selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            # If no main content found, use body
            if not main_content:
                main_content = soup.find("body")

            if not main_content:
                main_content = soup

            # Extract text
            text = main_content.get_text()

            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            cleaned_text = " ".join(chunk for chunk in chunks if chunk)

            # Extract metadata
            metadata = {
                "url": url,
                "domain": urlparse(url).netloc,
                "title": title,
                "content_length": len(cleaned_text),
            }

            # Extract meta description
            description_tag = soup.find("meta", attrs={"name": "description"})
            if description_tag:
                metadata["description"] = description_tag.get("content", "")

            # Extract meta keywords
            keywords_tag = soup.find("meta", attrs={"name": "keywords"})
            if keywords_tag:
                metadata["keywords"] = keywords_tag.get("content", "")

            return cleaned_text, title, metadata

        except Exception as e:
            self.logger.error(f"❌ Error processing HTML content: {e}")
            return html_content, "", {"url": url, "error": str(e)}

    def extract_urls_from_content(
        self, content: str, base_url: str = None
    ) -> List[str]:
        """Extract URLs from text content.

        Args:
            content: Text content to search
            base_url: Base URL for relative links

        Returns:
            List of discovered URLs
        """
        urls = []

        # Regular expression for URLs
        url_pattern = r'https?://[^\s\)>\]"\']*'
        found_urls = re.findall(url_pattern, content)

        for url in found_urls:
            # Clean up URL (remove trailing punctuation)
            cleaned_url = re.sub(r"[.,;!?]+$", "", url)
            if cleaned_url and self._is_valid_url(cleaned_url):
                urls.append(cleaned_url)

        # If base_url provided, also look for relative URLs
        if base_url:
            relative_pattern = r'href=["\']/[^"\']*["\']'
            relative_matches = re.findall(relative_pattern, content)
            for match in relative_matches:
                relative_url = (
                    match.split('"')[1] if '"' in match else match.split("'")[1]
                )
                full_url = urljoin(base_url, relative_url)
                if self._is_valid_url(full_url):
                    urls.append(full_url)

        return list(set(urls))  # Remove duplicates

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid for fetching.

        Args:
            url: URL to validate

        Returns:
            True if URL is valid
        """
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                return False

            # Skip certain file types
            skip_extensions = [
                ".pdf",
                ".doc",
                ".docx",
                ".xls",
                ".xlsx",
                ".zip",
                ".tar",
                ".gz",
            ]
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False

            # Skip localhost and private IPs
            if "localhost" in url or "127.0.0.1" in url:
                return False

            return True

        except Exception:
            return False

    def get_page_summary(self, web_content: WebContent, max_length: int = 500) -> str:
        """Generate a summary of web page content.

        Args:
            web_content: WebContent object to summarize
            max_length: Maximum summary length

        Returns:
            Summary text
        """
        if web_content.error:
            return f"Error fetching {web_content.url}: {web_content.error}"

        if not web_content.content:
            return f"No content available from {web_content.url}"

        # Create summary
        summary_parts = []

        if web_content.title:
            summary_parts.append(f"Title: {web_content.title}")

        # Take first paragraph or sentences up to max_length
        content = web_content.content.strip()
        if len(content) <= max_length:
            summary_parts.append(content)
        else:
            # Try to break at sentence boundaries
            sentences = content.split(". ")
            current_length = 0
            included_sentences = []

            for sentence in sentences:
                if current_length + len(sentence) + 2 <= max_length:
                    included_sentences.append(sentence)
                    current_length += len(sentence) + 2
                else:
                    break

            if included_sentences:
                summary_parts.append(". ".join(included_sentences) + ".")
            else:
                summary_parts.append(content[:max_length] + "...")

        return "\n\n".join(summary_parts)
