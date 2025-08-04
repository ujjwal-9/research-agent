"""
Link extraction tool for collecting URLs from Qdrant document chunks.
"""

import logging
from typing import List, Dict, Set, Any
from dataclasses import dataclass
from collections import defaultdict

from .document_retriever import DocumentRetriever, SearchResult


@dataclass
class ExtractedLink:
    """Represents an extracted link with context."""

    url: str
    link_type: str  # "markdown", "url", "email"
    display_text: str
    source_document: str
    context: str  # Surrounding text content
    metadata: Dict[str, Any]


class LinkExtractor:
    """Tool for extracting and organizing links from document chunks."""

    def __init__(self, document_retriever: DocumentRetriever):
        """Initialize the link extractor.

        Args:
            document_retriever: Document retriever instance to use
        """
        self.logger = logging.getLogger(__name__)
        self.document_retriever = document_retriever

    def extract_links_from_search(
        self, query: str, limit: int = 50, score_threshold: float = 0.3
    ) -> List[ExtractedLink]:
        """Extract all links from documents matching a search query.

        Args:
            query: Search query to find relevant documents
            limit: Maximum number of documents to search
            score_threshold: Minimum similarity score threshold

        Returns:
            List of ExtractedLink objects
        """
        try:
            # Search for relevant documents
            search_results = self.document_retriever.search_documents(
                query=query, limit=limit, score_threshold=score_threshold
            )

            # Extract links from results
            extracted_links = []
            for result in search_results:
                links = self._extract_links_from_result(result)
                extracted_links.extend(links)

            # Deduplicate links while preserving context
            unique_links = self._deduplicate_links(extracted_links)

            self.logger.info(
                f"✅ Extracted {len(unique_links)} unique links from {len(search_results)} documents"
            )
            return unique_links

        except Exception as e:
            self.logger.error(f"❌ Error extracting links from search: {e}")
            return []

    def extract_links_from_source(
        self, source_name: str, limit: int = 100
    ) -> List[ExtractedLink]:
        """Extract all links from a specific document source.

        Args:
            source_name: Name of the document source
            limit: Maximum number of chunks to process

        Returns:
            List of ExtractedLink objects
        """
        try:
            # Get all documents from source
            documents = self.document_retriever.get_documents_by_source(
                source_name=source_name, limit=limit
            )

            # Extract links from all documents
            extracted_links = []
            for doc in documents:
                links = self._extract_links_from_result(doc)
                extracted_links.extend(links)

            # Deduplicate links
            unique_links = self._deduplicate_links(extracted_links)

            self.logger.info(
                f"✅ Extracted {len(unique_links)} unique links from source: {source_name}"
            )
            return unique_links

        except Exception as e:
            self.logger.error(
                f"❌ Error extracting links from source {source_name}: {e}"
            )
            return []

    def get_links_by_domain(
        self, query: str, target_domains: List[str] = None, limit: int = 50
    ) -> Dict[str, List[ExtractedLink]]:
        """Extract links grouped by domain.

        Args:
            query: Search query to find relevant documents
            target_domains: Optional list of specific domains to filter for
            limit: Maximum number of documents to search

        Returns:
            Dictionary mapping domain names to lists of ExtractedLink objects
        """
        try:
            # Extract all links from search
            all_links = self.extract_links_from_search(query, limit)

            # Group by domain
            links_by_domain = defaultdict(list)

            for link in all_links:
                if link.link_type in ["url", "markdown"]:
                    domain = self._extract_domain(link.url)
                    if domain:
                        # Filter by target domains if specified
                        if not target_domains or domain in target_domains:
                            links_by_domain[domain].append(link)

            result = dict(links_by_domain)
            self.logger.info(f"✅ Grouped links into {len(result)} domains")
            return result

        except Exception as e:
            self.logger.error(f"❌ Error grouping links by domain: {e}")
            return {}

    def _extract_links_from_result(self, result: SearchResult) -> List[ExtractedLink]:
        """Extract links from a single search result.

        Args:
            result: SearchResult containing document content and metadata

        Returns:
            List of ExtractedLink objects
        """
        extracted_links = []

        # Check if links are already extracted in metadata
        if result.links:
            for link_data in result.links:
                extracted_link = ExtractedLink(
                    url=link_data.get("url", ""),
                    link_type=link_data.get("type", "unknown"),
                    display_text=link_data.get("text", ""),
                    source_document=result.source,
                    context=result.content[:500],  # First 500 chars as context
                    metadata=result.metadata,
                )
                extracted_links.append(extracted_link)

        return extracted_links

    def _deduplicate_links(self, links: List[ExtractedLink]) -> List[ExtractedLink]:
        """Remove duplicate links while preserving the best context.

        Args:
            links: List of ExtractedLink objects that may contain duplicates

        Returns:
            List of unique ExtractedLink objects
        """
        # Group by URL
        url_groups = defaultdict(list)
        for link in links:
            url_groups[link.url].append(link)

        # Select best link for each URL (prefer longer context)
        unique_links = []
        for url, link_group in url_groups.items():
            if url and url.strip():  # Skip empty URLs
                # Choose link with longest context
                best_link = max(link_group, key=lambda x: len(x.context))
                unique_links.append(best_link)

        return unique_links

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL.

        Args:
            url: URL string

        Returns:
            Domain name or empty string if invalid
        """
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            return parsed.netloc.lower()
        except Exception:
            return ""

    def get_external_research_candidates(
        self, query: str, limit: int = 30
    ) -> List[ExtractedLink]:
        """Get external links that are good candidates for web research.

        Args:
            query: Search query to find relevant documents
            limit: Maximum number of documents to search

        Returns:
            List of ExtractedLink objects filtered for research value
        """
        try:
            # Extract all links
            all_links = self.extract_links_from_search(query, limit)

            # Filter for external research candidates
            research_candidates = []

            for link in all_links:
                if self._is_research_candidate(link):
                    research_candidates.append(link)

            # Sort by perceived research value
            research_candidates.sort(
                key=lambda x: self._calculate_research_value(x), reverse=True
            )

            self.logger.info(
                f"✅ Found {len(research_candidates)} research candidate links"
            )
            return research_candidates[:20]  # Return top 20

        except Exception as e:
            self.logger.error(f"❌ Error finding research candidates: {e}")
            return []

    def _is_research_candidate(self, link: ExtractedLink) -> bool:
        """Determine if a link is a good candidate for external research.

        Args:
            link: ExtractedLink to evaluate

        Returns:
            True if link is suitable for research
        """
        if link.link_type == "email":
            return False

        if not link.url or not link.url.startswith(("http://", "https://")):
            return False

        # Skip obviously internal or file links
        skip_patterns = [
            "localhost",
            "127.0.0.1",
            "file://",
            ".pdf",
            ".doc",
            ".xlsx",
            ".zip",
        ]

        for pattern in skip_patterns:
            if pattern in link.url.lower():
                return False

        return True

    def _calculate_research_value(self, link: ExtractedLink) -> float:
        """Calculate a research value score for a link.

        Args:
            link: ExtractedLink to score

        Returns:
            Research value score (higher is better)
        """
        score = 0.0

        # Favor links with meaningful display text
        if link.display_text and len(link.display_text) > 5:
            score += 2.0

        # Favor links from authoritative domains
        domain = self._extract_domain(link.url)
        authoritative_domains = [
            "github.com",
            "docs.microsoft.com",
            "python.org",
            "stackoverflow.com",
            "medium.com",
            "wikipedia.org",
            "arxiv.org",
            "ieee.org",
            "acm.org",
        ]

        for auth_domain in authoritative_domains:
            if auth_domain in domain:
                score += 3.0
                break

        # Favor links with good context
        if len(link.context) > 100:
            score += 1.0

        return score
