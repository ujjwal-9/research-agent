"""
Web researcher agent for external research and link analysis.
"""

import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from src.tools.web_content_fetcher import WebContentFetcher, WebContent
from src.tools.link_extractor import ExtractedLink

try:
    # Try the new package name first
    from ddgs import DDGS
except ImportError:
    try:
        # Fall back to old package name
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None


@dataclass
class WebResearchAnalysis:
    """Results from web research analysis."""

    research_question: str
    external_topics: List[str]
    web_content: List[WebContent]
    link_research: List[WebContent]
    research_summary: Dict[str, Any]
    key_insights: List[str]
    external_sources: List[str]


class WebResearcherAgent(BaseAgent):
    """Agent responsible for external web research."""

    def __init__(self, timeout: int = 30, max_concurrent: int = 5):
        """Initialize the web researcher agent.

        Args:
            timeout: Request timeout in seconds
            max_concurrent: Maximum concurrent requests
        """
        super().__init__("web_researcher")
        self.web_fetcher = WebContentFetcher(
            timeout=timeout, concurrent_requests=max_concurrent
        )

    async def execute(self, context: AgentContext) -> AgentResult:
        """Conduct external web research based on research plan and extracted links.

        Args:
            context: Shared context containing research plan and document analysis

        Returns:
            AgentResult containing web research analysis
        """
        try:
            # Get research plan and document analysis from context
            research_plan = context.agent_results.get("research_planner")
            document_analysis = context.agent_results.get("document_analyst")

            if not research_plan or not research_plan.data:
                raise ValueError("Research plan not found in context")

            plan = research_plan.data
            analysis = document_analysis.data if document_analysis else None

            self.logger.info(
                f"🌐 Conducting web research for: {plan.research_question}"
            )

            # Research external topics from plan
            self.logger.info(
                f"🔍 Starting external topic research with {len(plan.external_search_topics)} topics"
            )
            for i, topic in enumerate(plan.external_search_topics, 1):
                self.logger.info(f"   {i}. {topic}")

            topic_research = await self._research_external_topics(
                plan.external_search_topics
            )

            self.logger.info(
                f"📥 External topic research returned {len(topic_research)} results"
            )

            # Research extracted links if available
            link_research = []
            if analysis and analysis.extracted_links:
                self.logger.info(
                    f"🔗 Starting extracted link research with {len(analysis.extracted_links)} links"
                )
                link_research = await self._research_extracted_links(
                    analysis.extracted_links
                )
                self.logger.info(
                    f"📥 Link research returned {len(link_research)} results"
                )
            else:
                self.logger.info("🔗 No extracted links found for research")

            # Combine all web content
            all_web_content = topic_research + link_research
            self.logger.info(
                f"📊 Total web content collected: {len(all_web_content)} items"
            )

            # Analyze and summarize findings
            research_summary = await self._create_research_summary(
                all_web_content, plan.external_search_topics
            )

            # Extract key insights
            key_insights = await self._extract_key_insights(
                plan.research_question, all_web_content
            )

            # Compile external sources
            external_sources = self._compile_external_sources(all_web_content)

            # Create web research analysis
            web_analysis = WebResearchAnalysis(
                research_question=plan.research_question,
                external_topics=plan.external_search_topics,
                web_content=topic_research,
                link_research=link_research,
                research_summary=research_summary,
                key_insights=key_insights,
                external_sources=external_sources,
            )

            successful_fetches = len([w for w in all_web_content if not w.error])
            self.logger.info(
                f"✅ Completed web research: {successful_fetches}/{len(all_web_content)} "
                f"successful fetches, {len(key_insights)} insights found"
            )

            return AgentResult(
                agent_name=self.name,
                status=AgentState.COMPLETED,
                data=web_analysis,
                metadata={
                    "total_topics_researched": len(plan.external_search_topics),
                    "total_links_researched": len(link_research),
                    "successful_web_fetches": successful_fetches,
                    "failed_web_fetches": len(all_web_content) - successful_fetches,
                    "key_insights_count": len(key_insights),
                },
            )

        except Exception as e:
            self.logger.error(f"❌ Error conducting web research: {e}")
            return AgentResult(
                agent_name=self.name, status=AgentState.ERROR, data=None, error=str(e)
            )

    async def _research_external_topics(self, topics: List[str]) -> List[WebContent]:
        """Research external topics using DuckDuckGo search.

        Args:
            topics: List of topics to research

        Returns:
            List of WebContent objects
        """
        try:
            if not topics:
                self.logger.warning("⚠️ No external topics provided for research")
                return []

            if not DDGS:
                self.logger.warning(
                    "⚠️ DuckDuckGo search not available, using fallback URLs"
                )
                return await self._research_external_topics_fallback(topics)

            # Collect URLs from DuckDuckGo search results
            search_urls = []
            successful_searches = 0

            for i, topic in enumerate(topics, 1):
                self.logger.info(f"🔍 Searching topic {i}/{len(topics)}: {topic}")
                try:
                    topic_urls = await self._search_duckduckgo(topic, max_results=5)
                    if topic_urls:
                        search_urls.extend(topic_urls)
                        successful_searches += 1
                        self.logger.info(
                            f"   ✅ Found {len(topic_urls)} URLs for: {topic}"
                        )
                    else:
                        self.logger.warning(f"   ⚠️ No URLs found for: {topic}")
                except Exception as e:
                    self.logger.error(f"   ❌ Search failed for topic '{topic}': {e}")

            self.logger.info(
                f"📊 Search summary: {successful_searches}/{len(topics)} topics found URLs"
            )

            if not search_urls:
                self.logger.warning(
                    "⚠️ No URLs found from any external search, using fallback"
                )
                return await self._research_external_topics_fallback(topics)

            # Limit the number of URLs to avoid overwhelming
            limited_urls = search_urls[:20]  # Limit to 20 URLs total

            self.logger.info(
                f"🔍 Fetching content from {len(limited_urls)} external URLs"
            )

            # Fetch content from URLs
            web_content = await self.web_fetcher.fetch_multiple_urls(limited_urls)

            # Filter successful results and log details
            successful_content = []
            for content in web_content:
                if (
                    not content.error
                    and content.content
                    and len(content.content.strip()) > 50
                ):
                    successful_content.append(content)

            self.logger.info(
                f"📥 Successfully fetched {len(successful_content)}/{len(limited_urls)} external sources"
            )

            if successful_content:
                self.logger.info("✅ External research completed with results")
            else:
                self.logger.warning(
                    "⚠️ External research completed but no usable content found"
                )

            return web_content

        except Exception as e:
            self.logger.error(f"❌ Error researching external topics: {e}")
            # Try fallback as last resort
            self.logger.info("🔧 Attempting fallback external research")
            return await self._research_external_topics_fallback(topics)

    async def _search_duckduckgo(self, query: str, max_results: int = 10) -> List[str]:
        """Search DuckDuckGo for URLs related to the query.

        Args:
            query: Search query
            max_results: Maximum number of results to return

        Returns:
            List of URLs from search results
        """
        try:
            # Try multiple search strategies for better medical results
            medical_urls = []

            # Strategy 1: Specific medical sites
            medical_sites_query = f'"{query}" (site:pubmed.ncbi.nlm.nih.gov OR site:nejm.org OR site:thelancet.com OR site:bmj.com OR site:nature.com)'

            # Strategy 2: Broader medical search with medical terms
            medical_terms_query = f'"{query}" COPD pulmonary lung function assessment'

            # Strategy 3: Academic search with review/study terms
            academic_query = f'"{query}" systematic review meta-analysis clinical study'

            def search_ddgs_strategy(search_query: str, max_res: int = 5) -> List[str]:
                ddgs = DDGS()
                try:
                    results = ddgs.text(search_query, max_results=max_res)
                    found_urls = [
                        result["href"] for result in results if "href" in result
                    ]
                    # Filter for medical/academic domains
                    medical_domains = [
                        "pubmed.ncbi.nlm.nih.gov",
                        "ncbi.nlm.nih.gov",
                        "nejm.org",
                        "thelancet.com",
                        "bmj.com",
                        "nature.com",
                        "science.org",
                        "cell.com",
                        "nih.gov",
                        "ats.org",
                        "thoracic.org",
                        "erj.ersjournals.com",
                        "academic.oup.com",
                        "journals.lww.com",
                        "springer.com",
                        "onlinelibrary.wiley.com",
                        "sciencedirect.com",
                    ]
                    filtered_urls = []
                    for url in found_urls:
                        if any(domain in url.lower() for domain in medical_domains):
                            filtered_urls.append(url)
                    return filtered_urls
                except Exception:
                    return []

            # Execute search strategies in thread pool
            loop = asyncio.get_event_loop()

            # Try medical sites first
            medical_urls = await loop.run_in_executor(
                None, search_ddgs_strategy, medical_sites_query, 8
            )

            # If not enough results, try broader medical search
            if len(medical_urls) < 3:
                additional_urls = await loop.run_in_executor(
                    None, search_ddgs_strategy, medical_terms_query, 6
                )
                medical_urls.extend(additional_urls)

            # If still not enough, try academic search
            if len(medical_urls) < 3:
                academic_urls = await loop.run_in_executor(
                    None, search_ddgs_strategy, academic_query, 6
                )
                medical_urls.extend(academic_urls)

            # Remove duplicates while preserving order
            unique_urls = []
            seen = set()
            for url in medical_urls:
                if url not in seen:
                    unique_urls.append(url)
                    seen.add(url)

            # Limit results
            final_urls = unique_urls[:max_results]

            self.logger.info(
                f'🔍 Found {len(final_urls)} medical URLs for query: "{query}"'
            )
            if final_urls:
                self.logger.info(
                    f"🏥 Sample domains: {[url.split('/')[2] for url in final_urls[:3]]}"
                )

            return final_urls

        except Exception as e:
            self.logger.warning(f"⚠️ DuckDuckGo search failed for '{query}': {e}")
            # Fallback to basic medical search URLs
            return self._generate_fallback_medical_urls(query)

    def _generate_fallback_medical_urls(self, query: str) -> List[str]:
        """Generate fallback medical search URLs when DuckDuckGo fails.

        Args:
            query: Search query

        Returns:
            List of fallback URLs
        """
        import urllib.parse

        # Clean and encode query for different search engines
        query_encoded = urllib.parse.quote_plus(query)
        query_simple = query.replace(" ", "+")

        # Add COPD-specific terms to improve relevance
        medical_query = f"{query} COPD lung function"
        medical_encoded = urllib.parse.quote_plus(medical_query)

        fallback_urls = [
            # PubMed with original query
            f"https://pubmed.ncbi.nlm.nih.gov/?term={query_encoded}",
            # PubMed with enhanced medical terms
            f"https://pubmed.ncbi.nlm.nih.gov/?term={medical_encoded}",
            # PMC (full-text articles)
            f"https://www.ncbi.nlm.nih.gov/pmc/?term={query_encoded}",
            # Google Scholar with medical focus
            f"https://scholar.google.com/scholar?q={query_encoded}+COPD+lung+function",
            # European Respiratory Journal
            f"https://erj.ersjournals.com/search?text1={query_simple}&field1=ABSTRACT_TEXT",
            # American Journal of Respiratory and Critical Care Medicine
            f"https://www.atsjournals.org/action/doSearch?text1={query_simple}&field1=Abstract",
        ]

        self.logger.info(
            f"🔧 Generated {len(fallback_urls)} fallback medical URLs for: {query}"
        )
        return fallback_urls[:5]  # Return top 5 fallback URLs

    async def _research_external_topics_fallback(
        self, topics: List[str]
    ) -> List[WebContent]:
        """Fallback method for external research when DuckDuckGo is not available.

        Args:
            topics: List of topics to research

        Returns:
            List of WebContent objects
        """
        try:
            search_urls = []
            self.logger.info(f"🔧 Using fallback URLs for {len(topics)} topics")

            for i, topic in enumerate(topics, 1):
                self.logger.info(f"🔧 Generating fallback URLs for topic {i}: {topic}")
                # Generate fallback medical URLs
                topic_urls = self._generate_fallback_medical_urls(topic)
                search_urls.extend(topic_urls)
                self.logger.info(f"   Generated {len(topic_urls)} fallback URLs")

            # Limit the number of URLs
            limited_urls = search_urls[:25]  # Slightly more URLs for fallback

            self.logger.info(
                f"🔍 Fetching content from {len(limited_urls)} fallback URLs"
            )

            # Fetch content from URLs
            web_content = await self.web_fetcher.fetch_multiple_urls(limited_urls)

            # Filter and count successful results
            successful_content = [
                w
                for w in web_content
                if not w.error and w.content and len(w.content.strip()) > 50
            ]

            self.logger.info(
                f"📥 Fallback research: {len(successful_content)}/{len(limited_urls)} URLs returned usable content"
            )

            return web_content

        except Exception as e:
            self.logger.error(f"❌ Error in fallback external research: {e}")
            return []

    async def _research_extracted_links(
        self, links: List[ExtractedLink]
    ) -> List[WebContent]:
        """Research the extracted links from internal documents.

        Args:
            links: List of extracted links to research

        Returns:
            List of WebContent objects
        """
        try:
            # Extract URLs from links
            urls = [link.url for link in links if self._is_researchable_url(link.url)]

            # Limit the number of links to research
            limited_urls = urls[:15]  # Limit to 15 links

            self.logger.info(f"🔗 Researching {len(limited_urls)} extracted links")

            # Fetch content from links
            web_content = await self.web_fetcher.fetch_multiple_urls(limited_urls)

            successful_content = [w for w in web_content if not w.error and w.content]

            self.logger.info(
                f"Successfully fetched {len(successful_content)}/{len(limited_urls)} extracted links"
            )

            return web_content

        except Exception as e:
            self.logger.error(f"Error researching extracted links: {e}")
            return []

    def _is_researchable_url(self, url: str) -> bool:
        """Check if a URL is suitable for research.

        Args:
            url: URL to check

        Returns:
            True if URL can be researched
        """
        if not url or not url.startswith(("http://", "https://")):
            return False

        # Skip certain file types and domains
        skip_patterns = [
            "localhost",
            "127.0.0.1",
            "file://",
            ".pdf",
            ".doc",
            ".docx",
            ".xls",
            ".xlsx",
            ".zip",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
        ]

        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return False

        return True

    async def _create_research_summary(
        self, web_content: List[WebContent], topics: List[str]
    ) -> Dict[str, Any]:
        """Create a summary of the web research conducted.

        Args:
            web_content: Web content fetched
            topics: Original topics researched

        Returns:
            Research summary
        """
        successful_content = [w for w in web_content if not w.error and w.content]
        failed_content = [w for w in web_content if w.error]

        # Analyze domains covered
        domains = set()
        for content in successful_content:
            if content.metadata and "domain" in content.metadata:
                domains.add(content.metadata["domain"])

        # Calculate content statistics
        total_content_length = sum(len(c.content) for c in successful_content)
        avg_content_length = (
            total_content_length / len(successful_content) if successful_content else 0
        )

        # Analyze fetch performance
        total_fetch_time = sum(c.fetch_time for c in web_content)
        avg_fetch_time = total_fetch_time / len(web_content) if web_content else 0

        summary = {
            "topics_researched": len(topics),
            "urls_attempted": len(web_content),
            "successful_fetches": len(successful_content),
            "failed_fetches": len(failed_content),
            "success_rate": (
                len(successful_content) / len(web_content) if web_content else 0
            ),
            "domains_covered": list(domains),
            "unique_domains": len(domains),
            "total_content_length": total_content_length,
            "average_content_length": avg_content_length,
            "total_fetch_time": total_fetch_time,
            "average_fetch_time": avg_fetch_time,
            "error_summary": self._summarize_errors(failed_content),
        }

        return summary

    def _summarize_errors(self, failed_content: List[WebContent]) -> Dict[str, int]:
        """Summarize errors from failed web content fetches.

        Args:
            failed_content: List of failed WebContent objects

        Returns:
            Dictionary summarizing error types
        """
        error_counts = {}
        for content in failed_content:
            error_type = content.error or "Unknown error"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        return error_counts

    async def _extract_key_insights(
        self, research_question: str, web_content: List[WebContent]
    ) -> List[str]:
        """Extract key insights from web research content.

        Args:
            research_question: Original research question
            web_content: Web content to analyze

        Returns:
            List of key insights
        """
        insights = []
        successful_content = [w for w in web_content if not w.error and w.content]

        if not successful_content:
            insights.append(
                "No external content was successfully retrieved for analysis."
            )
            return insights

        # Analyze content sources
        domains = set()
        for content in successful_content:
            if content.metadata and "domain" in content.metadata:
                domains.add(content.metadata["domain"])

        if domains:
            insights.append(
                f"External research covered {len(domains)} different domains: "
                f"{', '.join(sorted(list(domains))[:5])}"
            )

        # Analyze content characteristics
        total_content = sum(len(c.content) for c in successful_content)
        avg_length = total_content / len(successful_content)

        if avg_length > 5000:
            insights.append("Retrieved comprehensive content with detailed information")
        elif avg_length > 1000:
            insights.append("Retrieved moderate-length content with useful information")
        else:
            insights.append("Retrieved brief content summaries")

        # Check for recent content (based on titles/URLs)
        recent_indicators = ["2024", "2023", "latest", "recent", "new", "updated"]
        recent_content_count = 0

        for content in successful_content:
            title_url_text = f"{content.title} {content.url}".lower()
            if any(indicator in title_url_text for indicator in recent_indicators):
                recent_content_count += 1

        if recent_content_count > 0:
            insights.append(
                f"Found {recent_content_count} sources with recent or updated information"
            )

        # Analyze content diversity
        if len(successful_content) >= 5:
            insights.append(
                "Gathered diverse external perspectives from multiple sources"
            )
        elif len(successful_content) >= 2:
            insights.append("Obtained external validation from multiple sources")
        else:
            insights.append("Limited external sources available")

        # Check for authoritative sources
        authoritative_domains = [
            "github.com",
            "docs.microsoft.com",
            "python.org",
            "stackoverflow.com",
            "arxiv.org",
            "ieee.org",
            "acm.org",
        ]

        authoritative_count = 0
        for content in successful_content:
            if content.metadata and "domain" in content.metadata:
                domain = content.metadata["domain"]
                if any(auth_domain in domain for auth_domain in authoritative_domains):
                    authoritative_count += 1

        if authoritative_count > 0:
            insights.append(
                f"Accessed {authoritative_count} authoritative sources for reliable information"
            )

        return insights

    def _compile_external_sources(self, web_content: List[WebContent]) -> List[str]:
        """Compile list of external sources accessed.

        Args:
            web_content: Web content fetched

        Returns:
            List of source descriptions
        """
        sources = []
        successful_content = [w for w in web_content if not w.error and w.content]

        for content in successful_content:
            if content.title:
                source_desc = f"{content.title} ({content.url})"
            else:
                source_desc = content.url

            sources.append(source_desc)

        # Sort and limit
        sources.sort()
        return sources[:20]  # Return top 20 sources
