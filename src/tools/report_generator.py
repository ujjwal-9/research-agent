"""
Report generation tool for creating structured research reports.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json

from .document_retriever import SearchResult
from .link_extractor import ExtractedLink
from .web_content_fetcher import WebContent


@dataclass
class ResearchEvidence:
    """Represents a piece of evidence from research."""

    source_type: str  # "internal_document", "external_web", "extracted_link"
    source_name: str
    content: str
    relevance_score: float
    url: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ResearchSection:
    """Represents a section of the research report."""

    title: str
    content: str
    evidence: List[ResearchEvidence]
    subsections: List["ResearchSection"] = None


@dataclass
class ResearchReport:
    """Complete research report structure."""

    title: str
    executive_summary: str
    sections: List[ResearchSection]
    methodology: str
    sources: List[str]
    generated_at: datetime
    metadata: Dict[str, Any] = None


class ReportGenerator:
    """Tool for generating structured research reports."""

    def __init__(self):
        """Initialize the report generator."""
        self.logger = logging.getLogger(__name__)

    def create_research_report(
        self,
        title: str,
        research_question: str,
        internal_results: List[SearchResult],
        external_results: List[WebContent],
        extracted_links: List[ExtractedLink],
        analysis_sections: List[Dict[str, Any]] = None,
    ) -> ResearchReport:
        """Create a comprehensive research report.

        Args:
            title: Report title
            research_question: Original research question
            internal_results: Results from internal document search
            external_results: Results from web content fetching
            extracted_links: Links extracted from documents
            analysis_sections: Custom analysis sections

        Returns:
            ResearchReport object
        """
        try:
            self.logger.info(f"📄 Generating research report: {title}")

            # Convert results to evidence
            evidence = self._compile_evidence(
                internal_results, external_results, extracted_links
            )

            # Generate executive summary
            executive_summary = self._generate_executive_summary(
                research_question, evidence
            )

            # Create main sections
            sections = []

            # Internal sources section
            if internal_results:
                internal_section = self._create_internal_sources_section(
                    internal_results
                )
                sections.append(internal_section)

            # External sources section
            if external_results:
                external_section = self._create_external_sources_section(
                    external_results
                )
                sections.append(external_section)

            # Links and resources section
            if extracted_links:
                links_section = self._create_links_section(extracted_links)
                sections.append(links_section)

            # Custom analysis sections
            if analysis_sections:
                for analysis in analysis_sections:
                    custom_section = ResearchSection(
                        title=analysis.get("title", "Analysis"),
                        content=analysis.get("content", ""),
                        evidence=[],
                    )
                    sections.append(custom_section)

            # Generate methodology section
            methodology = self._generate_methodology(
                len(internal_results), len(external_results), len(extracted_links)
            )

            # Compile sources list
            sources = self._compile_sources_list(
                internal_results, external_results, extracted_links
            )

            # Create report
            report = ResearchReport(
                title=title,
                executive_summary=executive_summary,
                sections=sections,
                methodology=methodology,
                sources=sources,
                generated_at=datetime.now(),
                metadata={
                    "research_question": research_question,
                    "total_internal_sources": len(internal_results),
                    "total_external_sources": len(external_results),
                    "total_extracted_links": len(extracted_links),
                },
            )

            self.logger.info(f"✅ Generated report with {len(sections)} sections")
            return report

        except Exception as e:
            self.logger.error(f"❌ Error generating research report: {e}")
            raise

    def _compile_evidence(
        self,
        internal_results: List[SearchResult],
        external_results: List[WebContent],
        extracted_links: List[ExtractedLink],
    ) -> List[ResearchEvidence]:
        """Compile all research results into evidence objects.

        Args:
            internal_results: Internal document search results
            external_results: External web content results
            extracted_links: Extracted links

        Returns:
            List of ResearchEvidence objects
        """
        evidence = []

        # Process internal results
        for result in internal_results:
            evidence.append(
                ResearchEvidence(
                    source_type="internal_document",
                    source_name=result.source,
                    content=result.content,
                    relevance_score=result.score,
                    metadata=result.metadata,
                )
            )

        # Process external results
        for result in external_results:
            if not result.error and result.content:
                evidence.append(
                    ResearchEvidence(
                        source_type="external_web",
                        source_name=result.title or result.url,
                        content=result.content,
                        relevance_score=1.0,  # Web content doesn't have similarity scores
                        url=result.url,
                        metadata=result.metadata,
                    )
                )

        # Process extracted links
        for link in extracted_links:
            evidence.append(
                ResearchEvidence(
                    source_type="extracted_link",
                    source_name=link.source_document,
                    content=link.context,
                    relevance_score=0.8,  # Default relevance for links
                    url=link.url,
                    metadata=link.metadata,
                )
            )

        return evidence

    def _generate_executive_summary(
        self, research_question: str, evidence: List[ResearchEvidence]
    ) -> str:
        """Generate an executive summary of the research.

        Args:
            research_question: Original research question
            evidence: List of research evidence

        Returns:
            Executive summary text
        """
        total_sources = len(evidence)
        internal_count = len(
            [e for e in evidence if e.source_type == "internal_document"]
        )
        external_count = len([e for e in evidence if e.source_type == "external_web"])
        links_count = len([e for e in evidence if e.source_type == "extracted_link"])

        summary = f"""
This research report addresses the question: "{research_question}"

The analysis draws from {total_sources} total sources, including:
- {internal_count} internal documents from our knowledge base
- {external_count} external web sources 
- {links_count} extracted links and references

Key findings and insights are organized in the sections below, with supporting evidence from both internal documentation and external research.
        """.strip()

        return summary

    def _create_internal_sources_section(
        self, internal_results: List[SearchResult]
    ) -> ResearchSection:
        """Create a section for internal document sources.

        Args:
            internal_results: Internal search results

        Returns:
            ResearchSection object
        """
        # Group by source document
        sources_by_doc = {}
        for result in internal_results:
            doc_name = result.source
            if doc_name not in sources_by_doc:
                sources_by_doc[doc_name] = []
            sources_by_doc[doc_name].append(result)

        content_parts = []
        content_parts.append(f"Analysis of {len(sources_by_doc)} internal documents:\n")

        for doc_name, results in sources_by_doc.items():
            content_parts.append(f"\n### {doc_name}")
            content_parts.append(f"Found {len(results)} relevant sections:")

            for i, result in enumerate(results[:3], 1):  # Show top 3 results per doc
                preview = (
                    result.content[:200] + "..."
                    if len(result.content) > 200
                    else result.content
                )
                content_parts.append(f"\n{i}. (Score: {result.score:.2f}) {preview}")

        # Convert to evidence
        evidence = [
            ResearchEvidence(
                source_type="internal_document",
                source_name=result.source,
                content=result.content,
                relevance_score=result.score,
                metadata=result.metadata,
            )
            for result in internal_results
        ]

        return ResearchSection(
            title="Internal Document Analysis",
            content="\n".join(content_parts),
            evidence=evidence,
        )

    def _create_external_sources_section(
        self, external_results: List[WebContent]
    ) -> ResearchSection:
        """Create a section for external web sources.

        Args:
            external_results: External web content results

        Returns:
            ResearchSection object
        """
        successful_results = [r for r in external_results if not r.error and r.content]

        content_parts = []
        content_parts.append(
            f"Analysis of {len(successful_results)} external web sources:\n"
        )

        for i, result in enumerate(successful_results, 1):
            content_parts.append(f"\n### {i}. {result.title or 'Untitled'}")
            content_parts.append(f"**URL:** {result.url}")

            if result.metadata and "description" in result.metadata:
                content_parts.append(
                    f"**Description:** {result.metadata['description']}"
                )

            # Summary of content
            preview = (
                result.content[:300] + "..."
                if len(result.content) > 300
                else result.content
            )
            content_parts.append(f"**Content Summary:** {preview}")

        # Convert to evidence
        evidence = [
            ResearchEvidence(
                source_type="external_web",
                source_name=result.title or result.url,
                content=result.content,
                relevance_score=1.0,
                url=result.url,
                metadata=result.metadata,
            )
            for result in successful_results
        ]

        return ResearchSection(
            title="External Web Sources",
            content="\n".join(content_parts),
            evidence=evidence,
        )

    def _create_links_section(
        self, extracted_links: List[ExtractedLink]
    ) -> ResearchSection:
        """Create a section for extracted links and resources.

        Args:
            extracted_links: Extracted links

        Returns:
            ResearchSection object
        """
        # Group by domain
        links_by_domain = {}
        for link in extracted_links:
            domain = (
                link.url.split("//")[1].split("/")[0] if "//" in link.url else "unknown"
            )
            if domain not in links_by_domain:
                links_by_domain[domain] = []
            links_by_domain[domain].append(link)

        content_parts = []
        content_parts.append(
            f"Extracted {len(extracted_links)} relevant links from internal documents:\n"
        )

        for domain, links in links_by_domain.items():
            content_parts.append(f"\n### {domain}")
            content_parts.append(f"Found {len(links)} links:")

            for link in links[:5]:  # Show top 5 links per domain
                display_text = link.display_text or link.url
                content_parts.append(f"- [{display_text}]({link.url})")
                if link.context:
                    context_preview = (
                        link.context[:100] + "..."
                        if len(link.context) > 100
                        else link.context
                    )
                    content_parts.append(f"  Context: {context_preview}")

        # Convert to evidence
        evidence = [
            ResearchEvidence(
                source_type="extracted_link",
                source_name=link.source_document,
                content=link.context,
                relevance_score=0.8,
                url=link.url,
                metadata=link.metadata,
            )
            for link in extracted_links
        ]

        return ResearchSection(
            title="Related Links and Resources",
            content="\n".join(content_parts),
            evidence=evidence,
        )

    def _generate_methodology(
        self, internal_count: int, external_count: int, links_count: int
    ) -> str:
        """Generate methodology description.

        Args:
            internal_count: Number of internal sources
            external_count: Number of external sources
            links_count: Number of extracted links

        Returns:
            Methodology description
        """
        methodology = f"""
**Research Methodology:**

1. **Internal Document Search**: Performed semantic search across internal knowledge base, retrieving {internal_count} relevant documents using vector similarity.

2. **Link Extraction**: Extracted {links_count} relevant links from internal documents to identify external resources.

3. **External Web Research**: Fetched and analyzed content from {external_count} external web sources identified through link extraction.

4. **Content Analysis**: Processed and analyzed all sources to identify key themes, insights, and relevant information.

5. **Synthesis**: Combined findings from internal and external sources to provide comprehensive analysis.
        """.strip()

        return methodology

    def _compile_sources_list(
        self,
        internal_results: List[SearchResult],
        external_results: List[WebContent],
        extracted_links: List[ExtractedLink],
    ) -> List[str]:
        """Compile a list of all sources used.

        Args:
            internal_results: Internal search results
            external_results: External web content
            extracted_links: Extracted links

        Returns:
            List of source references
        """
        sources = []

        # Internal sources
        internal_sources = set(result.source for result in internal_results)
        for source in sorted(internal_sources):
            sources.append(f"Internal Document: {source}")

        # External sources
        successful_external = [r for r in external_results if not r.error and r.content]
        for result in successful_external:
            title = result.title or "Untitled"
            sources.append(f"Web Source: {title} ({result.url})")

        # Extracted links (unique URLs only)
        link_urls = set(link.url for link in extracted_links)
        for url in sorted(link_urls):
            sources.append(f"Reference Link: {url}")

        return sources

    def export_to_markdown(
        self, report: ResearchReport, comprehensive_answer: str = None
    ) -> str:
        """Export research report to markdown format.

        Args:
            report: ResearchReport to export
            comprehensive_answer: Optional comprehensive answer to include

        Returns:
            Markdown formatted report
        """
        md_parts = []

        # Title and metadata
        md_parts.append(f"# {report.title}")
        md_parts.append(
            f"\n*Generated on: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}*"
        )

        if report.metadata and "research_question" in report.metadata:
            md_parts.append(
                f"\n**Research Question:** {report.metadata['research_question']}"
            )

        # Executive Summary
        md_parts.append("\n## Executive Summary")
        md_parts.append(f"\n{report.executive_summary}")

        # Sections
        for section in report.sections:
            md_parts.append(f"\n## {section.title}")
            md_parts.append(f"\n{section.content}")

        # Comprehensive Answer (if provided)
        if comprehensive_answer:
            md_parts.append("\n## Comprehensive Research Answer")
            md_parts.append(f"\n{comprehensive_answer}")

        # Methodology
        md_parts.append("\n## Methodology")
        md_parts.append(f"\n{report.methodology}")

        # Sources
        md_parts.append("\n## Sources")
        for i, source in enumerate(report.sources, 1):
            md_parts.append(f"\n{i}. {source}")

        return "\n".join(md_parts)

    def export_to_json(self, report: ResearchReport) -> str:
        """Export research report to JSON format.

        Args:
            report: ResearchReport to export

        Returns:
            JSON formatted report
        """

        def serialize_section(section):
            return {
                "title": section.title,
                "content": section.content,
                "evidence_count": len(section.evidence),
                "subsections": (
                    [serialize_section(sub) for sub in section.subsections]
                    if section.subsections
                    else []
                ),
            }

        report_dict = {
            "title": report.title,
            "executive_summary": report.executive_summary,
            "sections": [serialize_section(section) for section in report.sections],
            "methodology": report.methodology,
            "sources": report.sources,
            "generated_at": report.generated_at.isoformat(),
            "metadata": report.metadata,
        }

        return json.dumps(report_dict, indent=2, ensure_ascii=False)
