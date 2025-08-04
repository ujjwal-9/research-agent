"""
Synthesis agent for combining research findings into comprehensive reports.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from ..tools.report_generator import ReportGenerator, ResearchReport
from ..tools.document_retriever import SearchResult
from ..tools.web_content_fetcher import WebContent


@dataclass
class SynthesisResult:
    """Results from research synthesis."""

    research_report: ResearchReport
    synthesis_summary: Dict[str, Any]
    integration_analysis: Dict[str, Any]
    recommendations: List[str]
    comprehensive_answer: str = ""


class SynthesisAgent(BaseAgent):
    """Agent responsible for synthesizing research findings into comprehensive reports."""

    def __init__(self):
        """Initialize the synthesis agent."""
        super().__init__("synthesis_agent")
        self.report_generator = ReportGenerator()

    async def execute(self, context: AgentContext) -> AgentResult:
        """Synthesize all research findings into a comprehensive report.

        Args:
            context: Shared context containing all research results

        Returns:
            AgentResult containing synthesis results
        """
        try:
            # Get all research results from context
            research_plan = context.agent_results.get("research_planner")
            document_analysis = context.agent_results.get("document_analyst")
            web_research = context.agent_results.get("web_researcher")

            if not research_plan or not research_plan.data:
                raise ValueError("Research plan not found in context")

            plan = research_plan.data
            doc_analysis = document_analysis.data if document_analysis else None
            web_analysis = web_research.data if web_research else None

            self.logger.info(
                f"🔬 Synthesizing research findings for: {plan.research_question}"
            )

            # Extract data for report generation
            internal_results = doc_analysis.search_results if doc_analysis else []
            external_results = []
            extracted_links = doc_analysis.extracted_links if doc_analysis else []

            if web_analysis:
                external_results.extend(web_analysis.web_content)
                external_results.extend(web_analysis.link_research)

            # Generate comprehensive report
            report = self.report_generator.create_research_report(
                title=f"Research Report: {plan.research_question}",
                research_question=plan.research_question,
                internal_results=internal_results,
                external_results=external_results,
                extracted_links=extracted_links,
                analysis_sections=await self._create_analysis_sections(
                    plan, doc_analysis, web_analysis
                ),
            )

            # Create synthesis summary
            synthesis_summary = await self._create_synthesis_summary(
                plan, doc_analysis, web_analysis
            )

            # Analyze integration of findings
            integration_analysis = await self._analyze_integration(
                internal_results, external_results
            )

            # Generate recommendations
            recommendations = await self._generate_recommendations(
                plan.research_question, doc_analysis, web_analysis
            )

            # Generate comprehensive answer to the research question
            comprehensive_answer = await self._generate_comprehensive_answer(
                plan.research_question, doc_analysis, web_analysis
            )

            # Create synthesis result
            synthesis_result = SynthesisResult(
                research_report=report,
                synthesis_summary=synthesis_summary,
                integration_analysis=integration_analysis,
                recommendations=recommendations,
                comprehensive_answer=comprehensive_answer,
            )

            self.logger.info(
                f"✅ Synthesized research with {len(report.sections)} sections, "
                f"{len(recommendations)} recommendations"
            )

            return AgentResult(
                agent_name=self.name,
                status=AgentState.COMPLETED,
                data=synthesis_result,
                metadata={
                    "report_sections": len(report.sections),
                    "total_sources": len(report.sources),
                    "recommendations_count": len(recommendations),
                    "internal_sources_analyzed": len(internal_results),
                    "external_sources_analyzed": len(
                        [r for r in external_results if not r.error]
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"❌ Error synthesizing research findings: {e}")
            return AgentResult(
                agent_name=self.name, status=AgentState.ERROR, data=None, error=str(e)
            )

    async def _create_analysis_sections(
        self, plan, doc_analysis, web_analysis
    ) -> List[Dict[str, Any]]:
        """Create custom analysis sections for the report.

        Args:
            plan: Research plan
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            List of analysis section dictionaries
        """
        sections = []

        # Research question analysis section
        sections.append(
            {
                "title": "Research Question Analysis",
                "content": await self._create_question_analysis(plan.research_question),
            }
        )

        # Findings integration section
        if doc_analysis and web_analysis:
            sections.append(
                {
                    "title": "Internal vs External Findings",
                    "content": await self._create_findings_comparison(
                        doc_analysis, web_analysis
                    ),
                }
            )

        # Coverage assessment section
        sections.append(
            {
                "title": "Research Coverage Assessment",
                "content": await self._create_coverage_assessment(
                    plan, doc_analysis, web_analysis
                ),
            }
        )

        # Key insights section
        sections.append(
            {
                "title": "Key Insights and Conclusions",
                "content": await self._create_key_insights_section(
                    plan, doc_analysis, web_analysis
                ),
            }
        )

        return sections

    async def _create_question_analysis(self, research_question: str) -> str:
        """Create analysis of how well the research question was addressed.

        Args:
            research_question: Original research question

        Returns:
            Analysis content
        """
        analysis_parts = [
            f"**Original Research Question:** {research_question}",
            "",
            "**Question Scope Analysis:**",
        ]

        # Analyze question characteristics
        question_lower = research_question.lower()

        if "how" in question_lower:
            analysis_parts.append(
                "- Procedural question requiring step-by-step information"
            )
        elif "what" in question_lower:
            analysis_parts.append(
                "- Definitional question requiring explanation and description"
            )
        elif "why" in question_lower:
            analysis_parts.append(
                "- Causal question requiring explanation of reasons and benefits"
            )
        elif "compare" in question_lower:
            analysis_parts.append(
                "- Comparative question requiring analysis of alternatives"
            )
        else:
            analysis_parts.append(
                "- Exploratory question requiring comprehensive investigation"
            )

        # Assess complexity
        word_count = len(research_question.split())
        if word_count > 15:
            analysis_parts.append(
                "- High complexity question requiring multi-faceted analysis"
            )
        elif word_count > 8:
            analysis_parts.append(
                "- Moderate complexity question with multiple components"
            )
        else:
            analysis_parts.append("- Focused question with clear scope")

        return "\n".join(analysis_parts)

    async def _create_findings_comparison(self, doc_analysis, web_analysis) -> str:
        """Create comparison between internal and external findings.

        Args:
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            Comparison content
        """
        comparison_parts = ["**Internal vs External Research Comparison:**", ""]

        # Internal findings summary
        if doc_analysis:
            internal_count = len(doc_analysis.search_results)
            internal_sources = len(set(r.source for r in doc_analysis.search_results))
            comparison_parts.extend(
                [
                    "**Internal Findings:**",
                    f"- {internal_count} relevant documents found",
                    f"- {internal_sources} unique internal sources analyzed",
                    f"- Key findings: {'; '.join(doc_analysis.key_findings[:3])}",
                ]
            )

        comparison_parts.append("")

        # External findings summary
        if web_analysis:
            external_successful = len(
                [
                    w
                    for w in web_analysis.web_content + web_analysis.link_research
                    if not w.error
                ]
            )
            domains = set()
            for content in web_analysis.web_content + web_analysis.link_research:
                if content.metadata and "domain" in content.metadata:
                    domains.add(content.metadata["domain"])

            comparison_parts.extend(
                [
                    "**External Findings:**",
                    f"- {external_successful} external sources successfully analyzed",
                    f"- {len(domains)} unique domains covered",
                    f"- Key insights: {'; '.join(web_analysis.key_insights[:3])}",
                ]
            )

        comparison_parts.extend(
            [
                "",
                "**Integration Assessment:**",
                "- Internal sources provide organizational context and established practices",
                "- External sources offer industry perspectives and current trends",
                "- Combined findings provide comprehensive view of the topic",
            ]
        )

        return "\n".join(comparison_parts)

    async def _create_coverage_assessment(
        self, plan, doc_analysis, web_analysis
    ) -> str:
        """Create assessment of research coverage.

        Args:
            plan: Research plan
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            Coverage assessment content
        """
        assessment_parts = ["**Research Coverage Assessment:**", ""]

        # Planned vs executed
        planned_internal = len(plan.internal_search_queries)
        planned_external = len(plan.external_search_topics)

        assessment_parts.extend(
            [
                "**Planned Research:**",
                f"- {planned_internal} internal search queries planned",
                f"- {planned_external} external topics planned",
            ]
        )

        # Actual execution
        if doc_analysis:
            actual_results = len(doc_analysis.search_results)
            assessment_parts.append(f"- {actual_results} internal documents retrieved")

        if web_analysis:
            external_attempted = len(web_analysis.web_content) + len(
                web_analysis.link_research
            )
            external_successful = len(
                [
                    w
                    for w in web_analysis.web_content + web_analysis.link_research
                    if not w.error
                ]
            )
            assessment_parts.extend(
                [
                    f"- {external_attempted} external sources attempted",
                    f"- {external_successful} external sources successfully retrieved",
                ]
            )

        # Coverage quality assessment
        assessment_parts.extend(["", "**Coverage Quality:**"])

        if doc_analysis and doc_analysis.coverage_analysis:
            coverage = doc_analysis.coverage_analysis
            effectiveness = coverage.get("query_effectiveness", 0)
            if effectiveness > 0.8:
                assessment_parts.append(
                    "- Excellent internal coverage with high query effectiveness"
                )
            elif effectiveness > 0.5:
                assessment_parts.append(
                    "- Good internal coverage with moderate query effectiveness"
                )
            else:
                assessment_parts.append(
                    "- Limited internal coverage, may need refined queries"
                )

        if web_analysis and web_analysis.research_summary:
            success_rate = web_analysis.research_summary.get("success_rate", 0)
            if success_rate > 0.7:
                assessment_parts.append(
                    "- Strong external research with high success rate"
                )
            elif success_rate > 0.4:
                assessment_parts.append("- Moderate external research success")
            else:
                assessment_parts.append(
                    "- Limited external research success due to access constraints"
                )

        return "\n".join(assessment_parts)

    async def _create_key_insights_section(
        self, plan, doc_analysis, web_analysis
    ) -> str:
        """Create key insights section combining all findings.

        Args:
            plan: Research plan
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            Key insights content
        """
        insights_parts = ["**Key Insights and Conclusions:**", ""]

        # Combine insights from all sources
        all_insights = []

        if doc_analysis and doc_analysis.key_findings:
            insights_parts.append("**From Internal Analysis:**")
            for finding in doc_analysis.key_findings:
                insights_parts.append(f"- {finding}")
            insights_parts.append("")

        if web_analysis and web_analysis.key_insights:
            insights_parts.append("**From External Research:**")
            for insight in web_analysis.key_insights:
                insights_parts.append(f"- {insight}")
            insights_parts.append("")

        # Overall conclusions
        insights_parts.extend(
            [
                "**Overall Conclusions:**",
                f"- The research question '{plan.research_question}' has been addressed through multi-source analysis",
                "- Both internal organizational knowledge and external industry perspectives were incorporated",
                "- The findings provide actionable insights based on current information",
            ]
        )

        return "\n".join(insights_parts)

    async def _create_synthesis_summary(
        self, plan, doc_analysis, web_analysis
    ) -> Dict[str, Any]:
        """Create summary of the synthesis process.

        Args:
            plan: Research plan
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            Synthesis summary
        """
        # Count all sources and results
        internal_count = len(doc_analysis.search_results) if doc_analysis else 0
        external_count = 0
        external_successful = 0

        if web_analysis:
            all_external = web_analysis.web_content + web_analysis.link_research
            external_count = len(all_external)
            external_successful = len([w for w in all_external if not w.error])

        links_extracted = len(doc_analysis.extracted_links) if doc_analysis else 0

        summary = {
            "research_question": plan.research_question,
            "research_approach": {
                "internal_queries_planned": len(plan.internal_search_queries),
                "external_topics_planned": len(plan.external_search_topics),
                "internal_results_found": internal_count,
                "external_sources_attempted": external_count,
                "external_sources_successful": external_successful,
                "links_extracted": links_extracted,
            },
            "synthesis_metrics": {
                "total_sources_analyzed": internal_count + external_successful,
                "internal_external_ratio": internal_count / max(external_successful, 1),
                "research_success_rate": (internal_count + external_successful)
                / max(len(plan.internal_search_queries) + external_count, 1),
            },
            "research_quality": {
                "comprehensive_coverage": internal_count > 5
                and external_successful > 3,
                "diverse_sources": (internal_count > 0 and external_successful > 0),
                "recent_information": external_successful
                > 0,  # External sources provide recent info
            },
        }

        return summary

    async def _analyze_integration(
        self, internal_results: List[SearchResult], external_results: List[WebContent]
    ) -> Dict[str, Any]:
        """Analyze how well internal and external findings integrate.

        Args:
            internal_results: Internal search results
            external_results: External web content

        Returns:
            Integration analysis
        """
        successful_external = [r for r in external_results if not r.error and r.content]

        analysis = {
            "data_sources": {
                "internal_sources": len(set(r.source for r in internal_results)),
                "external_domains": len(
                    set(
                        r.metadata.get("domain", "")
                        for r in successful_external
                        if r.metadata and "domain" in r.metadata
                    )
                ),
                "total_content_analyzed": len(internal_results)
                + len(successful_external),
            },
            "content_characteristics": {
                "internal_content_length": sum(
                    len(r.content) for r in internal_results
                ),
                "external_content_length": sum(
                    len(r.content) for r in successful_external
                ),
                "average_internal_relevance": (
                    sum(r.score for r in internal_results) / len(internal_results)
                    if internal_results
                    else 0
                ),
            },
            "integration_quality": {
                "balanced_sources": len(internal_results) > 0
                and len(successful_external) > 0,
                "sufficient_internal_depth": len(internal_results) >= 3,
                "sufficient_external_breadth": len(successful_external) >= 2,
                "comprehensive_analysis": (
                    len(internal_results) + len(successful_external)
                )
                >= 5,
            },
        }

        return analysis

    async def _generate_recommendations(
        self, research_question: str, doc_analysis, web_analysis
    ) -> List[str]:
        """Generate comprehensive recommendations based on research findings using LLM analysis.

        Args:
            research_question: Original research question
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            List of recommendations
        """
        try:
            # Try to generate content-based recommendations using LLM
            llm_recommendations = await self._generate_llm_recommendations(
                research_question, doc_analysis, web_analysis
            )
            if llm_recommendations:
                return llm_recommendations
        except Exception as e:
            self.logger.warning(
                f"⚠️ LLM recommendation generation failed: {e}, using fallback"
            )

        # Fallback to basic recommendations
        return await self._generate_basic_recommendations(
            research_question, doc_analysis, web_analysis
        )

    async def _generate_llm_recommendations(
        self, research_question: str, doc_analysis, web_analysis
    ) -> List[str]:
        """Generate recommendations using LLM analysis of actual content.

        Args:
            research_question: Original research question
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            List of LLM-generated recommendations
        """
        try:
            import os
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Collect actual content for analysis
            internal_content = []
            external_content = []

            if doc_analysis and doc_analysis.search_results:
                for result in doc_analysis.search_results[
                    :10
                ]:  # Top 10 internal results
                    if result.content and len(result.content.strip()) > 20:
                        internal_content.append(
                            f"Source: {result.source}\nContent: {result.content[:500]}..."
                        )

            if web_analysis:
                for content in (web_analysis.web_content + web_analysis.link_research)[
                    :10
                ]:  # Top 10 external
                    if (
                        content.content
                        and not content.error
                        and len(content.content.strip()) > 20
                    ):
                        external_content.append(
                            f"Source: {content.title or content.url}\nContent: {content.content[:500]}..."
                        )

            # Create comprehensive content summary
            content_summary = f"""Research Question: {research_question}

INTERNAL FINDINGS ({len(internal_content)} sources):
{chr(10).join(internal_content[:5])}

EXTERNAL FINDINGS ({len(external_content)} sources):
{chr(10).join(external_content[:5])}"""

            prompt = f"""Based on the research question and findings below, generate 5-7 specific, actionable recommendations.

{content_summary[:4000]}  # Limit to avoid token limits

Please provide recommendations that:
1. Address the specific research question directly
2. Synthesize insights from both internal and external sources
3. Identify key limitations or gaps revealed by the research
4. Suggest practical next steps or areas for further investigation
5. Highlight the most important findings that answer the research question

Format as a list with clear, actionable recommendations using relevant emojis."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research analyst expert at synthesizing findings and generating actionable recommendations based on comprehensive research analysis.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
            )

            recommendations_text = response.choices[0].message.content.strip()

            # Parse recommendations from response
            recommendations = []
            for line in recommendations_text.split("\n"):
                line = line.strip()
                if line and (
                    line.startswith("-")
                    or line.startswith("•")
                    or any(
                        emoji in line
                        for emoji in [
                            "✅",
                            "⚠️",
                            "🔍",
                            "📊",
                            "🎯",
                            "💡",
                            "⭐",
                            "🔬",
                            "📋",
                            "🌐",
                        ]
                    )
                ):
                    # Clean up the line
                    cleaned = line.lstrip("- •").strip()
                    if cleaned:
                        recommendations.append(cleaned)

            # Add basic metadata recommendations
            if doc_analysis and web_analysis:
                recommendations.append(
                    f"📊 Research completed with {len(doc_analysis.search_results)} internal and {len([w for w in web_analysis.web_content + web_analysis.link_research if not w.error])} external sources"
                )

            self.logger.info(
                f"✅ Generated {len(recommendations)} LLM-based recommendations"
            )
            return recommendations[:7]  # Limit to 7 recommendations

        except Exception as e:
            self.logger.error(f"Error generating LLM recommendations: {e}")
            raise

    async def _generate_basic_recommendations(
        self, research_question: str, doc_analysis, web_analysis
    ) -> List[str]:
        """Generate basic recommendations as fallback when LLM fails.

        Args:
            research_question: Original research question
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            List of basic recommendations
        """
        recommendations = []

        # Analyze research coverage and quality
        has_internal = doc_analysis and len(doc_analysis.search_results) > 0
        has_external = (
            web_analysis
            and len(
                [
                    w
                    for w in web_analysis.web_content + web_analysis.link_research
                    if not w.error
                ]
            )
            > 0
        )

        # Research completeness recommendations
        if has_internal and has_external:
            recommendations.append(
                "✅ Comprehensive research completed with both internal and external sources analyzed"
            )
        elif has_internal:
            recommendations.append(
                "⚠️ Consider supplementing with additional external research for broader perspective"
            )
        elif has_external:
            recommendations.append(
                "⚠️ Consider reviewing internal documentation for organizational context"
            )
        else:
            recommendations.append(
                "❌ Limited research results - consider refining search terms or expanding scope"
            )

        # Source quality recommendations
        if doc_analysis and doc_analysis.coverage_analysis:
            effectiveness = doc_analysis.coverage_analysis.get("query_effectiveness", 0)
            if effectiveness < 0.5:
                recommendations.append(
                    "🔍 Consider refining internal search queries for better coverage"
                )

        if web_analysis and web_analysis.research_summary:
            success_rate = web_analysis.research_summary.get("success_rate", 0)
            if success_rate < 0.4:
                recommendations.append(
                    "🌐 External research had limited success - consider alternative sources or approaches"
                )

        # Follow-up research recommendations
        if doc_analysis and doc_analysis.extracted_links:
            recommendations.append(
                f"🔗 {len(doc_analysis.extracted_links)} additional links identified for potential follow-up research"
            )

        # Specific question-type recommendations
        question_lower = research_question.lower()
        if "how" in question_lower and has_internal:
            recommendations.append(
                "📋 For implementation guidance, prioritize internal documentation and established procedures"
            )
        elif "what" in question_lower and has_external:
            recommendations.append(
                "📚 For definitions and concepts, external authoritative sources provide valuable context"
            )
        elif "compare" in question_lower:
            recommendations.append(
                "⚖️ Ensure comparison criteria are clearly defined for objective evaluation"
            )

        # General research quality recommendations
        recommendations.append(
            "📊 Cross-reference findings between internal and external sources for validation"
        )

        recommendations.append(
            "⏰ Consider the recency of information, especially for rapidly evolving topics"
        )

        return recommendations

    async def _generate_comprehensive_answer(
        self, research_question: str, doc_analysis, web_analysis
    ) -> str:
        """Generate a comprehensive answer to the research question using all findings.

        Args:
            research_question: Original research question
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            Comprehensive answer synthesizing all findings
        """
        try:
            import os
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            # Collect the best content for analysis
            internal_findings = []
            external_findings = []

            if doc_analysis and doc_analysis.search_results:
                # Get top scoring internal results
                sorted_results = sorted(
                    doc_analysis.search_results, key=lambda x: x.score, reverse=True
                )
                for result in sorted_results[:15]:  # Top 15 internal results
                    if result.content and len(result.content.strip()) > 30:
                        internal_findings.append(
                            {
                                "source": result.source,
                                "score": result.score,
                                "content": result.content[
                                    :800
                                ],  # More content for comprehensive analysis
                            }
                        )

            if web_analysis:
                # Get successful external content
                all_external = web_analysis.web_content + web_analysis.link_research
                successful_external = [
                    w
                    for w in all_external
                    if w.content and not w.error and len(w.content.strip()) > 30
                ]
                for content in successful_external[:10]:  # Top 10 external results
                    external_findings.append(
                        {
                            "source": content.title or content.url,
                            "content": content.content[:800],
                        }
                    )

            # Create comprehensive content for analysis
            internal_section = ""
            if internal_findings:
                internal_section = "INTERNAL RESEARCH FINDINGS:\n\n"
                for i, finding in enumerate(internal_findings[:8], 1):
                    internal_section += f"{i}. Source: {finding['source']} (Score: {finding['score']:.2f})\n"
                    internal_section += f"   Content: {finding['content']}\n\n"

            external_section = ""
            if external_findings:
                external_section = "EXTERNAL RESEARCH FINDINGS:\n\n"
                for i, finding in enumerate(external_findings[:6], 1):
                    external_section += f"{i}. Source: {finding['source']}\n"
                    external_section += f"   Content: {finding['content']}\n\n"

            content_for_analysis = f"{internal_section}{external_section}"

            prompt = f"""Research Question: {research_question}

Based on the comprehensive research findings below, provide a detailed, well-structured answer that directly addresses the research question.

{content_for_analysis[:6000]}  # Limit to avoid token limits

Please provide:
1. A direct answer to the research question
2. Key findings and evidence from the research
3. Analysis of the most important insights
4. Integration of internal and external findings
5. Clear, actionable conclusions

Structure your response with clear headings and bullet points where appropriate. Focus on the specific question asked and provide concrete, evidence-based insights."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research analyst expert at synthesizing complex research findings into clear, comprehensive answers. Provide detailed, evidence-based responses that directly address the research question.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=1500,
            )

            comprehensive_answer = response.choices[0].message.content.strip()

            # Add source summary
            source_summary = f"\n\n---\n**Research Sources:** {len(internal_findings)} internal documents, {len(external_findings)} external sources"
            comprehensive_answer += source_summary

            self.logger.info(
                f"✅ Generated comprehensive answer ({len(comprehensive_answer)} characters)"
            )
            return comprehensive_answer

        except Exception as e:
            self.logger.error(f"Error generating comprehensive answer: {e}")
            # Fallback to basic summary
            return await self._generate_basic_answer(
                research_question, doc_analysis, web_analysis
            )

    async def _generate_basic_answer(
        self, research_question: str, doc_analysis, web_analysis
    ) -> str:
        """Generate a basic answer when LLM analysis fails.

        Args:
            research_question: Original research question
            doc_analysis: Document analysis results
            web_analysis: Web research results

        Returns:
            Basic answer summarizing key findings
        """
        answer_parts = [f"# Answer to: {research_question}\n"]

        if doc_analysis and doc_analysis.key_findings:
            answer_parts.append("## Internal Research Findings:")
            for finding in doc_analysis.key_findings[:5]:
                answer_parts.append(f"- {finding}")
            answer_parts.append("")

        if web_analysis and web_analysis.key_insights:
            answer_parts.append("## External Research Insights:")
            for insight in web_analysis.key_insights[:5]:
                answer_parts.append(f"- {insight}")
            answer_parts.append("")

        # Add source counts
        internal_count = len(doc_analysis.search_results) if doc_analysis else 0
        external_count = (
            len(
                [
                    w
                    for w in (web_analysis.web_content + web_analysis.link_research)
                    if not w.error
                ]
            )
            if web_analysis
            else 0
        )

        answer_parts.append(
            f"**Research completed with {internal_count} internal and {external_count} external sources.**"
        )

        return "\n".join(answer_parts)
