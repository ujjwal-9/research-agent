"""Synthesizer agent for combining research results into coherent reports."""

from typing import Dict, Any, List
import json
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent, AgentRole
from loguru import logger


@dataclass
class SynthesisResult:
    """Result of synthesis process."""
    report: str
    executive_summary: str
    key_findings: List[str]
    sources_used: List[str]
    confidence_score: float
    recommendations: List[str]


class SynthesizerAgent(BaseAgent):
    """Agent responsible for synthesizing research results into coherent reports."""
    
    def __init__(self, agent_id: str = "synthesizer"):
        super().__init__(agent_id, AgentRole.SYNTHESIZER)
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a synthesis task."""
        task_type = task.get("type")
        
        if task_type == "synthesize_report":
            return await self._synthesize_research_report(
                task["query"],
                task["plan"],
                task["research_results"]
            )
        elif task_type == "create_executive_summary":
            return await self._create_executive_summary(task["content"])
        elif task_type == "extract_key_findings":
            return await self._extract_key_findings(task["research_results"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _synthesize_research_report(self, query: str, plan, research_results: List[Any]) -> Dict[str, Any]:
        """Synthesize research results into a comprehensive report."""
        await self.update_status("synthesizing", "Creating comprehensive research report")
        
        try:
            # Organize results by source type
            internal_results = [r for r in research_results if r.source_type == "internal"]
            external_results = [r for r in research_results if r.source_type == "external"]
            
            # Extract key information
            key_findings = await self._extract_key_findings_from_results(research_results)
            sources_used = self._extract_sources(research_results)
            
            # Calculate overall confidence
            confidence_score = self._calculate_overall_confidence(research_results)
            
            # Generate main report
            report_content = await self._generate_main_report(
                query, plan, internal_results, external_results, key_findings
            )
            
            # Generate executive summary
            executive_summary = await self._generate_executive_summary(report_content, key_findings)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(query, key_findings, research_results)
            
            # Create final synthesis result
            synthesis_result = SynthesisResult(
                report=report_content,
                executive_summary=executive_summary,
                key_findings=key_findings,
                sources_used=sources_used,
                confidence_score=confidence_score,
                recommendations=recommendations
            )
            
            # Format final report
            final_report = self._format_final_report(synthesis_result)
            
            logger.info(f"Successfully synthesized research report with {len(key_findings)} key findings")
            
            return {
                "success": True,
                "report": final_report,
                "synthesis_result": synthesis_result
            }
            
        except Exception as e:
            logger.error(f"Error synthesizing research report: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _extract_key_findings_from_results(self, research_results: List[Any]) -> List[str]:
        """Extract key findings from research results."""
        if not research_results:
            return []
        
        # Combine all summaries
        summaries = [result.summary for result in research_results if result.summary]
        combined_summaries = "\\n\\n".join(summaries)
        
        prompt = f"""
        Extract the key findings from the following research summaries:
        
        {combined_summaries}
        
        Please provide a JSON list of the most important findings, each as a concise statement.
        Focus on factual information, insights, and significant discoveries.
        
        Format: ["finding 1", "finding 2", "finding 3", ...]
        """
        
        try:
            response = await self.call_llm([
                {"role": "system", "content": "You are a research analyst extracting key findings."},
                {"role": "user", "content": prompt}
            ])
            
            findings = json.loads(response)
            return findings if isinstance(findings, list) else []
            
        except Exception as e:
            logger.error(f"Error extracting key findings: {e}")
            return ["Unable to extract key findings due to processing error"]
    
    def _extract_sources(self, research_results: List[Any]) -> List[str]:
        """Extract unique sources from research results."""
        sources = set()
        
        for result in research_results:
            if result.source_type == "internal":
                # Extract file paths from internal results
                for res in result.results:
                    file_path = res.get("metadata", {}).get("file_path", "")
                    if file_path:
                        sources.add(f"Internal: {file_path}")
            else:
                # Extract URLs from external results
                for res in result.results:
                    url = res.get("url", "")
                    if url:
                        sources.add(f"External: {url}")
        
        return list(sources)
    
    def _calculate_overall_confidence(self, research_results: List[Any]) -> float:
        """Calculate overall confidence score for the research."""
        if not research_results:
            return 0.0
        
        # Average confidence scores from all results
        confidence_scores = [result.confidence for result in research_results if result.confidence is not None]
        
        if not confidence_scores:
            return 0.5  # Default moderate confidence
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Adjust based on number of sources
        source_factor = min(len(research_results) / 5.0, 1.0)  # More sources = higher confidence
        
        return round(avg_confidence * source_factor, 2)
    
    async def _generate_main_report(self, query: str, plan, internal_results: List[Any], 
                                  external_results: List[Any], key_findings: List[str]) -> str:
        """Generate the main research report content."""
        
        # Prepare content sections
        internal_content = self._format_internal_results(internal_results)
        external_content = self._format_external_results(external_results)
        
        prompt = f"""
        Create a comprehensive research report based on the following information:
        
        Research Query: {query}
        
        Research Objectives:
        {chr(10).join(f"• {obj}" for obj in plan.research_objectives)}
        
        Key Findings:
        {chr(10).join(f"• {finding}" for finding in key_findings)}
        
        Internal Document Analysis:
        {internal_content}
        
        External Source Analysis:
        {external_content}
        
        Please create a well-structured research report that:
        1. Addresses the original research query comprehensively
        2. Integrates findings from both internal and external sources
        3. Provides clear analysis and insights
        4. Maintains academic rigor and objectivity
        5. Uses proper citations and references
        
        Structure the report with clear sections and subsections.
        """
        
        try:
            report = await self.call_llm([
                {"role": "system", "content": "You are a senior research analyst creating comprehensive reports."},
                {"role": "user", "content": prompt}
            ], max_tokens=3000)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating main report: {e}")
            return f"Error generating report: {str(e)}"
    
    async def _generate_executive_summary(self, report_content: str, key_findings: List[str]) -> str:
        """Generate executive summary of the report."""
        prompt = f"""
        Create a concise executive summary for the following research report:
        
        Report Content:
        {report_content[:2000]}...
        
        Key Findings:
        {chr(10).join(f"• {finding}" for finding in key_findings)}
        
        The executive summary should be 2-3 paragraphs and highlight:
        1. The main research question addressed
        2. Key findings and insights
        3. Primary conclusions and implications
        """
        
        try:
            summary = await self.call_llm([
                {"role": "system", "content": "You are creating an executive summary for a research report."},
                {"role": "user", "content": prompt}
            ], max_tokens=500)
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return "Executive summary could not be generated due to processing error."
    
    async def _generate_recommendations(self, query: str, key_findings: List[str], 
                                      research_results: List[Any]) -> List[str]:
        """Generate actionable recommendations based on research findings."""
        prompt = f"""
        Based on the research query and findings, provide actionable recommendations:
        
        Research Query: {query}
        
        Key Findings:
        {chr(10).join(f"• {finding}" for finding in key_findings)}
        
        Please provide a JSON list of specific, actionable recommendations.
        Each recommendation should be practical and directly related to the research findings.
        
        Format: ["recommendation 1", "recommendation 2", "recommendation 3", ...]
        """
        
        try:
            response = await self.call_llm([
                {"role": "system", "content": "You are providing actionable recommendations based on research."},
                {"role": "user", "content": prompt}
            ])
            
            recommendations = json.loads(response)
            return recommendations if isinstance(recommendations, list) else []
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Unable to generate recommendations due to processing error"]
    
    def _format_internal_results(self, internal_results: List[Any]) -> str:
        """Format internal research results for report inclusion."""
        if not internal_results:
            return "No internal documents were found relevant to this query."
        
        formatted_sections = []
        for result in internal_results:
            section = f"Task: {result.query}\\n"
            section += f"Summary: {result.summary}\\n"
            section += f"Documents found: {len(result.results)}\\n"
            section += f"Confidence: {result.confidence}\\n"
            formatted_sections.append(section)
        
        return "\\n\\n".join(formatted_sections)
    
    def _format_external_results(self, external_results: List[Any]) -> str:
        """Format external research results for report inclusion."""
        if not external_results:
            return "No external sources were searched for this query."
        
        formatted_sections = []
        for result in external_results:
            section = f"Search: {result.query}\\n"
            section += f"Summary: {result.summary}\\n"
            section += f"Sources found: {len(result.results)}\\n"
            section += f"Confidence: {result.confidence}\\n"
            formatted_sections.append(section)
        
        return "\\n\\n".join(formatted_sections)
    
    def _format_final_report(self, synthesis_result: SynthesisResult) -> str:
        """Format the final research report."""
        report = f"""
# Research Report

## Executive Summary
{synthesis_result.executive_summary}

## Key Findings
{chr(10).join(f"• {finding}" for finding in synthesis_result.key_findings)}

## Detailed Analysis
{synthesis_result.report}

## Recommendations
{chr(10).join(f"• {rec}" for rec in synthesis_result.recommendations)}

## Sources
{chr(10).join(f"• {source}" for source in synthesis_result.sources_used)}

## Confidence Assessment
Overall confidence in findings: {synthesis_result.confidence_score:.1%}

---
*This report was generated by an AI research system combining internal document analysis with external source research.*
"""
        return report.strip()
    
    async def _create_executive_summary(self, content: str) -> Dict[str, Any]:
        """Create executive summary from content."""
        summary = await self._generate_executive_summary(content, [])
        return {"success": True, "summary": summary}
    
    async def _extract_key_findings(self, research_results: List[Any]) -> Dict[str, Any]:
        """Extract key findings from research results."""
        findings = await self._extract_key_findings_from_results(research_results)
        return {"success": True, "findings": findings}