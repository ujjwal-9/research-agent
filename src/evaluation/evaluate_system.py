"""System evaluation framework for assessing research quality."""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from src.agents.research_orchestrator import ResearchOrchestrator
from src.ingestion.document_store import DocumentStore
from loguru import logger


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating research quality."""
    query: str
    response_time: float
    document_coverage: float  # Percentage of relevant docs found
    source_diversity: float   # Variety of sources used
    answer_completeness: float  # How complete the answer is
    factual_accuracy: float   # Accuracy of facts (if verifiable)
    coherence_score: float    # How well-structured the response is
    confidence_score: float   # System's confidence in the answer
    user_satisfaction: Optional[float] = None  # User rating if available


@dataclass
class EvaluationResult:
    """Result of system evaluation."""
    test_queries: List[str]
    metrics: List[EvaluationMetrics]
    average_metrics: Dict[str, float]
    recommendations: List[str]


class SystemEvaluator:
    """Evaluates the research system performance."""
    
    def __init__(self):
        self.orchestrator = ResearchOrchestrator()
        self.document_store = DocumentStore()
    
    async def evaluate_query_set(self, queries: List[str]) -> EvaluationResult:
        """Evaluate system performance on a set of queries."""
        logger.info(f"Starting evaluation with {len(queries)} queries")
        
        metrics = []
        for i, query in enumerate(queries, 1):
            logger.info(f"Evaluating query {i}/{len(queries)}: {query[:50]}...")
            
            metric = await self._evaluate_single_query(query)
            metrics.append(metric)
        
        # Calculate average metrics
        average_metrics = self._calculate_average_metrics(metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics)
        
        return EvaluationResult(
            test_queries=queries,
            metrics=metrics,
            average_metrics=average_metrics,
            recommendations=recommendations
        )
    
    async def _evaluate_single_query(self, query: str) -> EvaluationMetrics:
        """Evaluate system performance on a single query."""
        start_time = time.time()
        
        # Execute research
        session = await self.orchestrator.run_research(query)
        
        response_time = time.time() - start_time
        
        # Calculate metrics
        document_coverage = self._calculate_document_coverage(session)
        source_diversity = self._calculate_source_diversity(session)
        answer_completeness = await self._assess_answer_completeness(query, session.final_report)
        factual_accuracy = await self._assess_factual_accuracy(session.final_report)
        coherence_score = await self._assess_coherence(session.final_report)
        confidence_score = self._extract_confidence_score(session)
        
        return EvaluationMetrics(
            query=query,
            response_time=response_time,
            document_coverage=document_coverage,
            source_diversity=source_diversity,
            answer_completeness=answer_completeness,
            factual_accuracy=factual_accuracy,
            coherence_score=coherence_score,
            confidence_score=confidence_score
        )
    
    def _calculate_document_coverage(self, session) -> float:
        """Calculate what percentage of relevant documents were found."""
        if not session.research_results:
            return 0.0
        
        # Simple heuristic: more internal results = better coverage
        internal_results = [r for r in session.research_results if r.source_type == "internal"]
        
        if not internal_results:
            return 0.0
        
        total_docs = self.document_store.get_document_count()
        if total_docs == 0:
            return 1.0
        
        # Calculate based on number of unique documents found
        unique_docs = set()
        for result in internal_results:
            for res in result.results:
                file_path = res.get("metadata", {}).get("file_path")
                if file_path:
                    unique_docs.add(file_path)
        
        coverage = min(len(unique_docs) / max(total_docs * 0.1, 1), 1.0)  # Assume 10% relevance
        return round(coverage, 2)
    
    def _calculate_source_diversity(self, session) -> float:
        """Calculate diversity of sources used."""
        if not session.research_results:
            return 0.0
        
        source_types = set()
        for result in session.research_results:
            source_types.add(result.source_type)
            
            # Count different file types for internal sources
            if result.source_type == "internal":
                for res in result.results:
                    file_type = res.get("metadata", {}).get("file_type")
                    if file_type:
                        source_types.add(f"internal_{file_type}")
        
        # Normalize by expected maximum diversity
        max_diversity = 6  # internal_docx, internal_pdf, internal_xlsx, external, etc.
        diversity = min(len(source_types) / max_diversity, 1.0)
        return round(diversity, 2)
    
    async def _assess_answer_completeness(self, query: str, report: str) -> float:
        """Assess how complete the answer is using LLM."""
        if not report:
            return 0.0
        
        prompt = f"""
        Evaluate the completeness of this research report for the given query.
        
        Query: {query}
        
        Report: {report[:2000]}...
        
        Rate the completeness on a scale of 0.0 to 1.0 where:
        - 1.0 = Fully addresses all aspects of the query
        - 0.8 = Addresses most aspects with minor gaps
        - 0.6 = Addresses main aspects but missing some important details
        - 0.4 = Partially addresses the query
        - 0.2 = Minimal coverage of the query
        - 0.0 = Does not address the query
        
        Respond with only a number between 0.0 and 1.0.
        """
        
        try:
            from src.agents.base_agent import BaseAgent, AgentRole
            evaluator = BaseAgent("evaluator", AgentRole.VALIDATOR)
            
            response = await evaluator.call_llm([
                {"role": "system", "content": "You are an expert evaluator of research reports."},
                {"role": "user", "content": prompt}
            ])
            
            score = float(response.strip())
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error assessing completeness: {e}")
            return 0.5  # Default moderate score
    
    async def _assess_factual_accuracy(self, report: str) -> float:
        """Assess factual accuracy of the report."""
        if not report:
            return 0.0
        
        # Simple heuristic: look for confidence indicators, citations, etc.
        accuracy_indicators = [
            "according to",
            "based on",
            "source:",
            "reference:",
            "study shows",
            "research indicates"
        ]
        
        uncertainty_indicators = [
            "might",
            "could",
            "possibly",
            "unclear",
            "uncertain"
        ]
        
        report_lower = report.lower()
        
        accuracy_count = sum(1 for indicator in accuracy_indicators if indicator in report_lower)
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in report_lower)
        
        # Simple scoring based on indicators
        base_score = 0.7  # Default moderate accuracy
        accuracy_boost = min(accuracy_count * 0.05, 0.2)
        uncertainty_penalty = min(uncertainty_count * 0.03, 0.15)
        
        score = base_score + accuracy_boost - uncertainty_penalty
        return round(max(0.0, min(1.0, score)), 2)
    
    async def _assess_coherence(self, report: str) -> float:
        """Assess coherence and structure of the report."""
        if not report:
            return 0.0
        
        # Simple heuristics for coherence
        structure_indicators = [
            "# ",  # Headers
            "## ",
            "### ",
            "1. ",  # Lists
            "2. ",
            "• ",
            "- ",
            "executive summary",
            "conclusion",
            "recommendations"
        ]
        
        report_lower = report.lower()
        structure_count = sum(1 for indicator in structure_indicators if indicator in report_lower)
        
        # Check for logical flow
        has_intro = any(word in report_lower[:200] for word in ["overview", "summary", "introduction"])
        has_conclusion = any(word in report_lower[-200:] for word in ["conclusion", "summary", "recommendations"])
        
        base_score = 0.6
        structure_boost = min(structure_count * 0.03, 0.2)
        flow_boost = 0.1 if has_intro and has_conclusion else 0.0
        
        score = base_score + structure_boost + flow_boost
        return round(max(0.0, min(1.0, score)), 2)
    
    def _extract_confidence_score(self, session) -> float:
        """Extract confidence score from research session."""
        if not session.research_results:
            return 0.0
        
        # Average confidence from all research results
        confidence_scores = [r.confidence for r in session.research_results if r.confidence is not None]
        
        if not confidence_scores:
            return 0.5
        
        return round(sum(confidence_scores) / len(confidence_scores), 2)
    
    def _calculate_average_metrics(self, metrics: List[EvaluationMetrics]) -> Dict[str, float]:
        """Calculate average metrics across all queries."""
        if not metrics:
            return {}
        
        avg_metrics = {}
        metric_fields = [
            "response_time", "document_coverage", "source_diversity",
            "answer_completeness", "factual_accuracy", "coherence_score", "confidence_score"
        ]
        
        for field in metric_fields:
            values = [getattr(m, field) for m in metrics if getattr(m, field) is not None]
            if values:
                avg_metrics[field] = round(sum(values) / len(values), 3)
        
        return avg_metrics
    
    def _generate_recommendations(self, metrics: List[EvaluationMetrics]) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        if not metrics:
            return ["No metrics available for recommendations"]
        
        avg_metrics = self._calculate_average_metrics(metrics)
        
        # Response time recommendations
        if avg_metrics.get("response_time", 0) > 30:
            recommendations.append("Consider optimizing response time - average is over 30 seconds")
        
        # Document coverage recommendations
        if avg_metrics.get("document_coverage", 0) < 0.3:
            recommendations.append("Improve document indexing and search relevance")
        
        # Source diversity recommendations
        if avg_metrics.get("source_diversity", 0) < 0.5:
            recommendations.append("Enhance source diversity by improving external search capabilities")
        
        # Answer completeness recommendations
        if avg_metrics.get("answer_completeness", 0) < 0.7:
            recommendations.append("Improve answer completeness through better synthesis and planning")
        
        # Factual accuracy recommendations
        if avg_metrics.get("factual_accuracy", 0) < 0.8:
            recommendations.append("Enhance factual accuracy through better source validation")
        
        # Coherence recommendations
        if avg_metrics.get("coherence_score", 0) < 0.7:
            recommendations.append("Improve report structure and coherence in synthesis phase")
        
        if not recommendations:
            recommendations.append("System performance is good across all metrics")
        
        return recommendations


# Default test queries for evaluation
DEFAULT_TEST_QUERIES = [
    "What are the main challenges in clinical trial communication?",
    "Summarize the key findings from the UPMC pilot study",
    "What are the financial projections for the Saki product?",
    "Describe the competitive landscape analysis",
    "What are the main consumer insights from the research?",
    "Summarize the investor sentiment findings",
    "What are the key UX design recommendations?",
    "Describe the go-to-market strategy insights"
]


async def main():
    """Run system evaluation."""
    evaluator = SystemEvaluator()
    
    # Use default queries or load from file
    test_queries = DEFAULT_TEST_QUERIES
    
    print("Starting system evaluation...")
    result = await evaluator.evaluate_query_set(test_queries)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print(f"Queries evaluated: {len(result.test_queries)}")
    print("\nAverage Metrics:")
    for metric, value in result.average_metrics.items():
        print(f"  {metric}: {value}")
    
    print("\nRecommendations:")
    for rec in result.recommendations:
        print(f"  • {rec}")
    
    # Save detailed results
    output_file = Path("evaluation_results.json")
    with open(output_file, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())