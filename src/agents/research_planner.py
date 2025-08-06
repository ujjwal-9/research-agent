"""
Research planner agent for creating comprehensive research plans.
"""

from typing import Dict, Any, List
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentContext, AgentResult, AgentState
from src.tools.document_retriever import DocumentRetriever


@dataclass
class ResearchPlan:
    """Represents a comprehensive research plan."""

    research_question: str
    clarifying_questions: List[str]
    internal_search_queries: List[str]
    external_search_topics: List[str]
    expected_sources: List[str]
    research_methodology: str
    success_criteria: List[str]
    estimated_timeline: Dict[str, str]
    plan_rationale: str


class ResearchPlannerAgent(BaseAgent):
    """Agent responsible for creating detailed research plans."""

    def __init__(self):
        """Initialize the research planner agent."""
        super().__init__("research_planner")
        self.document_retriever = DocumentRetriever()

    async def execute(self, context: AgentContext) -> AgentResult:
        """Create a comprehensive research plan.

        Args:
            context: Shared context containing research question

        Returns:
            AgentResult containing the research plan
        """
        try:
            self.logger.info(
                f"🧠 Creating research plan for: {context.research_question}"
            )

            # Analyze the research question
            question_analysis = await self._analyze_research_question(
                context.research_question
            )

            # Identify potential ambiguities and create clarifying questions
            clarifying_questions = await self._generate_clarifying_questions(
                context.research_question, question_analysis
            )

            # Get available internal sources
            available_sources = self.document_retriever.get_all_sources()

            # Generate internal search queries
            internal_queries = await self._generate_internal_search_queries(
                context.research_question, available_sources
            )

            # Generate external research topics
            external_topics = await self._generate_external_search_topics(
                context.research_question, question_analysis
            )

            # Create methodology
            methodology = await self._create_research_methodology(
                context.research_question, internal_queries, external_topics
            )

            # Define success criteria
            success_criteria = await self._define_success_criteria(
                context.research_question
            )

            # Estimate timeline
            timeline = self._estimate_timeline(internal_queries, external_topics)

            # Create rationale
            rationale = await self._create_plan_rationale(
                context.research_question, internal_queries, external_topics
            )

            # Build research plan
            research_plan = ResearchPlan(
                research_question=context.research_question,
                clarifying_questions=clarifying_questions,
                internal_search_queries=internal_queries,
                external_search_topics=external_topics,
                expected_sources=available_sources,
                research_methodology=methodology,
                success_criteria=success_criteria,
                estimated_timeline=timeline,
                plan_rationale=rationale,
            )

            self.logger.info(
                f"✅ Created research plan with {len(internal_queries)} internal queries and {len(external_topics)} external topics"
            )

            return AgentResult(
                agent_name=self.name,
                status=AgentState.COMPLETED,
                data=research_plan,
                metadata={
                    "internal_queries_count": len(internal_queries),
                    "external_topics_count": len(external_topics),
                    "clarifying_questions_count": len(clarifying_questions),
                    "available_sources_count": len(available_sources),
                },
            )

        except Exception as e:
            self.logger.error(f"❌ Error creating research plan: {e}")
            return AgentResult(
                agent_name=self.name, status=AgentState.ERROR, data=None, error=str(e)
            )

    async def _analyze_research_question(self, question: str) -> Dict[str, Any]:
        """Analyze the research question to understand scope and requirements.

        Args:
            question: Research question to analyze

        Returns:
            Analysis results
        """
        # Analyze question characteristics
        analysis = {
            "question_type": self._classify_question_type(question),
            "key_concepts": self._extract_key_concepts(question),
            "scope": self._determine_scope(question),
            "complexity": self._assess_complexity(question),
            "domain": self._identify_domain(question),
        }

        return analysis

    def _classify_question_type(self, question: str) -> str:
        """Classify the type of research question.

        Args:
            question: Research question

        Returns:
            Question type classification
        """
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in ["how", "process", "procedure", "workflow"]
        ):
            return "procedural"
        elif any(
            word in question_lower for word in ["what", "define", "explain", "describe"]
        ):
            return "definitional"
        elif any(
            word in question_lower for word in ["why", "reason", "cause", "benefit"]
        ):
            return "causal"
        elif any(
            word in question_lower for word in ["compare", "difference", "versus", "vs"]
        ):
            return "comparative"
        elif any(
            word in question_lower for word in ["analyze", "assessment", "evaluation"]
        ):
            return "analytical"
        else:
            return "exploratory"

    def _extract_key_concepts(self, question: str) -> List[str]:
        """Extract key concepts from the research question.

        Args:
            question: Research question

        Returns:
            List of key concepts
        """
        # Simple keyword extraction (in a real implementation, you might use NLP)
        import re

        # Remove common question words
        common_words = {
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "which",
            "is",
            "are",
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "can",
            "could",
            "should",
            "would",
            "will",
            "shall",
            "may",
            "might",
            "must",
            "do",
            "does",
        }

        # Extract words and filter
        words = re.findall(r"\b[a-zA-Z]+\b", question.lower())
        concepts = [
            word for word in words if len(word) > 3 and word not in common_words
        ]

        # Remove duplicates while preserving order
        unique_concepts = []
        for concept in concepts:
            if concept not in unique_concepts:
                unique_concepts.append(concept)

        return unique_concepts[:10]  # Return top 10 concepts

    def _determine_scope(self, question: str) -> str:
        """Determine the scope of the research question.

        Args:
            question: Research question

        Returns:
            Scope classification
        """
        question_lower = question.lower()

        if any(
            word in question_lower
            for word in ["comprehensive", "complete", "all", "entire", "full"]
        ):
            return "comprehensive"
        elif any(
            word in question_lower
            for word in ["specific", "particular", "focused", "detailed"]
        ):
            return "focused"
        elif any(
            word in question_lower
            for word in ["overview", "summary", "general", "basic"]
        ):
            return "overview"
        else:
            return "moderate"

    def _assess_complexity(self, question: str) -> str:
        """Assess the complexity of the research question.

        Args:
            question: Research question

        Returns:
            Complexity level
        """
        # Simple heuristics for complexity assessment
        word_count = len(question.split())
        concept_count = len(self._extract_key_concepts(question))

        if word_count > 20 or concept_count > 8:
            return "high"
        elif word_count > 10 or concept_count > 4:
            return "medium"
        else:
            return "low"

    def _identify_domain(self, question: str) -> str:
        """Identify the domain/field of the research question.

        Args:
            question: Research question

        Returns:
            Domain classification
        """
        question_lower = question.lower()

        # Simple domain classification based on keywords
        if any(
            word in question_lower
            for word in ["technology", "software", "system", "algorithm", "data"]
        ):
            return "technology"
        elif any(
            word in question_lower
            for word in ["business", "market", "financial", "revenue", "cost"]
        ):
            return "business"
        elif any(
            word in question_lower
            for word in ["research", "study", "analysis", "academic"]
        ):
            return "academic"
        elif any(
            word in question_lower
            for word in ["process", "workflow", "procedure", "operation"]
        ):
            return "operational"
        else:
            return "general"

    async def _generate_clarifying_questions(
        self, question: str, analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate clarifying questions to address ambiguities.

        Args:
            question: Original research question
            analysis: Question analysis results

        Returns:
            List of clarifying questions
        """
        clarifying_questions = []

        # Generate questions based on question type and complexity
        if analysis["complexity"] == "high":
            clarifying_questions.append(
                "Would you like this research to focus on any specific aspect or subset?"
            )

        if analysis["scope"] == "comprehensive":
            clarifying_questions.append(
                "Are there any particular timeframes, regions, or contexts to prioritize?"
            )

        if analysis["question_type"] == "comparative":
            clarifying_questions.append(
                "What specific criteria should be used for comparison?"
            )

        if analysis["domain"] == "technology":
            clarifying_questions.append(
                "Are you interested in current implementations, future trends, or historical context?"
            )

        # Always ask about depth preference
        clarifying_questions.append(
            "Would you prefer a high-level overview or detailed technical analysis?"
        )

        return clarifying_questions

    async def _generate_internal_search_queries(
        self, question: str, available_sources: List[str]
    ) -> List[str]:
        """Generate search queries for internal document search using LLM.

        Args:
            question: Research question
            available_sources: List of available document sources

        Returns:
            List of search queries
        """
        try:
            import os
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = f"""Given the research question: "{question}"

Generate 10 diverse, specific search queries that would help find relevant documents to answer this question. 
The queries should be:
1. Semantically meaningful and specific to the domain
2. Different perspectives and angles on the topic
3. Include both broad and narrow queries
4. Use medical/technical terminology when appropriate
5. Focus on finding actual content rather than just definitions

Research Question: {question}

Generate exactly 10 search queries, one per line, without numbering or bullets:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a research librarian expert at generating precise search queries for academic and medical research.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )

            llm_queries = [
                query.strip()
                for query in response.choices[0].message.content.strip().split("\n")
                if query.strip()
            ]

            # Start with the original question
            queries = [question]

            # Add LLM-generated queries
            queries.extend(llm_queries[:10])

            # Add some fallback queries in case LLM fails
            key_concepts = self._extract_key_concepts(question)
            if len(key_concepts) >= 2:
                for i in range(min(3, len(key_concepts) - 1)):
                    combined = f"{key_concepts[i]} {key_concepts[i + 1]}"
                    if combined not in queries:
                        queries.append(combined)

            self.logger.info(f"✅ Generated {len(queries)} search queries using LLM")
            return queries[:15]  # Limit to 15 queries max

        except Exception as e:
            self.logger.warning(
                f"⚠️ LLM query generation failed: {e}, falling back to basic method"
            )
            return await self._generate_internal_search_queries_fallback(
                question, available_sources
            )

    async def _generate_internal_search_queries_fallback(
        self, question: str, available_sources: List[str]
    ) -> List[str]:
        """Fallback method for generating search queries when LLM fails.

        Args:
            question: Research question
            available_sources: List of available document sources

        Returns:
            List of search queries
        """
        queries = []
        key_concepts = self._extract_key_concepts(question)

        # Direct question as query
        queries.append(question)

        # Individual key concepts (only meaningful ones)
        for concept in key_concepts[:5]:
            if len(concept) > 4:  # Only use longer, more meaningful concepts
                queries.append(concept)

        # Concept combinations
        if len(key_concepts) >= 2:
            for i in range(len(key_concepts) - 1):
                combined = f"{key_concepts[i]} {key_concepts[i + 1]}"
                queries.append(combined)

        # Domain-specific queries
        question_lower = question.lower()
        if "how" in question_lower:
            queries.append(
                f"implementation of {key_concepts[0] if key_concepts else 'process'}"
            )
            queries.append(
                f"steps for {key_concepts[0] if key_concepts else 'procedure'}"
            )

        if "what" in question_lower:
            queries.append(
                f"definition of {key_concepts[0] if key_concepts else 'concept'}"
            )
            queries.append(
                f"overview of {key_concepts[0] if key_concepts else 'topic'}"
            )

        # Remove duplicates while preserving order
        unique_queries = []
        for query in queries:
            if query not in unique_queries:
                unique_queries.append(query)

        return unique_queries

    async def _generate_external_search_topics(
        self, question: str, analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate topics for external web research using LLM.

        Args:
            question: Research question
            analysis: Question analysis results

        Returns:
            List of external search topics
        """
        try:
            import os
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = f"""Given the research question: "{question}"

Generate 6-8 specific web search queries that would help find relevant external information to answer this question.
These should be suitable for web search engines (Google, PubMed, etc.) and should focus on:

1. Recent research publications and studies
2. Clinical guidelines and recommendations 
3. Medical literature and review articles
4. Expert opinions and professional guidelines
5. Current best practices and methodologies
6. Comparative studies and systematic reviews

Make the queries specific to the medical/scientific domain and use appropriate terminology.
Avoid generic queries - focus on finding authoritative, current information.

Research Question: {question}

Generate 6-8 web search queries, one per line, without numbering or bullets:"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a medical research librarian expert at generating precise web search queries for medical and scientific research.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=400,
            )

            llm_topics = [
                topic.strip()
                for topic in response.choices[0].message.content.strip().split("\n")
                if topic.strip()
            ]

            # Add some fallback topics in case LLM fails
            key_concepts = self._extract_key_concepts(question)
            fallback_topics = []
            if key_concepts:
                fallback_topics.extend(
                    [
                        f"clinical assessment {key_concepts[0] if len(key_concepts) > 0 else 'methods'}",
                        f"diagnostic methods {key_concepts[1] if len(key_concepts) > 1 else 'medical'}",
                    ]
                )

            topics = llm_topics[:8] + fallback_topics[:2]
            self.logger.info(
                f"✅ Generated {len(topics)} external search topics using LLM"
            )
            return topics[:8]  # Limit to 8 topics max

        except Exception as e:
            self.logger.warning(
                f"⚠️ LLM topic generation failed: {e}, falling back to basic method"
            )
            return await self._generate_external_search_topics_fallback(
                question, analysis
            )

    async def _generate_external_search_topics_fallback(
        self, question: str, analysis: Dict[str, Any]
    ) -> List[str]:
        """Fallback method for generating external search topics when LLM fails.

        Args:
            question: Research question
            analysis: Question analysis results

        Returns:
            List of external search topics
        """
        topics = []
        key_concepts = self._extract_key_concepts(question)

        # Medical/scientific focus with meaningful concepts only
        meaningful_concepts = [c for c in key_concepts if len(c) > 4]

        if meaningful_concepts:
            topics.append(f"{meaningful_concepts[0]} research latest")
            topics.append(f"clinical {meaningful_concepts[0]} guidelines")

        # Best practices and guides
        if analysis["question_type"] == "procedural":
            topics.append(
                f"best practices for {meaningful_concepts[0] if meaningful_concepts else 'clinical assessment'}"
            )
            topics.append(
                f"clinical guidelines {meaningful_concepts[0] if meaningful_concepts else 'medical'}"
            )

        # Academic and research perspectives
        if analysis["complexity"] == "high":
            topics.append(
                f"systematic review {meaningful_concepts[0] if meaningful_concepts else 'medical'}"
            )
            topics.append(
                f"meta analysis {meaningful_concepts[0] if meaningful_concepts else 'clinical'}"
            )

        # Case studies and examples
        topics.append(
            f"clinical studies {meaningful_concepts[0] if meaningful_concepts else 'medical'}"
        )
        topics.append(
            f"evidence based {meaningful_concepts[0] if meaningful_concepts else 'medicine'}"
        )

        return topics

    async def _create_research_methodology(
        self, question: str, internal_queries: List[str], external_topics: List[str]
    ) -> str:
        """Create a research methodology description.

        Args:
            question: Research question
            internal_queries: Internal search queries
            external_topics: External search topics

        Returns:
            Methodology description
        """
        methodology = f"""
**Research Methodology for: "{question}"**

1. **Internal Knowledge Base Analysis**
   - Execute {len(internal_queries)} targeted semantic searches across internal documents
   - Focus on relevant documents and extract supporting evidence
   - Analyze content for direct answers and related information

2. **External Web Research**
   - Research {len(external_topics)} external topics to gather current information
   - Follow extracted links from internal documents for additional context
   - Collect information from authoritative sources and recent publications

3. **Cross-Reference Analysis**
   - Compare internal knowledge with external findings
   - Identify gaps, contradictions, or confirmations
   - Synthesize comprehensive understanding

4. **Evidence Compilation**
   - Organize findings by relevance and authority
   - Create structured evidence base with source attribution
   - Prioritize most recent and authoritative information

5. **Report Synthesis**
   - Combine all research findings into coherent analysis
   - Address original research question with supporting evidence
   - Provide recommendations and identify areas for further research
        """.strip()

        return methodology

    async def _define_success_criteria(self, question: str) -> List[str]:
        """Define success criteria for the research.

        Args:
            question: Research question

        Returns:
            List of success criteria
        """
        criteria = [
            "Comprehensive answer to the original research question",
            "Supporting evidence from multiple authoritative sources",
            "Both internal and external perspectives included",
            "Current and up-to-date information gathered",
            "Clear identification of key findings and insights",
        ]

        # Add question-specific criteria
        question_lower = question.lower()

        if "how" in question_lower:
            criteria.append("Step-by-step process or implementation guidance provided")

        if "compare" in question_lower or "versus" in question_lower:
            criteria.append("Clear comparison framework with objective criteria")

        if "best" in question_lower or "optimal" in question_lower:
            criteria.append("Evaluation of alternatives with recommendations")

        return criteria

    def _estimate_timeline(
        self, internal_queries: List[str], external_topics: List[str]
    ) -> Dict[str, str]:
        """Estimate timeline for research execution.

        Args:
            internal_queries: Internal search queries
            external_topics: External search topics

        Returns:
            Timeline estimation
        """
        # Simple estimation based on query/topic counts
        internal_time = len(internal_queries) * 0.5  # 30 seconds per query
        external_time = len(external_topics) * 2  # 2 minutes per topic
        synthesis_time = 3  # 3 minutes for synthesis

        total_minutes = internal_time + external_time + synthesis_time

        timeline = {
            "internal_research": f"{internal_time:.1f} minutes",
            "external_research": f"{external_time:.1f} minutes",
            "synthesis": f"{synthesis_time} minutes",
            "total_estimated": f"{total_minutes:.1f} minutes",
        }

        return timeline

    async def _create_plan_rationale(
        self, question: str, internal_queries: List[str], external_topics: List[str]
    ) -> str:
        """Create rationale for the research plan.

        Args:
            question: Research question
            internal_queries: Internal search queries
            external_topics: External search topics

        Returns:
            Plan rationale
        """
        rationale = f"""
This research plan is designed to provide a comprehensive answer to: "{question}"

**Internal Research Approach:**
- {len(internal_queries)} targeted queries will search our knowledge base for existing relevant information
- Queries are designed to capture both direct matches and related concepts
- Internal sources provide our organization's current understanding and documented practices

**External Research Approach:**
- {len(external_topics)} external topics will gather current industry perspectives and best practices
- Focus on authoritative sources and recent developments
- External research ensures our findings are current and comprehensive

**Integration Strategy:**
- Internal and external findings will be cross-referenced for validation
- Gaps in internal knowledge will be identified and addressed through external research
- Final synthesis will provide actionable insights based on best available evidence

This approach ensures both depth (from internal expertise) and breadth (from external sources) while maintaining focus on answering the specific research question.
        """.strip()

        return rationale
