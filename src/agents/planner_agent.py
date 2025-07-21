"""Planning agent for research workflow orchestration."""

from typing import Dict, Any, List, Optional
import json
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent, AgentRole
from loguru import logger


@dataclass
class ResearchPlan:
    """Represents a comprehensive research plan."""
    query: str
    clarifications_needed: List[str]
    research_objectives: List[str]
    internal_search_tasks: List[Dict[str, Any]]
    external_search_tasks: List[Dict[str, Any]]
    synthesis_requirements: List[str]
    estimated_time: str
    success_criteria: List[str]


class PlannerAgent(BaseAgent):
    """Agent responsible for creating and managing research plans."""
    
    def __init__(self, agent_id: str = "planner"):
        super().__init__(agent_id, AgentRole.PLANNER)
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a planning task."""
        task_type = task.get("type")
        
        if task_type == "create_plan":
            return await self._create_research_plan(task["query"], task.get("context", {}))
        elif task_type == "refine_plan":
            return await self._refine_research_plan(task["plan"], task["feedback"])
        elif task_type == "validate_plan":
            return await self._validate_research_plan(task["plan"])
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _create_research_plan(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive research plan for the given query."""
        await self.update_status("creating_plan", f"Planning research for: {query[:50]}...")
        
        # Analyze the query and create initial plan
        analysis_prompt = self._get_query_analysis_prompt(query, context)
        analysis_response = await self.call_llm([
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": analysis_prompt}
        ])
        
        try:
            # Parse the analysis response
            analysis = json.loads(analysis_response)
            
            # Create detailed research plan
            plan_prompt = self._get_plan_creation_prompt(query, analysis)
            plan_response = await self.call_llm([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": plan_prompt}
            ])
            
            plan_data = json.loads(plan_response)
            
            # Create ResearchPlan object
            research_plan = ResearchPlan(
                query=query,
                clarifications_needed=analysis.get("clarifications_needed", []),
                research_objectives=plan_data.get("research_objectives", []),
                internal_search_tasks=plan_data.get("internal_search_tasks", []),
                external_search_tasks=plan_data.get("external_search_tasks", []),
                synthesis_requirements=plan_data.get("synthesis_requirements", []),
                estimated_time=plan_data.get("estimated_time", "Unknown"),
                success_criteria=plan_data.get("success_criteria", [])
            )
            
            logger.info(f"Created research plan with {len(research_plan.internal_search_tasks)} internal and {len(research_plan.external_search_tasks)} external tasks")
            
            return {
                "success": True,
                "plan": research_plan,
                "requires_clarification": len(research_plan.clarifications_needed) > 0
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing LLM response: {e}")
            return {"success": False, "error": "Failed to parse planning response"}
        except Exception as e:
            logger.error(f"Error creating research plan: {e}")
            return {"success": False, "error": str(e)}
    
    async def _refine_research_plan(self, plan: ResearchPlan, feedback: str) -> Dict[str, Any]:
        """Refine an existing research plan based on feedback."""
        await self.update_status("refining_plan", "Refining research plan based on feedback")
        
        refinement_prompt = f"""
        Current research plan:
        Query: {plan.query}
        Objectives: {plan.research_objectives}
        Internal tasks: {plan.internal_search_tasks}
        External tasks: {plan.external_search_tasks}
        
        User feedback: {feedback}
        
        Please refine the research plan based on the feedback. Return a JSON object with the updated plan structure.
        """
        
        try:
            response = await self.call_llm([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": refinement_prompt}
            ])
            
            refined_data = json.loads(response)
            
            # Update the plan
            plan.research_objectives = refined_data.get("research_objectives", plan.research_objectives)
            plan.internal_search_tasks = refined_data.get("internal_search_tasks", plan.internal_search_tasks)
            plan.external_search_tasks = refined_data.get("external_search_tasks", plan.external_search_tasks)
            plan.synthesis_requirements = refined_data.get("synthesis_requirements", plan.synthesis_requirements)
            
            return {"success": True, "plan": plan}
            
        except Exception as e:
            logger.error(f"Error refining research plan: {e}")
            return {"success": False, "error": str(e)}
    
    async def _validate_research_plan(self, plan: ResearchPlan) -> Dict[str, Any]:
        """Validate a research plan for completeness and feasibility."""
        await self.update_status("validating_plan", "Validating research plan")
        
        validation_issues = []
        
        # Check for essential components
        if not plan.research_objectives:
            validation_issues.append("No research objectives defined")
        
        if not plan.internal_search_tasks and not plan.external_search_tasks:
            validation_issues.append("No search tasks defined")
        
        if not plan.synthesis_requirements:
            validation_issues.append("No synthesis requirements specified")
        
        # Validate task structure
        for task in plan.internal_search_tasks:
            if not task.get("query") or not task.get("description"):
                validation_issues.append(f"Incomplete internal search task: {task}")
        
        for task in plan.external_search_tasks:
            if not task.get("query") or not task.get("description"):
                validation_issues.append(f"Incomplete external search task: {task}")
        
        is_valid = len(validation_issues) == 0
        
        return {
            "success": True,
            "is_valid": is_valid,
            "issues": validation_issues,
            "plan": plan
        }
    
    def _get_query_analysis_prompt(self, query: str, context: Dict[str, Any]) -> str:
        """Generate prompt for query analysis."""
        return f"""
        Analyze the following research query and identify any ambiguities or clarifications needed:
        
        Query: "{query}"
        Context: {json.dumps(context, indent=2)}
        
        Please return a JSON object with the following structure:
        {{
            "query_type": "descriptive|analytical|comparative|evaluative",
            "scope": "broad|narrow|specific",
            "complexity": "low|medium|high",
            "clarifications_needed": ["list of questions to ask user"],
            "key_concepts": ["list of key concepts to research"],
            "potential_sources": ["types of sources that might be relevant"]
        }}
        """
    
    def _get_plan_creation_prompt(self, query: str, analysis: Dict[str, Any]) -> str:
        """Generate prompt for creating detailed research plan."""
        return f"""
        Create a detailed research plan for the following query:
        
        Query: "{query}"
        Analysis: {json.dumps(analysis, indent=2)}
        
        Please return a JSON object with the following structure:
        {{
            "research_objectives": ["specific, measurable objectives"],
            "internal_search_tasks": [
                {{
                    "id": "task_id",
                    "query": "search query for internal documents",
                    "description": "what this task aims to find",
                    "file_types": ["docx", "pdf", "xlsx"],
                    "priority": "high|medium|low"
                }}
            ],
            "external_search_tasks": [
                {{
                    "id": "task_id", 
                    "query": "search query for web search",
                    "description": "what this task aims to find",
                    "sources": ["academic", "news", "industry"],
                    "priority": "high|medium|low"
                }}
            ],
            "synthesis_requirements": ["how to combine and present findings"],
            "estimated_time": "estimated completion time",
            "success_criteria": ["criteria for successful completion"]
        }}
        
        Ensure tasks are specific, actionable, and can be executed in parallel where possible.
        """