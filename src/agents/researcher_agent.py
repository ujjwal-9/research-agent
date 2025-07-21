"""Research agent for executing search tasks."""

from typing import Dict, Any, List, Optional
import asyncio
from dataclasses import dataclass

from src.agents.base_agent import BaseAgent, AgentRole
from src.ingestion.document_store import DocumentStore
from src.tools.web_search import WebSearchTool
from loguru import logger


@dataclass
class ResearchResult:
    """Represents the result of a research task."""
    task_id: str
    query: str
    source_type: str  # "internal" or "external"
    results: List[Dict[str, Any]]
    summary: str
    confidence: float
    execution_time: float


class ResearcherAgent(BaseAgent):
    """Agent responsible for executing research tasks."""
    
    def __init__(self, agent_id: str = "researcher"):
        super().__init__(agent_id, AgentRole.RESEARCHER)
        self.document_store = DocumentStore()
        self.web_search = WebSearchTool()
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research task."""
        task_type = task.get("type")
        
        if task_type == "execute_research_plan":
            return await self._execute_research_plan(task["plan"])
        elif task_type == "internal_search":
            return await self._execute_internal_search(task)
        elif task_type == "external_search":
            return await self._execute_external_search(task)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def _execute_research_plan(self, plan) -> Dict[str, Any]:
        """Execute all research tasks in a plan."""
        await self.update_status("executing_plan", "Executing research plan")
        
        # Prepare all tasks
        internal_tasks = [
            self._execute_internal_search_task(task) 
            for task in plan.internal_search_tasks
        ]
        external_tasks = [
            self._execute_external_search_task(task) 
            for task in plan.external_search_tasks
        ]
        
        # Execute tasks in parallel
        all_tasks = internal_tasks + external_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Separate successful results from exceptions
        successful_results = []
        failed_tasks = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                task_info = plan.internal_search_tasks[i] if i < len(internal_tasks) else plan.external_search_tasks[i - len(internal_tasks)]
                failed_tasks.append({
                    "task": task_info,
                    "error": str(result)
                })
                logger.error(f"Task failed: {result}")
            else:
                successful_results.append(result)
        
        logger.info(f"Research plan execution completed: {len(successful_results)} successful, {len(failed_tasks)} failed")
        
        return {
            "success": True,
            "results": successful_results,
            "failed_tasks": failed_tasks,
            "total_tasks": len(all_tasks)
        }
    
    async def _execute_internal_search_task(self, task: Dict[str, Any]) -> ResearchResult:
        """Execute an internal document search task."""
        import time
        start_time = time.time()
        
        task_id = task.get("id", "unknown")
        query = task.get("query", "")
        file_types = task.get("file_types", [])
        
        logger.info(f"Executing internal search task {task_id}: {query}")
        
        # Search internal documents
        search_results = self.document_store.search_documents(
            query=query,
            n_results=20,
            file_types=file_types if file_types else None
        )
        
        # Generate summary of findings
        summary = await self._summarize_internal_results(query, search_results)
        
        # Calculate confidence based on result quality
        confidence = self._calculate_confidence(search_results)
        
        execution_time = time.time() - start_time
        
        return ResearchResult(
            task_id=task_id,
            query=query,
            source_type="internal",
            results=search_results,
            summary=summary,
            confidence=confidence,
            execution_time=execution_time
        )
    
    async def _execute_external_search_task(self, task: Dict[str, Any]) -> ResearchResult:
        """Execute an external web search task."""
        import time
        start_time = time.time()
        
        task_id = task.get("id", "unknown")
        query = task.get("query", "")
        sources = task.get("sources", ["general"])
        
        logger.info(f"Executing external search task {task_id}: {query}")
        
        # Search external sources
        search_results = await self.web_search.search(query, max_results=10)
        
        # Generate summary of findings
        summary = await self._summarize_external_results(query, search_results)
        
        # Calculate confidence
        confidence = self._calculate_confidence(search_results)
        
        execution_time = time.time() - start_time
        
        return ResearchResult(
            task_id=task_id,
            query=query,
            source_type="external",
            results=search_results,
            summary=summary,
            confidence=confidence,
            execution_time=execution_time
        )
    
    async def _execute_internal_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single internal search task."""
        try:
            result = await self._execute_internal_search_task(task)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Internal search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_external_search(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single external search task."""
        try:
            result = await self._execute_external_search_task(task)
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"External search failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _summarize_internal_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a summary of internal search results."""
        if not results:
            return "No relevant internal documents found."
        
        # Prepare content for summarization
        content_snippets = []
        for result in results[:10]:  # Limit to top 10 results
            content = result.get('content', '')[:500]  # Limit content length
            file_path = result.get('metadata', {}).get('file_path', 'Unknown')
            content_snippets.append(f"From {file_path}: {content}")
        
        combined_content = "\n\n".join(content_snippets)
        
        prompt = f"""
        Summarize the following internal document search results for the query: "{query}"
        
        Search Results:
        {combined_content}
        
        Please provide a concise summary highlighting the key findings and their relevance to the query.
        Include specific document references where appropriate.
        """
        
        try:
            summary = await self.call_llm([
                {"role": "system", "content": "You are a research assistant summarizing document search results."},
                {"role": "user", "content": prompt}
            ], max_tokens=1000)
            return summary
        except Exception as e:
            logger.error(f"Error generating internal summary: {e}")
            return f"Found {len(results)} relevant documents but failed to generate summary."
    
    async def _summarize_external_results(self, query: str, results: List[Dict[str, Any]]) -> str:
        """Generate a summary of external search results."""
        if not results:
            return "No relevant external sources found."
        
        # Prepare content for summarization
        content_snippets = []
        for result in results[:10]:  # Limit to top 10 results
            title = result.get('title', 'No title')
            snippet = result.get('snippet', '')[:300]  # Limit snippet length
            url = result.get('url', 'No URL')
            content_snippets.append(f"Title: {title}\nURL: {url}\nContent: {snippet}")
        
        combined_content = "\n\n".join(content_snippets)
        
        prompt = f"""
        Summarize the following web search results for the query: "{query}"
        
        Search Results:
        {combined_content}
        
        Please provide a concise summary highlighting the key findings and their relevance to the query.
        Include source URLs where appropriate.
        """
        
        try:
            summary = await self.call_llm([
                {"role": "system", "content": "You are a research assistant summarizing web search results."},
                {"role": "user", "content": prompt}
            ], max_tokens=1000)
            return summary
        except Exception as e:
            logger.error(f"Error generating external summary: {e}")
            return f"Found {len(results)} relevant web sources but failed to generate summary."
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on search results quality."""
        if not results:
            return 0.0
        
        # Simple confidence calculation based on:
        # - Number of results
        # - Average relevance score (if available)
        # - Content quality indicators
        
        base_confidence = min(len(results) / 10.0, 1.0)  # More results = higher confidence, capped at 1.0
        
        # Adjust based on relevance scores if available
        if results and 'distance' in results[0]:
            avg_distance = sum(r.get('distance', 1.0) for r in results) / len(results)
            relevance_factor = max(0.0, 1.0 - avg_distance)  # Lower distance = higher relevance
            base_confidence *= relevance_factor
        
        return round(base_confidence, 2)