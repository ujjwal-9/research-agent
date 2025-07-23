"""LLM-based entity extraction and relationship identification using OpenAI API."""

import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import re
from loguru import logger
from openai import AsyncOpenAI

from src.config import settings
from src.knowledge_graph.entity_extractor import Entity, Relationship


@dataclass
class LLMEntity:
    """Enhanced entity with LLM-extracted metadata."""
    text: str
    label: str
    confidence: float
    description: str = ""
    aliases: List[str] = None
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []
        if self.properties is None:
            self.properties = {}
    
    def to_entity(self) -> Entity:
        """Convert to base Entity class."""
        return Entity(
            text=self.text,
            label=self.label,
            start=0,  # LLM extraction doesn't provide position
            end=len(self.text),
            confidence=self.confidence,
            normalized_text=self.text.lower().strip()
        )


@dataclass
class LLMRelationship:
    """Enhanced relationship with LLM-extracted metadata."""
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    description: str = ""
    properties: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class LLMEntityExtractor:
    """Extracts entities and relationships using OpenAI's LLM."""
    
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.llm_extraction_model
        self.max_text_length = settings.llm_max_text_length
        self.temperature = settings.llm_extraction_temperature
    
    async def extract_entities_and_relationships(
        self, 
        text: str, 
        chunk_id: str = None,
        domain_context: str = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text using LLM."""
        try:
            # Extract entities
            llm_entities = await self._extract_entities_llm(text, domain_context)
            
            # Extract relationships
            llm_relationships = await self._extract_relationships_llm(text, llm_entities, domain_context)
            
            # Convert to base classes
            entities = [ent.to_entity() for ent in llm_entities]
            relationships = self._convert_relationships(llm_relationships, entities)
            
            logger.info(f"LLM extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk_id}")
            
            return entities, relationships
            
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
            return [], []
    
    async def _extract_entities_llm(
        self, 
        text: str, 
        domain_context: str = None
    ) -> List[LLMEntity]:
        """Extract entities using LLM."""
        
        system_prompt = self._get_entity_extraction_prompt(domain_context)
        
        user_prompt = f"""
        Extract entities from the following text. Focus on:
        - People (PERSON)
        - Organizations (ORG) 
        - Locations (GPE, LOC)
        - Products/Services (PRODUCT)
        - Events (EVENT)
        - Concepts/Topics (CONCEPT)
        - Technologies (TECH)
        - Dates/Times (DATE)
        - Money/Financial (MONEY)
        
        Text to analyze:
        {text[:self.max_text_length]}  # Limit text length for API
        
        Return a JSON array of entities with this structure:
        {{
            "text": "entity name",
            "label": "entity type",
            "confidence": 0.95,
            "description": "brief description",
            "aliases": ["alternative names"],
            "properties": {{"key": "value"}}
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=2000
            )
            
            content = response.choices[0].message.content
            entities_data = self._parse_json_response(content)
            
            entities = []
            for entity_dict in entities_data:
                try:
                    entity = LLMEntity(**entity_dict)
                    entities.append(entity)
                except Exception as e:
                    logger.warning(f"Failed to parse entity: {entity_dict}, error: {e}")
            
            return entities
            
        except Exception as e:
            logger.error(f"Error in LLM entity extraction: {e}")
            return []
    
    async def _extract_relationships_llm(
        self, 
        text: str, 
        entities: List[LLMEntity],
        domain_context: str = None
    ) -> List[LLMRelationship]:
        """Extract relationships using LLM."""
        
        if len(entities) < 2:
            return []
        
        entity_list = "\n".join([f"- {ent.text} ({ent.label})" for ent in entities])
        
        system_prompt = self._get_relationship_extraction_prompt(domain_context)
        
        user_prompt = f"""
        Given the following text and extracted entities, identify relationships between entities.
        
        Entities:
        {entity_list}
        
        Text:
        {text[:self.max_text_length - 1000]}  # Limit text length, reserve space for entities list
        
        Common relationship types:
        - WORKS_FOR, EMPLOYED_BY
        - OWNS, FOUNDED, CREATED
        - LOCATED_IN, BASED_IN
        - PART_OF, SUBSIDIARY_OF
        - PARTNERED_WITH, COLLABORATED_WITH
        - ACQUIRED, MERGED_WITH
        - RELATED_TO, ASSOCIATED_WITH
        - LEADS, MANAGES
        - COMPETES_WITH
        - SUPPLIES, PROVIDES
        
        Return a JSON array of relationships:
        {{
            "source_entity": "entity name",
            "target_entity": "entity name", 
            "relation_type": "relationship type",
            "confidence": 0.85,
            "description": "brief description",
            "properties": {{"key": "value"}}
        }}
        """
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=1500
            )
            
            content = response.choices[0].message.content
            relationships_data = self._parse_json_response(content)
            
            relationships = []
            for rel_dict in relationships_data:
                try:
                    relationship = LLMRelationship(**rel_dict)
                    relationships.append(relationship)
                except Exception as e:
                    logger.warning(f"Failed to parse relationship: {rel_dict}, error: {e}")
            
            return relationships
            
        except Exception as e:
            logger.error(f"Error in LLM relationship extraction: {e}")
            return []
    
    def _get_entity_extraction_prompt(self, domain_context: str = None) -> str:
        """Get system prompt for entity extraction."""
        base_prompt = """
        You are an expert entity extraction system. Your task is to identify and extract meaningful entities from text.
        
        Guidelines:
        1. Extract only significant, meaningful entities
        2. Avoid extracting common words, pronouns, or generic terms
        3. Provide accurate entity types from the specified categories
        4. Include confidence scores based on clarity and importance
        5. Add brief descriptions to provide context
        6. Include alternative names/aliases when relevant
        7. Extract properties that add meaningful information
        
        Focus on entities that would be valuable in a knowledge graph for research and analysis.
        """
        
        if domain_context:
            base_prompt += f"\n\nDomain context: {domain_context}"
        
        return base_prompt
    
    def _get_relationship_extraction_prompt(self, domain_context: str = None) -> str:
        """Get system prompt for relationship extraction."""
        base_prompt = """
        You are an expert relationship extraction system. Your task is to identify meaningful relationships between entities in text.
        
        Guidelines:
        1. Only extract relationships that are explicitly stated or strongly implied
        2. Use appropriate relationship types from the provided list
        3. Ensure both entities exist in the provided entity list
        4. Provide confidence scores based on clarity of the relationship
        5. Add descriptions to explain the relationship context
        6. Include relevant properties (dates, amounts, etc.)
        
        Focus on relationships that add value to a knowledge graph for research and analysis.
        """
        
        if domain_context:
            base_prompt += f"\n\nDomain context: {domain_context}"
        
        return base_prompt
    
    def _parse_json_response(self, content: str) -> List[Dict]:
        """Parse JSON response from LLM, handling potential formatting issues."""
        try:
            # Try direct JSON parsing
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find JSON array in the text
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
            
            logger.warning(f"Failed to parse JSON response: {content[:200]}...")
            return []
    
    def _convert_relationships(
        self, 
        llm_relationships: List[LLMRelationship], 
        entities: List[Entity]
    ) -> List[Relationship]:
        """Convert LLM relationships to base Relationship objects."""
        relationships = []
        
        # Create entity lookup
        entity_lookup = {ent.text.lower(): ent for ent in entities}
        
        for llm_rel in llm_relationships:
            source_ent = entity_lookup.get(llm_rel.source_entity.lower())
            target_ent = entity_lookup.get(llm_rel.target_entity.lower())
            
            if source_ent and target_ent:
                relationship = Relationship(
                    source_entity=source_ent,
                    target_entity=target_ent,
                    relation_type=llm_rel.relation_type,
                    confidence=llm_rel.confidence,
                    context=llm_rel.description,
                    sentence=llm_rel.description
                )
                relationships.append(relationship)
        
        return relationships


class HybridEntityExtractor:
    """Combines spaCy and LLM-based entity extraction."""
    
    def __init__(self):
        from src.knowledge_graph.entity_extractor import EntityExtractor
        self.spacy_extractor = EntityExtractor()
        self.llm_extractor = LLMEntityExtractor()
    
    async def extract_entities_and_relationships(
        self, 
        text: str, 
        chunk_id: str = None,
        use_llm: bool = True,
        domain_context: str = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities using both spaCy and LLM, then merge results."""
        
        # Extract with spaCy
        spacy_entities, spacy_relationships = await self.spacy_extractor.extract_entities_and_relationships(
            text, chunk_id
        )
        
        if not use_llm:
            return spacy_entities, spacy_relationships
        
        # Extract with LLM
        llm_entities, llm_relationships = await self.llm_extractor.extract_entities_and_relationships(
            text, chunk_id, domain_context
        )
        
        # Merge results
        merged_entities = self._merge_entities(spacy_entities, llm_entities)
        merged_relationships = self._merge_relationships(spacy_relationships, llm_relationships)
        
        logger.info(f"Hybrid extraction: {len(merged_entities)} entities, {len(merged_relationships)} relationships")
        
        return merged_entities, merged_relationships
    
    def _merge_entities(self, spacy_entities: List[Entity], llm_entities: List[Entity]) -> List[Entity]:
        """Merge entities from both extractors, removing duplicates."""
        merged = []
        seen_normalized = set()
        
        # Add spaCy entities first (they have position information)
        for entity in spacy_entities:
            key = (entity.normalized_text, entity.label)
            if key not in seen_normalized:
                seen_normalized.add(key)
                merged.append(entity)
        
        # Add LLM entities that aren't duplicates
        for entity in llm_entities:
            key = (entity.normalized_text, entity.label)
            if key not in seen_normalized:
                seen_normalized.add(key)
                merged.append(entity)
        
        return merged
    
    def _merge_relationships(
        self, 
        spacy_relationships: List[Relationship], 
        llm_relationships: List[Relationship]
    ) -> List[Relationship]:
        """Merge relationships from both extractors."""
        merged = []
        seen_relationships = set()
        
        # Add all relationships, avoiding exact duplicates
        for rel_list in [spacy_relationships, llm_relationships]:
            for relationship in rel_list:
                key = (
                    relationship.source_entity.normalized_text,
                    relationship.target_entity.normalized_text,
                    relationship.relation_type
                )
                if key not in seen_relationships:
                    seen_relationships.add(key)
                    merged.append(relationship)
        
        return merged


# Convenience function for easy usage
async def extract_with_llm(
    text: str, 
    chunk_id: str = None,
    domain_context: str = None,
    use_hybrid: bool = True
) -> Tuple[List[Entity], List[Relationship]]:
    """Extract entities and relationships using LLM or hybrid approach."""
    
    if use_hybrid:
        extractor = HybridEntityExtractor()
        return await extractor.extract_entities_and_relationships(
            text, chunk_id, use_llm=True, domain_context=domain_context
        )
    else:
        extractor = LLMEntityExtractor()
        return await extractor.extract_entities_and_relationships(
            text, chunk_id, domain_context
        )