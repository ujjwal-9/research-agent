"""Knowledge Graph Extractor for entities and relationships."""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from dataclasses import dataclass

from .config import KnowledgeGraphConfig


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""

    id: str
    name: str
    type: str
    description: str
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class Relationship:
    """Represents a relationship between entities."""

    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    description: str
    properties: Dict[str, Any] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


@dataclass
class KnowledgeGraphData:
    """Container for extracted knowledge graph data."""

    entities: List[Entity]
    relationships: List[Relationship]
    source_document: str
    chunk_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "type": e.type,
                    "description": e.description,
                    "properties": e.properties,
                }
                for e in self.entities
            ],
            "relationships": [
                {
                    "source_entity_id": r.source_entity_id,
                    "target_entity_id": r.target_entity_id,
                    "relationship_type": r.relationship_type,
                    "description": r.description,
                    "properties": r.properties,
                    "confidence": r.confidence,
                }
                for r in self.relationships
            ],
            "source_document": self.source_document,
            "chunk_id": self.chunk_id,
        }


class KnowledgeGraphExtractor:
    """Extracts entities and relationships from text using LLM."""

    def __init__(self, config: KnowledgeGraphConfig):
        """Initialize the knowledge graph extractor."""
        self.config = config
        self.client = OpenAI(api_key=config.openai_api_key)
        self.logger = logging.getLogger(__name__)

    def extract_knowledge_graph(
        self,
        text: str,
        source_document: str,
        chunk_id: str,
        context: Optional[str] = None,
    ) -> KnowledgeGraphData:
        """
        Extract entities and relationships from text.

        Args:
            text: The text to process
            source_document: Source document path/name
            chunk_id: Unique identifier for this text chunk
            context: Optional context about the document

        Returns:
            KnowledgeGraphData object with extracted entities and relationships
        """
        self.logger.info(f"🧠 Extracting knowledge graph from chunk: {chunk_id}")

        try:
            # Create the extraction prompt
            prompt = self._create_extraction_prompt(text, context)

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.config.knowledge_graph_model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )

            # Parse the response
            result = json.loads(response.choices[0].message.content)

            # Convert to structured objects
            entities = self._parse_entities(result.get("entities", []))
            relationships = self._parse_relationships(result.get("relationships", []))

            # Validate and clean the data
            entities, relationships = self._validate_and_clean(entities, relationships)

            self.logger.info(
                f"✅ Extracted {len(entities)} entities and {len(relationships)} relationships"
            )

            return KnowledgeGraphData(
                entities=entities,
                relationships=relationships,
                source_document=source_document,
                chunk_id=chunk_id,
            )

        except Exception as e:
            self.logger.error(f"❌ Failed to extract knowledge graph: {e}")
            return KnowledgeGraphData(
                entities=[],
                relationships=[],
                source_document=source_document,
                chunk_id=chunk_id,
            )

    def _get_system_prompt(self) -> str:
        """Get the system prompt for entity/relationship extraction."""
        entity_types_str = ", ".join(self.config.entity_types)

        return f"""You are a knowledge graph extraction expert. Your task is to extract entities and relationships from text to build a comprehensive knowledge graph.

ENTITY TYPES TO EXTRACT: {entity_types_str}

EXTRACTION GUIDELINES:
1. Extract only the most important and relevant entities (max {self.config.max_entities_per_chunk} per chunk)
2. Focus on entities that have clear relationships with other entities  
3. Create meaningful relationships that capture the semantic connections (max {self.config.max_relationships_per_chunk} per chunk)
4. Use clear, descriptive names for entities and relationships
5. Include confidence scores for relationships (0.0 to 1.0)
6. Normalize entity names (e.g., "John Smith" not "john smith" or "JOHN SMITH")

ENTITY ID GENERATION:
- Use descriptive, unique IDs that combine type and name
- Format: TYPE_NORMALIZED_NAME (e.g., "PERSON_john_smith", "ORG_microsoft_corp")
- Replace spaces with underscores, convert to lowercase
- Remove special characters except underscores

RELATIONSHIP TYPES:
- WORKS_FOR, EMPLOYED_BY, FOUNDED, LEADS, MANAGES
- LOCATED_IN, BASED_IN, OPERATES_IN  
- OWNS, ACQUIRED_BY, INVESTED_IN, PARTNERS_WITH
- CREATED, DEVELOPED, LAUNCHED, PRODUCES
- MENTIONS, DISCUSSES, REFERENCES, RELATED_TO
- BEFORE, AFTER, DURING (temporal)
- PART_OF, CONTAINS, INCLUDES

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{{
    "entities": [
        {{
            "id": "entity_unique_id",
            "name": "Entity Name",
            "type": "ENTITY_TYPE", 
            "description": "Brief description of the entity",
            "properties": {{"key": "value"}}
        }}
    ],
    "relationships": [
        {{
            "source_entity_id": "source_id",
            "target_entity_id": "target_id", 
            "relationship_type": "RELATIONSHIP_TYPE",
            "description": "Description of the relationship",
            "confidence": 0.95,
            "properties": {{"key": "value"}}
        }}
    ]
}}"""

    def _create_extraction_prompt(
        self, text: str, context: Optional[str] = None
    ) -> str:
        """Create the extraction prompt for the given text."""
        prompt = f"""Extract entities and relationships from the following text:

TEXT TO ANALYZE:
{text}
"""

        if context:
            prompt = f"""DOCUMENT CONTEXT: {context}

{prompt}"""

        prompt += """

INSTRUCTIONS:
1. Identify key entities from the specified types
2. Extract meaningful relationships between entities
3. Focus on factual information, avoid speculation
4. Return only the JSON response, no additional text"""

        return prompt

    def _parse_entities(self, entities_data: List[Dict]) -> List[Entity]:
        """Parse entity data from API response."""
        entities = []

        for entity_dict in entities_data:
            try:
                entity = Entity(
                    id=entity_dict.get("id", ""),
                    name=entity_dict.get("name", ""),
                    type=entity_dict.get("type", "UNKNOWN"),
                    description=entity_dict.get("description", ""),
                    properties=entity_dict.get("properties", {}),
                )

                # Validate required fields
                if entity.id and entity.name and entity.type:
                    entities.append(entity)
                else:
                    self.logger.warning(f"Skipping invalid entity: {entity_dict}")

            except Exception as e:
                self.logger.warning(f"Failed to parse entity: {entity_dict} - {e}")

        return entities

    def _parse_relationships(
        self, relationships_data: List[Dict]
    ) -> List[Relationship]:
        """Parse relationship data from API response."""
        relationships = []

        for rel_dict in relationships_data:
            try:
                relationship = Relationship(
                    source_entity_id=rel_dict.get("source_entity_id", ""),
                    target_entity_id=rel_dict.get("target_entity_id", ""),
                    relationship_type=rel_dict.get("relationship_type", "RELATED_TO"),
                    description=rel_dict.get("description", ""),
                    confidence=float(rel_dict.get("confidence", 1.0)),
                    properties=rel_dict.get("properties", {}),
                )

                # Validate required fields
                if (
                    relationship.source_entity_id
                    and relationship.target_entity_id
                    and relationship.relationship_type
                ):
                    relationships.append(relationship)
                else:
                    self.logger.warning(f"Skipping invalid relationship: {rel_dict}")

            except Exception as e:
                self.logger.warning(f"Failed to parse relationship: {rel_dict} - {e}")

        return relationships

    def _validate_and_clean(
        self, entities: List[Entity], relationships: List[Relationship]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Validate and clean extracted entities and relationships."""

        # Create set of valid entity IDs
        entity_ids = {entity.id for entity in entities}

        # Filter relationships to only include those with valid entity references
        valid_relationships = []
        for rel in relationships:
            if (
                rel.source_entity_id in entity_ids
                and rel.target_entity_id in entity_ids
                and rel.source_entity_id != rel.target_entity_id
            ):  # Avoid self-references
                valid_relationships.append(rel)
            else:
                self.logger.debug(f"Filtered invalid relationship: {rel}")

        # Remove duplicate entities (by ID)
        unique_entities = {}
        for entity in entities:
            if entity.id not in unique_entities:
                unique_entities[entity.id] = entity
            else:
                self.logger.debug(f"Filtered duplicate entity: {entity.id}")

        return list(unique_entities.values()), valid_relationships
