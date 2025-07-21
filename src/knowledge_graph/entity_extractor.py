"""Entity extraction and relationship identification."""

import asyncio
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc, Span
import re
from loguru import logger

from src.config import settings


@dataclass
class Entity:
    """Represents an extracted entity."""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    normalized_text: str = None
    
    def __post_init__(self):
        if self.normalized_text is None:
            self.normalized_text = self._normalize_text(self.text)
    
    def _normalize_text(self, text: str) -> str:
        """Normalize entity text for consistent matching."""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        # Remove common prefixes/suffixes that don't add meaning
        normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
        normalized = re.sub(r'\s+(inc|corp|ltd|llc)\.?$', '', normalized)
        return normalized


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_entity: Entity
    target_entity: Entity
    relation_type: str
    confidence: float
    context: str
    sentence: str


class EntityExtractor:
    """Extracts entities and relationships from text using spaCy."""
    
    def __init__(self):
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load spaCy model."""
        try:
            self.nlp = spacy.load(settings.spacy_model)
            logger.info(f"Loaded spaCy model: {settings.spacy_model}")
        except OSError:
            logger.error(f"spaCy model {settings.spacy_model} not found. Please install it with: python -m spacy download {settings.spacy_model}")
            raise
    
    async def extract_entities_and_relationships(
        self, 
        text: str, 
        chunk_id: str = None
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from text."""
        if not self.nlp:
            return [], []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract entities
        entities = self._extract_entities(doc, chunk_id)
        
        # Extract relationships
        relationships = self._extract_relationships(doc, entities)
        
        logger.debug(f"Extracted {len(entities)} entities and {len(relationships)} relationships from chunk {chunk_id}")
        
        return entities, relationships
    
    def _extract_entities(self, doc: Doc, chunk_id: str = None) -> List[Entity]:
        """Extract named entities from spaCy doc."""
        entities = []
        
        for ent in doc.ents:
            # Filter by confidence and relevance
            if self._is_relevant_entity(ent):
                entity = Entity(
                    text=ent.text,
                    label=ent.label_,
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=self._calculate_entity_confidence(ent)
                )
                entities.append(entity)
        
        # Deduplicate entities with similar normalized text
        entities = self._deduplicate_entities(entities)
        
        # Limit number of entities per chunk
        if len(entities) > settings.max_entities_per_chunk:
            entities = sorted(entities, key=lambda x: x.confidence, reverse=True)[:settings.max_entities_per_chunk]
        
        return entities
    
    def _is_relevant_entity(self, ent: Span) -> bool:
        """Check if entity is relevant for knowledge graph."""
        # Filter out very short entities
        if len(ent.text.strip()) < 2:
            return False
        
        # Filter out entities that are mostly punctuation or numbers
        if re.match(r'^[^\w\s]*$', ent.text) or re.match(r'^\d+$', ent.text):
            return False
        
        # Include relevant entity types
        relevant_types = {
            'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART',
            'LAW', 'LANGUAGE', 'NORP', 'FAC', 'LOC', 'MONEY', 'PERCENT'
        }
        
        return ent.label_ in relevant_types
    
    def _calculate_entity_confidence(self, ent: Span) -> float:
        """Calculate confidence score for entity."""
        # Base confidence from spaCy (if available)
        base_confidence = getattr(ent, 'confidence', 0.8)
        
        # Adjust based on entity characteristics
        length_factor = min(len(ent.text) / 20.0, 1.0)  # Longer entities tend to be more reliable
        
        # Adjust based on entity type reliability
        type_confidence = {
            'PERSON': 0.9, 'ORG': 0.85, 'GPE': 0.9, 'PRODUCT': 0.7,
            'EVENT': 0.75, 'WORK_OF_ART': 0.8, 'LAW': 0.85, 'LANGUAGE': 0.9,
            'NORP': 0.8, 'FAC': 0.75, 'LOC': 0.85, 'MONEY': 0.95, 'PERCENT': 0.95
        }.get(ent.label_, 0.7)
        
        final_confidence = base_confidence * length_factor * type_confidence
        return round(min(final_confidence, 1.0), 2)
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities based on normalized text."""
        seen = set()
        unique_entities = []
        
        for entity in entities:
            key = (entity.normalized_text, entity.label)
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _extract_relationships(self, doc: Doc, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        
        # Create entity lookup by position
        entity_spans = {(ent.start, ent.end): ent for ent in entities}
        
        # Process each sentence
        for sent in doc.sents:
            sent_entities = [ent for ent in entities if sent.start_char <= ent.start < sent.end_char]
            
            if len(sent_entities) >= 2:
                # Extract relationships within sentence
                sent_relationships = self._extract_sentence_relationships(sent, sent_entities)
                relationships.extend(sent_relationships)
        
        return relationships
    
    def _extract_sentence_relationships(self, sent, entities: List[Entity]) -> List[Relationship]:
        """Extract relationships within a sentence."""
        relationships = []
        
        # Simple pattern-based relationship extraction
        sent_text = sent.text.lower()
        
        # Define relationship patterns
        patterns = [
            (r'(.+?)\s+(?:is|was|are|were)\s+(?:a|an|the)?\s*(.+)', 'IS_A'),
            (r'(.+?)\s+(?:works for|employed by|part of)\s+(.+)', 'WORKS_FOR'),
            (r'(.+?)\s+(?:owns|founded|created|developed)\s+(.+)', 'OWNS'),
            (r'(.+?)\s+(?:located in|based in|from)\s+(.+)', 'LOCATED_IN'),
            (r'(.+?)\s+(?:related to|connected to|associated with)\s+(.+)', 'RELATED_TO'),
            (r'(.+?)\s+(?:acquired|bought|purchased)\s+(.+)', 'ACQUIRED'),
            (r'(.+?)\s+(?:partnered with|collaborated with)\s+(.+)', 'PARTNERED_WITH'),
        ]
        
        # Try to match entities with patterns
        for i, entity1 in enumerate(entities):
            for j, entity2 in enumerate(entities[i+1:], i+1):
                # Check if entities appear in a relationship pattern
                for pattern, relation_type in patterns:
                    if self._entities_match_pattern(sent_text, entity1, entity2, pattern):
                        relationship = Relationship(
                            source_entity=entity1,
                            target_entity=entity2,
                            relation_type=relation_type,
                            confidence=0.7,  # Base confidence for pattern matching
                            context=sent.text,
                            sentence=sent.text
                        )
                        relationships.append(relationship)
                        break
        
        return relationships
    
    def _entities_match_pattern(self, sent_text: str, entity1: Entity, entity2: Entity, pattern: str) -> bool:
        """Check if two entities match a relationship pattern in the sentence."""
        # Simple implementation - can be enhanced with more sophisticated NLP
        ent1_text = entity1.normalized_text
        ent2_text = entity2.normalized_text
        
        # Check if both entities appear in sentence and pattern might match
        if ent1_text in sent_text and ent2_text in sent_text:
            # Simple distance check - entities should be reasonably close
            ent1_pos = sent_text.find(ent1_text)
            ent2_pos = sent_text.find(ent2_text)
            
            if abs(ent1_pos - ent2_pos) < 100:  # Within 100 characters
                return True
        
        return False


class EntityLinker:
    """Links entities across documents and resolves duplicates."""
    
    def __init__(self):
        self.entity_cache = {}  # Cache for entity resolution
    
    async def link_entities(self, entities: List[Entity]) -> List[Entity]:
        """Link and deduplicate entities across documents."""
        linked_entities = []
        
        for entity in entities:
            # Try to find existing entity with same normalized text
            canonical_entity = self._find_canonical_entity(entity)
            
            if canonical_entity:
                # Update confidence if this instance has higher confidence
                if entity.confidence > canonical_entity.confidence:
                    canonical_entity.confidence = entity.confidence
                linked_entities.append(canonical_entity)
            else:
                # New entity
                self.entity_cache[entity.normalized_text] = entity
                linked_entities.append(entity)
        
        return linked_entities
    
    def _find_canonical_entity(self, entity: Entity) -> Entity:
        """Find canonical version of entity."""
        # Exact match
        if entity.normalized_text in self.entity_cache:
            return self.entity_cache[entity.normalized_text]
        
        # Fuzzy matching for similar entities
        for cached_text, cached_entity in self.entity_cache.items():
            if self._entities_similar(entity.normalized_text, cached_text):
                return cached_entity
        
        return None
    
    def _entities_similar(self, text1: str, text2: str, threshold: float = 0.8) -> bool:
        """Check if two entity texts are similar enough to be the same entity."""
        # Simple similarity check - can be enhanced with more sophisticated matching
        if text1 == text2:
            return True
        
        # Check if one is substring of another
        if text1 in text2 or text2 in text1:
            return True
        
        # Jaccard similarity for word-based comparison
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        similarity = intersection / union
        return similarity >= threshold