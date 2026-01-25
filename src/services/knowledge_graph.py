"""
Knowledge Graph Service for Vira AI

Implements dynamic Subject-Predicate-Object (SPO) triple storage
for entity relationships using async SQLite.

Features:
- Store entity relationships as triples
- Query by subject, predicate, or object
- Traverse relationships (multi-hop queries)
- Confidence-weighted edges
- Temporal decay for relationships
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict

from src.database import DBConnection

logger = logging.getLogger(__name__)


@dataclass
class Triple:
    """Represents a Subject-Predicate-Object relationship."""
    id: Optional[int]
    user_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_memory_id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    metadata: Optional[Dict] = None


@dataclass
class Entity:
    """Represents an entity node in the graph."""
    name: str
    entity_type: str  # person, place, thing, concept, event
    aliases: List[str]
    properties: Dict[str, any]
    first_seen: datetime
    mention_count: int = 1


class KnowledgeGraph:
    """
    Async knowledge graph for storing and querying entity relationships.
    Uses SQLite with SPO triple storage pattern.
    """
    
    # Decay parameters
    DECAY_HALF_LIFE_DAYS = 30
    MIN_CONFIDENCE_THRESHOLD = 0.1
    
    # Predicate categories for semantic grouping
    PREDICATE_CATEGORIES = {
        'identity': ['is', 'is_a', 'type_of', 'called', 'named'],
        'possession': ['has', 'owns', 'possesses', 'contains'],
        'relation': ['knows', 'related_to', 'friend_of', 'sibling_of', 'parent_of', 'child_of'],
        'location': ['lives_in', 'works_at', 'located_in', 'from'],
        'preference': ['likes', 'dislikes', 'prefers', 'loves', 'hates'],
        'temporal': ['born_on', 'started', 'ended', 'scheduled_for'],
        'action': ['does', 'works_as', 'studies', 'plays']
    }
    
    def __init__(self, db: DBConnection):
        self.db = db
        self._lock = asyncio.Lock()
        self._entity_cache: Dict[str, Entity] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = 0
    
    async def initialize(self):
        """Create knowledge graph tables if they don't exist."""
        async with self._lock:
            # Triples table - core SPO storage
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS kg_triples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    predicate TEXT NOT NULL,
                    object TEXT NOT NULL,
                    confidence REAL DEFAULT 0.8,
                    source_memory_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT,
                    UNIQUE(user_id, subject, predicate, object)
                )
            """, ())
            
            # Entities table - entity metadata
            await self.db.execute("""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    entity_type TEXT DEFAULT 'unknown',
                    aliases TEXT,
                    properties TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    mention_count INTEGER DEFAULT 1,
                    UNIQUE(user_id, name)
                )
            """, ())
            
            # Create indexes for efficient queries
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_triples_user_subject ON kg_triples(user_id, subject)",
                ()
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_triples_user_object ON kg_triples(user_id, object)",
                ()
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_triples_predicate ON kg_triples(predicate)",
                ()
            )
            await self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_entities_user ON kg_entities(user_id, name)",
                ()
            )
            
            logger.info("[KNOWLEDGE-GRAPH] Tables initialized")
    
    async def add_triple(
        self,
        user_id: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source_memory_id: str = None,
        metadata: Dict = None
    ) -> Optional[int]:
        """
        Add or update a triple in the knowledge graph.
        Returns the triple ID.
        """
        subject = self._normalize_entity(subject)
        predicate = self._normalize_predicate(predicate)
        obj = self._normalize_entity(obj)
        
        if not all([subject, predicate, obj]):
            logger.debug("[KG] Invalid triple - missing components")
            return None
        
        async with self._lock:
            try:
                # Check for existing triple
                existing = await self.db.fetchone("""
                    SELECT id, confidence, access_count FROM kg_triples
                    WHERE user_id=? AND subject=? AND predicate=? AND object=?
                """, (user_id, subject, predicate, obj))
                
                if existing:
                    # Update existing triple with higher confidence
                    new_confidence = min(1.0, (existing[1] + confidence) / 2 + 0.05)
                    await self.db.execute("""
                        UPDATE kg_triples 
                        SET confidence=?, last_accessed=?, access_count=access_count+1,
                            metadata=COALESCE(?, metadata)
                        WHERE id=?
                    """, (new_confidence, datetime.now(), json.dumps(metadata) if metadata else None, existing[0]))
                    
                    logger.debug(f"[KG] Updated triple: {subject} -{predicate}-> {obj}")
                    return existing[0]
                
                # Insert new triple
                await self.db.execute("""
                    INSERT INTO kg_triples 
                    (user_id, subject, predicate, object, confidence, source_memory_id, 
                     created_at, last_accessed, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, subject, predicate, obj, confidence, source_memory_id,
                    datetime.now(), datetime.now(), json.dumps(metadata) if metadata else None
                ))
                
                # Get the inserted ID
                row = await self.db.fetchone("SELECT last_insert_rowid()", ())
                triple_id = row[0] if row else None
                
                # Ensure entities exist
                await self._ensure_entity(user_id, subject)
                await self._ensure_entity(user_id, obj)
                
                logger.info(f"[KG] Added triple: {subject} -{predicate}-> {obj} (conf: {confidence})")
                return triple_id
                
            except Exception as e:
                logger.error(f"[KG] Failed to add triple: {e}")
                return None
    
    async def add_triples_batch(
        self,
        user_id: str,
        triples: List[Tuple[str, str, str, float]]
    ) -> int:
        """
        Add multiple triples at once.
        Each tuple: (subject, predicate, object, confidence)
        Returns count of successfully added triples.
        """
        added = 0
        for subject, predicate, obj, confidence in triples:
            result = await self.add_triple(user_id, subject, predicate, obj, confidence)
            if result:
                added += 1
        return added
    
    async def query_by_subject(
        self,
        user_id: str,
        subject: str,
        predicate: str = None,
        min_confidence: float = 0.3
    ) -> List[Triple]:
        """Get all triples where the given entity is the subject."""
        subject = self._normalize_entity(subject)
        
        query = """
            SELECT id, user_id, subject, predicate, object, confidence,
                   source_memory_id, created_at, last_accessed, access_count, metadata
            FROM kg_triples
            WHERE user_id=? AND subject=? AND confidence >= ?
        """
        params = [user_id, subject, min_confidence]
        
        if predicate:
            query += " AND predicate=?"
            params.append(self._normalize_predicate(predicate))
        
        query += " ORDER BY confidence DESC, last_accessed DESC"
        
        rows = await self.db.fetchall(query, params)
        triples = [self._row_to_triple(row) for row in rows]
        
        # Update access timestamps
        if triples:
            await self._mark_accessed([t.id for t in triples if t.id])
        
        return triples
    
    async def query_by_object(
        self,
        user_id: str,
        obj: str,
        predicate: str = None,
        min_confidence: float = 0.3
    ) -> List[Triple]:
        """Get all triples where the given entity is the object."""
        obj = self._normalize_entity(obj)
        
        query = """
            SELECT id, user_id, subject, predicate, object, confidence,
                   source_memory_id, created_at, last_accessed, access_count, metadata
            FROM kg_triples
            WHERE user_id=? AND object=? AND confidence >= ?
        """
        params = [user_id, obj, min_confidence]
        
        if predicate:
            query += " AND predicate=?"
            params.append(self._normalize_predicate(predicate))
        
        query += " ORDER BY confidence DESC, last_accessed DESC"
        
        rows = await self.db.fetchall(query, params)
        triples = [self._row_to_triple(row) for row in rows]
        
        if triples:
            await self._mark_accessed([t.id for t in triples if t.id])
        
        return triples
    
    async def query_entity_relations(
        self,
        user_id: str,
        entity: str,
        max_results: int = 20
    ) -> Dict[str, List[Triple]]:
        """
        Get all relationships involving an entity (as subject or object).
        Returns dict with 'outgoing' and 'incoming' triple lists.
        """
        entity = self._normalize_entity(entity)
        
        outgoing = await self.query_by_subject(user_id, entity)
        incoming = await self.query_by_object(user_id, entity)
        
        return {
            'entity': entity,
            'outgoing': outgoing[:max_results],
            'incoming': incoming[:max_results],
            'total_connections': len(outgoing) + len(incoming)
        }
    
    async def traverse(
        self,
        user_id: str,
        start_entity: str,
        max_hops: int = 2,
        min_confidence: float = 0.4
    ) -> Dict[str, Set[str]]:
        """
        Traverse the graph from a starting entity.
        Returns entities reachable within max_hops.
        """
        start_entity = self._normalize_entity(start_entity)
        visited: Set[str] = {start_entity}
        frontier: Set[str] = {start_entity}
        
        hop_results: Dict[str, Set[str]] = {f"hop_0": {start_entity}}
        
        for hop in range(1, max_hops + 1):
            next_frontier: Set[str] = set()
            
            for entity in frontier:
                # Get outgoing connections
                outgoing = await self.query_by_subject(user_id, entity, min_confidence=min_confidence)
                for triple in outgoing:
                    if triple.object not in visited:
                        visited.add(triple.object)
                        next_frontier.add(triple.object)
                
                # Get incoming connections
                incoming = await self.query_by_object(user_id, entity, min_confidence=min_confidence)
                for triple in incoming:
                    if triple.subject not in visited:
                        visited.add(triple.subject)
                        next_frontier.add(triple.subject)
            
            hop_results[f"hop_{hop}"] = next_frontier
            frontier = next_frontier
            
            if not frontier:
                break
        
        return hop_results
    
    async def find_path(
        self,
        user_id: str,
        start: str,
        end: str,
        max_hops: int = 3
    ) -> Optional[List[Triple]]:
        """Find a path between two entities (BFS)."""
        start = self._normalize_entity(start)
        end = self._normalize_entity(end)
        
        if start == end:
            return []
        
        # BFS with path tracking
        queue: List[Tuple[str, List[Triple]]] = [(start, [])]
        visited: Set[str] = {start}
        
        while queue and len(queue[0][1]) < max_hops:
            current, path = queue.pop(0)
            
            outgoing = await self.query_by_subject(user_id, current)
            for triple in outgoing:
                if triple.object == end:
                    return path + [triple]
                
                if triple.object not in visited:
                    visited.add(triple.object)
                    queue.append((triple.object, path + [triple]))
        
        return None
    
    async def get_entity_summary(self, user_id: str, entity: str) -> Dict:
        """Get a summary of an entity and its relationships."""
        entity = self._normalize_entity(entity)
        
        # Get entity metadata
        entity_row = await self.db.fetchone("""
            SELECT name, entity_type, aliases, properties, first_seen, mention_count
            FROM kg_entities WHERE user_id=? AND name=?
        """, (user_id, entity))
        
        # Get relationships
        relations = await self.query_entity_relations(user_id, entity)
        
        # Group predicates
        predicate_summary = defaultdict(list)
        for triple in relations['outgoing']:
            predicate_summary[triple.predicate].append({
                'object': triple.object,
                'confidence': triple.confidence
            })
        
        return {
            'entity': entity,
            'type': entity_row[1] if entity_row else 'unknown',
            'aliases': json.loads(entity_row[2]) if entity_row and entity_row[2] else [],
            'properties': json.loads(entity_row[3]) if entity_row and entity_row[3] else {},
            'first_seen': entity_row[4] if entity_row else None,
            'mention_count': entity_row[5] if entity_row else 0,
            'relationships': dict(predicate_summary),
            'total_outgoing': len(relations['outgoing']),
            'total_incoming': len(relations['incoming'])
        }
    
    async def extract_and_store(
        self,
        user_id: str,
        text: str,
        source_memory_id: str = None,
        extracted_entities: List[Dict] = None
    ) -> int:
        """
        Extract entities and relationships from text and store them.
        Uses pre-extracted entities if provided, otherwise simple pattern matching.
        Returns count of triples added.
        """
        if extracted_entities:
            return await self._store_extracted_entities(user_id, extracted_entities, source_memory_id)
        
        # Simple pattern-based extraction (fallback)
        triples_added = 0
        
        # Common patterns: "X is Y", "X has Y", "X likes Y"
        patterns = [
            (r"(\w+) is (?:a |an )?(\w+)", "is_a"),
            (r"(\w+) has (\w+)", "has"),
            (r"(\w+) likes (\w+)", "likes"),
            (r"(\w+) works at (\w+)", "works_at"),
            (r"(\w+) lives in (\w+)", "lives_in"),
        ]
        
        import re
        for pattern, predicate in patterns:
            matches = re.findall(pattern, text.lower())
            for subject, obj in matches:
                result = await self.add_triple(
                    user_id, subject, predicate, obj,
                    confidence=0.6,
                    source_memory_id=source_memory_id
                )
                if result:
                    triples_added += 1
        
        return triples_added
    
    async def _store_extracted_entities(
        self,
        user_id: str,
        entities: List[Dict],
        source_memory_id: str
    ) -> int:
        """Store pre-extracted entity relationships."""
        triples_added = 0
        
        for entity_data in entities:
            subject = entity_data.get('entity') or entity_data.get('subject')
            predicate = entity_data.get('relation') or entity_data.get('predicate')
            obj = entity_data.get('value') or entity_data.get('object')
            confidence = entity_data.get('confidence', 0.7)
            
            if subject and predicate and obj:
                result = await self.add_triple(
                    user_id, subject, predicate, obj,
                    confidence=confidence,
                    source_memory_id=source_memory_id,
                    metadata={'source': 'extraction'}
                )
                if result:
                    triples_added += 1
        
        return triples_added
    
    async def apply_decay(self, user_id: str = None) -> int:
        """Apply confidence decay to old, unused triples."""
        cutoff = datetime.now() - timedelta(days=self.DECAY_HALF_LIFE_DAYS)
        
        query = """
            UPDATE kg_triples
            SET confidence = confidence * 0.9
            WHERE last_accessed < ? AND confidence > ?
        """
        params = [cutoff, self.MIN_CONFIDENCE_THRESHOLD]
        
        if user_id:
            query = query.replace("WHERE", "WHERE user_id=? AND")
            params = [user_id] + params
        
        await self.db.execute(query, params)
        
        # Remove very low confidence triples
        delete_query = "DELETE FROM kg_triples WHERE confidence < ?"
        if user_id:
            delete_query += " AND user_id=?"
            await self.db.execute(delete_query, (self.MIN_CONFIDENCE_THRESHOLD, user_id))
        else:
            await self.db.execute(delete_query, (self.MIN_CONFIDENCE_THRESHOLD,))
        
        logger.info("[KG] Applied decay to old triples")
        return 0
    
    async def get_stats(self, user_id: str) -> Dict:
        """Get knowledge graph statistics for a user."""
        triple_count = await self.db.fetchone(
            "SELECT COUNT(*) FROM kg_triples WHERE user_id=?", (user_id,)
        )
        
        entity_count = await self.db.fetchone(
            "SELECT COUNT(*) FROM kg_entities WHERE user_id=?", (user_id,)
        )
        
        predicate_counts = await self.db.fetchall("""
            SELECT predicate, COUNT(*) as cnt 
            FROM kg_triples WHERE user_id=?
            GROUP BY predicate ORDER BY cnt DESC LIMIT 10
        """, (user_id,))
        
        avg_confidence = await self.db.fetchone(
            "SELECT AVG(confidence) FROM kg_triples WHERE user_id=?", (user_id,)
        )
        
        return {
            'total_triples': triple_count[0] if triple_count else 0,
            'total_entities': entity_count[0] if entity_count else 0,
            'top_predicates': [(row[0], row[1]) for row in predicate_counts] if predicate_counts else [],
            'avg_confidence': round(avg_confidence[0], 3) if avg_confidence and avg_confidence[0] else 0.0
        }
    
    async def _ensure_entity(self, user_id: str, name: str):
        """Ensure an entity exists in the entities table."""
        name = self._normalize_entity(name)
        
        existing = await self.db.fetchone(
            "SELECT id FROM kg_entities WHERE user_id=? AND name=?",
            (user_id, name)
        )
        
        if existing:
            await self.db.execute(
                "UPDATE kg_entities SET mention_count = mention_count + 1 WHERE id=?",
                (existing[0],)
            )
        else:
            await self.db.execute("""
                INSERT INTO kg_entities (user_id, name, first_seen)
                VALUES (?, ?, ?)
            """, (user_id, name, datetime.now()))
    
    async def _mark_accessed(self, triple_ids: List[int]):
        """Update last_accessed timestamp for triples."""
        if not triple_ids:
            return
        
        placeholders = ','.join(['?'] * len(triple_ids))
        await self.db.execute(f"""
            UPDATE kg_triples 
            SET last_accessed=?, access_count=access_count+1
            WHERE id IN ({placeholders})
        """, [datetime.now()] + triple_ids)
    
    def _normalize_entity(self, entity: str) -> str:
        """Normalize entity name for consistent storage."""
        if not entity:
            return ""
        return entity.strip().lower()
    
    def _normalize_predicate(self, predicate: str) -> str:
        """Normalize predicate for consistent storage."""
        if not predicate:
            return ""
        return predicate.strip().lower().replace(' ', '_')
    
    def _row_to_triple(self, row) -> Triple:
        """Convert database row to Triple object."""
        return Triple(
            id=row[0],
            user_id=row[1],
            subject=row[2],
            predicate=row[3],
            object=row[4],
            confidence=row[5],
            source_memory_id=row[6],
            created_at=row[7],
            last_accessed=row[8],
            access_count=row[9],
            metadata=json.loads(row[10]) if row[10] else None
        )
    
    def get_predicate_category(self, predicate: str) -> str:
        """Get semantic category for a predicate."""
        predicate = self._normalize_predicate(predicate)
        
        for category, predicates in self.PREDICATE_CATEGORIES.items():
            if predicate in predicates:
                return category
        
        return 'other'
