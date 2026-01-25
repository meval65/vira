"""
Hippocampus Module - Memory and Knowledge Graph Management for Vira.

This module handles all memory operations including:
- Long-term memory storage and retrieval
- Knowledge graph (triple store) management
- Admin profile management
- Schedule management
- Persona management

Refactored to use MongoDB instead of SQLite for better scalability
and native document storage for embeddings.
"""

import os
import uuid
import json
import hashlib
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from bson import ObjectId

from src.brainstem import (
    GOOGLE_API_KEY, TIER_2_MODEL, TIER_3_MODEL,
    MemoryType, MAX_RETRIEVED_MEMORIES, MIN_RELEVANCE_SCORE,
    DECAY_DAYS_EMOTION, DECAY_DAYS_GENERAL, CANONICALIZATION_INSTRUCTION,
    NeuralEventBus
)
from src.db.mongo_client import get_mongo_client, MongoDBClient


class TripleRelation(str, Enum):
    HAS = "has"
    IS = "is"
    LIKES = "likes"
    DISLIKES = "dislikes"
    WORKS_AT = "works_at"
    LIVES_IN = "lives_in"
    KNOWS = "knows"
    RELATED_TO = "related_to"
    CREATED = "created"
    OWNS = "owns"


@dataclass
class Memory:
    id: str
    summary: str
    memory_type: str
    priority: float = 0.5
    confidence: float = 0.5
    fingerprint: Optional[str] = None
    entity: Optional[str] = None
    relation: Optional[str] = None
    value: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    use_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    status: str = "active"


@dataclass
class Triple:
    id: Optional[str]
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_memory_id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0


class AdminProfile(BaseModel):
    telegram_name: Optional[str] = None
    full_name: Optional[str] = None
    additional_info: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.now)


class Hippocampus:
    """
    Memory management system using MongoDB.
    
    Handles long-term memory storage, knowledge graph operations,
    and semantic search using NumPy for local vector similarity.
    """
    
    SIMILARITY_THRESHOLD: float = 0.90
    ARCHIVE_CHECK_DAYS: int = 90
    MIN_CONFIDENCE: float = 0.3

    def __init__(self):
        self._mongo: Optional[MongoDBClient] = None
        self._admin_profile: AdminProfile = AdminProfile()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._genai_client = None
        self._evolution_history: Dict[str, List[Dict]] = defaultdict(list)
        self._conflict_history: Dict[str, List[Dict]] = defaultdict(list)

    async def initialize(self) -> None:
        """Initialize MongoDB connection and load admin profile."""
        self._mongo = get_mongo_client()
        await self._mongo.connect()
        
        if GOOGLE_API_KEY:
            self._genai_client = genai.Client(api_key=GOOGLE_API_KEY)
        
        await self._load_admin_profile()

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._mongo:
            await self._mongo.close()
            self._mongo = None

    @property
    def admin_profile(self) -> AdminProfile:
        return self._admin_profile

    # ==================== ADMIN PROFILE ====================

    async def _load_admin_profile(self) -> None:
        """Load admin profile from MongoDB."""
        doc = await self._mongo.admin_profile.find_one({"_id": "admin"})
        if doc:
            self._admin_profile = AdminProfile(
                telegram_name=doc.get("telegram_name"),
                full_name=doc.get("full_name"),
                additional_info=doc.get("additional_info"),
                last_updated=doc.get("last_updated", datetime.now())
            )

    async def update_admin_profile(
        self,
        telegram_name: Optional[str] = None,
        full_name: Optional[str] = None,
        additional_info: Optional[str] = None
    ) -> None:
        """Update admin profile in MongoDB."""
        update_fields = {"last_updated": datetime.now()}
        if telegram_name:
            update_fields["telegram_name"] = telegram_name
            self._admin_profile.telegram_name = telegram_name
        if full_name:
            update_fields["full_name"] = full_name
            self._admin_profile.full_name = full_name
        if additional_info:
            update_fields["additional_info"] = additional_info
            self._admin_profile.additional_info = additional_info
        
        self._admin_profile.last_updated = update_fields["last_updated"]
        
        await self._mongo.admin_profile.update_one(
            {"_id": "admin"},
            {"$set": update_fields},
            upsert=True
        )

    # ==================== MEMORY OPERATIONS ====================

    async def store(
        self,
        summary: str,
        memory_type: str,
        priority: float = 0.5,
        embedding: Optional[List[float]] = None
    ) -> str:
        """Store a new memory or merge with existing one."""
        await NeuralEventBus.set_activity("hippocampus", f"Storing {memory_type}")
        await NeuralEventBus.emit("hippocampus", "hippocampus", f"store_memory:{memory_type}")
        
        canonical = await self._canonicalize(summary, memory_type)
        fingerprint = canonical.get("fingerprint")
        
        # Check for existing memory with same fingerprint
        if fingerprint:
            existing = await self._find_by_fingerprint(fingerprint)
            if existing:
                merged = await self._merge_memories(existing, canonical)
                await self._update_memory(existing["_id"], merged)
                self._record_evolution(fingerprint, existing.get("summary"), summary, "merge")
                await NeuralEventBus.clear_activity("hippocampus")
                return str(existing["_id"])
        
        # Create new memory
        memory_id = str(uuid.uuid4())
        doc = {
            "_id": memory_id,
            "summary": summary,
            "type": memory_type,
            "priority": priority,
            "embedding": embedding,  # Store as array directly
            "fingerprint": canonical.get("fingerprint"),
            "entity": canonical.get("entity"),
            "relation": canonical.get("relation"),
            "value": canonical.get("value"),
            "confidence": canonical.get("confidence", 0.5),
            "created_at": datetime.now(),
            "last_used": datetime.now(),
            "use_count": 0,
            "status": "active"
        }
        
        await self._mongo.memories.insert_one(doc)
        
        if fingerprint:
            self._record_evolution(fingerprint, None, summary, "create")
        
        await NeuralEventBus.clear_activity("hippocampus")
        return memory_id

    async def recall(
        self,
        query: str,
        limit: int = MAX_RETRIEVED_MEMORIES,
        query_embedding: Optional[List[float]] = None
    ) -> List[Memory]:
        """Retrieve relevant memories using vector similarity."""
        await NeuralEventBus.set_activity("hippocampus", "Recalling Memories")
        await NeuralEventBus.emit("hippocampus", "hippocampus", "recall_memory")
        
        # Fetch candidate memories
        cursor = self._mongo.memories.find(
            {"status": "active"}
        ).sort([("priority", -1), ("last_used", -1)]).limit(limit * 3)
        
        docs = await cursor.to_list(length=limit * 3)
        memories = []
        
        for doc in docs:
            emb = None
            if doc.get("embedding"):
                try:
                    emb = np.array(doc["embedding"], dtype=np.float32)
                except Exception:
                    pass
            
            memories.append(Memory(
                id=str(doc["_id"]),
                summary=doc.get("summary", ""),
                memory_type=doc.get("type", "general"),
                priority=doc.get("priority", 0.5),
                confidence=doc.get("confidence", 0.5),
                fingerprint=doc.get("fingerprint"),
                entity=doc.get("entity"),
                relation=doc.get("relation"),
                value=doc.get("value"),
                embedding=emb,
                use_count=doc.get("use_count", 0),
                created_at=doc.get("created_at", datetime.now()),
                last_used_at=doc.get("last_used", datetime.now()),
                status=doc.get("status", "active")
            ))
        
        # Vector similarity search using NumPy
        if query_embedding and memories:
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            
            scored = []
            for m in memories:
                if m.embedding is not None:
                    m_norm = m.embedding / (np.linalg.norm(m.embedding) + 1e-10)
                    sim = float(np.dot(query_norm, m_norm))
                    if sim >= MIN_RELEVANCE_SCORE:
                        scored.append((m, sim))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            memories = [m for m, _ in scored[:limit]]
        
        # Mark memories as used
        for m in memories[:limit]:
            await self._mark_memory_used(m.id)
        
        await NeuralEventBus.clear_activity("hippocampus")
        return memories[:limit]

    async def query_entity(self, entity: str) -> Dict[str, Any]:
        """Query knowledge graph for entity relationships."""
        entity_lower = entity.lower().strip()
        
        # Outgoing relationships
        outgoing_cursor = self._mongo.knowledge_graph.find(
            {"subject": entity_lower}
        ).sort("confidence", -1).limit(10)
        outgoing = await outgoing_cursor.to_list(length=10)
        
        # Incoming relationships
        incoming_cursor = self._mongo.knowledge_graph.find(
            {"object": entity_lower}
        ).sort("confidence", -1).limit(10)
        incoming = await incoming_cursor.to_list(length=10)
        
        # Related memories
        memories_cursor = self._mongo.memories.find(
            {"entity": entity_lower, "status": "active"}
        ).sort("priority", -1).limit(5)
        related_memories = await memories_cursor.to_list(length=5)
        
        # Update access counts
        for doc in outgoing + incoming:
            await self._mongo.knowledge_graph.update_one(
                {"_id": doc["_id"]},
                {"$set": {"last_accessed": datetime.now()}, "$inc": {"access_count": 1}}
            )
        
        return {
            "entity": entity,
            "outgoing": [
                {"subject": r["subject"], "predicate": r["predicate"], "object": r["object"], "confidence": r.get("confidence", 0.8)}
                for r in outgoing
            ],
            "incoming": [
                {"subject": r["subject"], "predicate": r["predicate"], "object": r["object"], "confidence": r.get("confidence", 0.8)}
                for r in incoming
            ],
            "memories": [
                {"id": str(m["_id"]), "summary": m["summary"], "type": m.get("type"), "confidence": m.get("confidence", 0.5)}
                for m in related_memories
            ]
        }

    async def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source_memory_id: Optional[str] = None
    ) -> str:
        """Add a knowledge graph triple."""
        await NeuralEventBus.emit("hippocampus", "hippocampus", "add_knowledge")
        
        subject = subject.lower().strip()
        predicate = predicate.lower().strip().replace(" ", "_")
        obj = obj.lower().strip()
        
        await self._ensure_entity(subject)
        await self._ensure_entity(obj)
        
        # Check for existing triple
        existing = await self._mongo.knowledge_graph.find_one({
            "subject": subject,
            "predicate": predicate,
            "object": obj
        })
        
        if existing:
            new_conf = min(1.0, existing.get("confidence", 0.8) + 0.1)
            await self._mongo.knowledge_graph.update_one(
                {"_id": existing["_id"]},
                {"$set": {"confidence": new_conf, "last_accessed": datetime.now()}}
            )
            return str(existing["_id"])
        
        # Insert new triple
        triple_id = str(uuid.uuid4())
        await self._mongo.knowledge_graph.insert_one({
            "_id": triple_id,
            "subject": subject,
            "predicate": predicate,
            "object": obj,
            "confidence": confidence,
            "source_memory_id": source_memory_id,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0
        })
        return triple_id

    async def traverse(
        self,
        start_entity: str,
        max_hops: int = 2,
        min_confidence: float = 0.4
    ) -> Dict[str, List[Triple]]:
        """Traverse knowledge graph from starting entity."""
        start = start_entity.lower().strip()
        visited = {start}
        result: Dict[str, List[Triple]] = {start: []}
        
        current_level = [start]
        for hop in range(max_hops):
            next_level = []
            for entity in current_level:
                cursor = self._mongo.knowledge_graph.find({
                    "$or": [
                        {"subject": entity},
                        {"object": entity}
                    ],
                    "confidence": {"$gte": min_confidence}
                })
                
                async for doc in cursor:
                    triple = Triple(
                        id=str(doc["_id"]),
                        subject=doc["subject"],
                        predicate=doc["predicate"],
                        object=doc["object"],
                        confidence=doc.get("confidence", 0.8),
                        source_memory_id=doc.get("source_memory_id"),
                        created_at=doc.get("created_at"),
                        last_accessed=doc.get("last_accessed"),
                        access_count=doc.get("access_count", 0)
                    )
                    
                    if entity not in result:
                        result[entity] = []
                    result[entity].append(triple)
                    
                    neighbor = doc["object"] if doc["subject"].lower() == entity else doc["subject"]
                    if neighbor.lower() not in visited:
                        visited.add(neighbor.lower())
                        next_level.append(neighbor.lower())
            
            current_level = next_level
        
        return result

    # ==================== SCHEDULE OPERATIONS ====================

    async def add_schedule(
        self,
        trigger_time: datetime,
        context: str,
        priority: int = 0,
        recurring: Optional[str] = None
    ) -> str:
        """Add a new schedule."""
        schedule_id = str(uuid.uuid4())
        await self._mongo.schedules.insert_one({
            "_id": schedule_id,
            "scheduled_at": trigger_time,
            "context": context,
            "priority": priority,
            "recurrence": recurring,
            "status": "pending",
            "created_at": datetime.now(),
            "executed_at": None,
            "execution_note": None
        })
        return schedule_id

    async def get_pending_schedules(self, limit: int = 10) -> List[Dict]:
        """Get pending schedules that are due."""
        cursor = self._mongo.schedules.find({
            "status": "pending",
            "scheduled_at": {"$lte": datetime.now()}
        }).sort("scheduled_at", 1).limit(limit)
        
        docs = await cursor.to_list(length=limit)
        return [
            {
                "id": str(d["_id"]),
                "scheduled_at": d["scheduled_at"],
                "context": d["context"],
                "priority": d.get("priority", 0),
                "recurrence": d.get("recurrence")
            }
            for d in docs
        ]

    async def get_upcoming_schedules(self, hours_ahead: int = 72) -> List[Dict]:
        """Get upcoming schedules within specified hours."""
        future = datetime.now() + timedelta(hours=hours_ahead)
        cursor = self._mongo.schedules.find({
            "status": "pending",
            "scheduled_at": {"$lte": future}
        }).sort("scheduled_at", 1)
        
        docs = await cursor.to_list(length=50)
        return [
            {
                "id": str(d["_id"]),
                "scheduled_at": d["scheduled_at"],
                "context": d["context"],
                "priority": d.get("priority", 0),
                "recurrence": d.get("recurrence")
            }
            for d in docs
        ]

    async def mark_schedule_executed(self, schedule_id: str, note: Optional[str] = None) -> None:
        """Mark a schedule as executed."""
        await self._mongo.schedules.update_one(
            {"_id": schedule_id},
            {"$set": {
                "status": "executed",
                "executed_at": datetime.now(),
                "execution_note": note
            }}
        )

    async def get_failed_schedules(self, limit: int = 10) -> List[Dict]:
        """Get failed schedules."""
        cursor = self._mongo.schedules.find(
            {"status": "failed"}
        ).sort("scheduled_at", -1).limit(limit)
        
        docs = await cursor.to_list(length=limit)
        return [dict(d) for d in docs]

    async def get_schedule_stats(self) -> Dict[str, int]:
        """Get schedule statistics."""
        pending = await self._mongo.schedules.count_documents({"status": "pending"})
        executed = await self._mongo.schedules.count_documents({"status": "executed"})
        return {"pending": pending, "executed": executed}

    # ==================== PERSONA OPERATIONS ====================

    async def get_personas(self) -> List[Dict]:
        """Get all personas."""
        cursor = self._mongo.personas.find().sort("name", 1)
        docs = await cursor.to_list(length=100)
        return [
            {
                "id": str(d["_id"]),
                "name": d["name"],
                "instruction": d["instruction"],
                "temperature": d.get("temperature", 0.7),
                "is_active": d.get("is_active", False),
                "created_at": d.get("created_at")
            }
            for d in docs
        ]

    async def get_active_persona(self) -> Optional[Dict]:
        """Get the currently active persona."""
        doc = await self._mongo.personas.find_one({"is_active": True})
        if doc:
            return {
                "id": str(doc["_id"]),
                "name": doc["name"],
                "instruction": doc["instruction"],
                "temperature": doc.get("temperature", 0.7),
                "is_active": True
            }
        return None

    async def create_persona(self, name: str, instruction: str, temperature: float = 0.7) -> str:
        """Create a new persona."""
        count = await self._mongo.personas.count_documents({})
        is_first = count == 0
        
        persona_id = str(uuid.uuid4())
        await self._mongo.personas.insert_one({
            "_id": persona_id,
            "name": name,
            "instruction": instruction,
            "temperature": temperature,
            "is_active": is_first,
            "created_at": datetime.now()
        })
        return persona_id

    async def update_persona(self, persona_id: str, data: Dict) -> bool:
        """Update a persona."""
        update_fields = {}
        for k in ["name", "instruction", "temperature"]:
            if k in data and data[k] is not None:
                update_fields[k] = data[k]
        
        if not update_fields:
            return False
        
        result = await self._mongo.personas.update_one(
            {"_id": persona_id},
            {"$set": update_fields}
        )
        return result.modified_count > 0

    async def delete_persona(self, persona_id: str) -> bool:
        """Delete a persona."""
        doc = await self._mongo.personas.find_one({"_id": persona_id})
        if not doc:
            return False
        
        if doc.get("is_active"):
            raise ValueError("Cannot delete active persona")
        
        result = await self._mongo.personas.delete_one({"_id": persona_id})
        return result.deleted_count > 0

    async def set_active_persona(self, persona_id: str) -> bool:
        """Set a persona as active."""
        await self._mongo.personas.update_many({}, {"$set": {"is_active": False}})
        result = await self._mongo.personas.update_one(
            {"_id": persona_id},
            {"$set": {"is_active": True}}
        )
        return result.modified_count > 0

    # ==================== EMOTIONAL STATE ====================

    async def save_emotional_state(
        self,
        mood: str,
        empathy: float,
        satisfaction: float,
        mood_history: List[Dict]
    ) -> None:
        """Save emotional state to MongoDB."""
        await self._mongo.emotional_state.update_one(
            {"_id": "state"},
            {"$set": {
                "current_mood": mood,
                "empathy_level": empathy,
                "satisfaction_level": satisfaction,
                "last_interaction": datetime.now(),
                "mood_history": mood_history[-10:],
                "updated_at": datetime.now()
            }},
            upsert=True
        )

    async def load_emotional_state(self) -> Optional[Dict]:
        """Load emotional state from MongoDB."""
        doc = await self._mongo.emotional_state.find_one({"_id": "state"})
        if doc:
            return {
                "mood": doc.get("current_mood", "neutral"),
                "empathy": doc.get("empathy_level", 0.5),
                "satisfaction": doc.get("satisfaction_level", 0.0),
                "last_interaction": doc.get("last_interaction"),
                "history": doc.get("mood_history", [])
            }
        return None

    # ==================== STATS ====================

    async def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        active = await self._mongo.memories.count_documents({"status": "active"})
        archived = await self._mongo.memories.count_documents({"status": "archived"})
        triples = await self._mongo.knowledge_graph.count_documents({})
        return {"active": active, "archived": archived, "triples": triples}

    # ==================== INTERNAL HELPERS ====================

    async def _ensure_entity(self, name: str) -> None:
        """Ensure entity exists in entities collection."""
        name_lower = name.lower().strip()
        result = await self._mongo.entities.update_one(
            {"name": name_lower},
            {
                "$inc": {"mention_count": 1},
                "$setOnInsert": {
                    "entity_type": "unknown",
                    "first_seen": datetime.now()
                }
            },
            upsert=True
        )

    async def _find_by_fingerprint(self, fingerprint: str) -> Optional[Dict]:
        """Find memory by fingerprint."""
        return await self._mongo.memories.find_one({
            "fingerprint": fingerprint,
            "status": "active"
        })

    async def _update_memory(self, memory_id: str, data: Dict) -> None:
        """Update an existing memory."""
        await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": {
                "summary": data.get("summary"),
                "priority": data.get("priority", 0.5),
                "confidence": data.get("confidence", 0.5),
                "value": data.get("value"),
                "last_used": datetime.now()
            }, "$inc": {"use_count": 1}}
        )

    async def _mark_memory_used(self, memory_id: str) -> None:
        """Mark a memory as used."""
        await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": {"last_used": datetime.now()}, "$inc": {"use_count": 1}}
        )

    async def _canonicalize(self, summary: str, memory_type: str) -> Dict:
        """Convert natural language to structured format using LLM."""
        if not self._genai_client:
            return self._fallback_canonicalize(summary, memory_type)
        
        try:
            prompt = f"{CANONICALIZATION_INSTRUCTION}\n\nInput: \"{summary}\""
            response = self._genai_client.models.generate_content(
                model=TIER_3_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256
                )
            )
            text = response.text.strip()
            return self._extract_json(text) or self._fallback_canonicalize(summary, memory_type)
        except Exception:
            return self._fallback_canonicalize(summary, memory_type)

    def _fallback_canonicalize(self, summary: str, memory_type: str) -> Dict:
        """Fallback canonicalization without LLM."""
        words = summary.lower().split()
        entity = words[-1] if words else "unknown"
        relation = "related_to"
        
        for w in ["likes", "loves", "enjoys", "prefers"]:
            if w in words:
                relation = "likes"
                break
        for w in ["hates", "dislikes"]:
            if w in words:
                relation = "dislikes"
                break
        for w in ["is", "am", "are"]:
            if w in words:
                relation = "is"
                break
        
        fingerprint = hashlib.md5(f"{memory_type}:{relation}:{entity}".encode()).hexdigest()[:16]
        
        return {
            "fingerprint": f"{memory_type}:{relation}:{fingerprint}",
            "type": memory_type,
            "entity": entity,
            "relation": relation,
            "value": True,
            "confidence": 0.6
        }

    async def _merge_memories(self, existing: Dict, new: Dict) -> Dict:
        """Merge new data into existing memory."""
        new_conf = min(1.0, existing.get("confidence", 0.5) + 0.1)
        new_priority = min(1.0, existing.get("priority", 0.5) + 0.05)
        
        return {
            "summary": new.get("summary", existing.get("summary")),
            "priority": new_priority,
            "confidence": new_conf,
            "value": new.get("value", existing.get("value"))
        }

    def _record_evolution(
        self,
        fingerprint: str,
        old_value: Optional[str],
        new_value: str,
        action: str
    ) -> None:
        """Record memory evolution in history."""
        self._evolution_history[fingerprint].append({
            "old_value": old_value,
            "new_value": new_value,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        import re
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None
