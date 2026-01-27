import os
import uuid
import json
import hashlib
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field

from pydantic import BaseModel, Field
from bson import ObjectId

from src.brainstem import (
    MemoryType, MAX_RETRIEVED_MEMORIES, MIN_RELEVANCE_SCORE,
    DECAY_DAYS_EMOTION, DECAY_DAYS_GENERAL, CANONICALIZATION_INSTRUCTION,
    MEMORY_COMPRESSION_INSTRUCTION, NeuralEventBus, get_openrouter_client
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
    is_compressed: bool = False

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
    semantic search using NumPy, and automatic memory compression.
    """
    
    SIMILARITY_THRESHOLD: float = 0.90
    ARCHIVE_CHECK_DAYS: int = 90
    MIN_CONFIDENCE: float = 0.3
    
    COMPRESSION_THRESHOLD: int = 5
    TOP_MEMORIES_COUNT: int = 30
    
    WEIGHT_RECENCY: float = 0.3
    WEIGHT_FREQUENCY: float = 0.3
    WEIGHT_PRIORITY: float = 0.4

    def __init__(self):
        self._mongo: Optional[MongoDBClient] = None
        self._admin_profile: AdminProfile = AdminProfile()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._openrouter = None
        self._evolution_history: Dict[str, List[Dict]] = defaultdict(list)
        self._conflict_history: Dict[str, List[Dict]] = defaultdict(list)

    async def initialize(self) -> None:
        """Initialize MongoDB connection and load admin profile."""
        self._mongo = get_mongo_client()
        await self._mongo.connect()
        self._openrouter = get_openrouter_client()
        await self._load_admin_profile()
        await self._ensure_default_persona()

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._mongo:
            await self._mongo.close()
            self._mongo = None

    @property
    def admin_profile(self) -> AdminProfile:
        return self._admin_profile

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

    async def store(
        self,
        summary: str,
        memory_type: str,
        priority: float = 0.5,
        embedding: Optional[List[float]] = None
    ) -> str:
        """Store a new memory or merge with existing one."""
        await NeuralEventBus.set_activity("hippocampus", f"Storing {memory_type}")
        await NeuralEventBus.emit("hippocampus", "hippocampus", f"store_memory:{memory_type}", payload={
            "summary_len": len(summary),
            "priority": priority
        })
        
        canonical = await self._canonicalize(summary, memory_type)
        fingerprint = canonical.get("fingerprint")
        
        if fingerprint:
            existing = await self._find_by_fingerprint(fingerprint)
            if existing:
                merged = await self._merge_memories(existing, canonical)
                await self._update_memory(existing["_id"], merged)
                self._record_evolution(fingerprint, existing.get("summary"), summary, "merge")
                await NeuralEventBus.clear_activity("hippocampus")
                return str(existing["_id"])
        
        memory_id = str(uuid.uuid4())
        doc = {
            "_id": memory_id,
            "summary": summary,
            "type": memory_type,
            "priority": priority,
            "embedding": embedding,
            "fingerprint": canonical.get("fingerprint"),
            "entity": canonical.get("entity"),
            "relation": canonical.get("relation"),
            "value": canonical.get("value"),
            "confidence": canonical.get("confidence", 0.5),
            "created_at": datetime.now(),
            "last_used": datetime.now(),
            "use_count": 0,
            "status": "active",
            "is_compressed": False
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
        await NeuralEventBus.emit("hippocampus", "hippocampus", "recall_memory", payload={
            "query": query[:50] + "..." if len(query) > 50 else query
        })
        
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
                status=doc.get("status", "active"),
                is_compressed=doc.get("is_compressed", False)
            ))
        
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
        
        for m in memories[:limit]:
            await self._mark_memory_used(m.id)
        
        await NeuralEventBus.clear_activity("hippocampus")
        return memories[:limit]

    async def query_entity(self, entity: str) -> Dict[str, Any]:
        """Query knowledge graph for entity relationships."""
        entity_lower = entity.lower().strip()
        
        outgoing_cursor = self._mongo.knowledge_graph.find(
            {"subject": entity_lower}
        ).sort("confidence", -1).limit(10)
        outgoing = await outgoing_cursor.to_list(length=10)
        
        incoming_cursor = self._mongo.knowledge_graph.find(
            {"object": entity_lower}
        ).sort("confidence", -1).limit(10)
        incoming = await incoming_cursor.to_list(length=10)
        
        memories_cursor = self._mongo.memories.find(
            {"entity": entity_lower, "status": "active"}
        ).sort("priority", -1).limit(5)
        related_memories = await memories_cursor.to_list(length=5)
        
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
        await NeuralEventBus.emit("hippocampus", "hippocampus", "add_knowledge", payload={
            "subject": subject,
            "predicate": predicate,
            "object": obj
        })
        
        subject = subject.lower().strip()
        predicate = predicate.lower().strip().replace(" ", "_")
        obj = obj.lower().strip()
        
        await self._ensure_entity(subject)
        await self._ensure_entity(obj)
        
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

    async def check_and_compress_memories(self) -> bool:
        """
        Check if memory compression is needed and execute if so.
        
        Returns True if compression was performed.
        """
        uncompressed_count = await self._count_uncompressed_memories()
        
        if uncompressed_count < self.COMPRESSION_THRESHOLD:
            return False
        
        await NeuralEventBus.set_activity("hippocampus", "Compressing Memories")
        await NeuralEventBus.emit("hippocampus", "cerebellum", "compression_started", payload={
            "uncompressed_count": uncompressed_count
        })
        
        try:
            memories = await self._select_top_memories(self.TOP_MEMORIES_COUNT)
            
            if len(memories) < 5:
                await NeuralEventBus.clear_activity("hippocampus")
                return False
            
            summary = await self._compress_memories(memories)
            
            if summary:
                await self._store_global_context(summary, [m["_id"] for m in memories])
                
                await self._mark_memories_compressed([m["_id"] for m in memories])
                
                await NeuralEventBus.emit("hippocampus", "thalamus", "global_context_updated", payload={
                    "memory_count": len(memories),
                    "summary_length": len(summary)
                })
                
                await self._log_compression(len(memories), summary)
            
            await NeuralEventBus.clear_activity("hippocampus")
            return True
            
        except Exception as e:
            print(f"⚠️ Memory compression failed: {e}")
            await NeuralEventBus.clear_activity("hippocampus")
            return False

    async def _count_uncompressed_memories(self) -> int:
        """Count memories that haven't been compressed yet."""
        return await self._mongo.memories.count_documents({
            "status": "active",
            "is_compressed": {"$ne": True}
        })

    async def _select_top_memories(self, count: int) -> List[Dict]:
        """
        Select top memories using weighted scoring.
        
        Weights:
        - Recency (30%): How recent the memory is
        - Frequency (30%): How often it's been accessed
        - Priority (40%): Assigned importance level
        """
        now = datetime.now()
        
        cursor = self._mongo.memories.find({"status": "active"}).limit(200)
        docs = await cursor.to_list(length=200)
        
        scored_memories = []
        for doc in docs:
            score = self._calculate_memory_score(doc, now)
            scored_memories.append((doc, score))
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_memories[:count]]

    def _calculate_memory_score(self, memory: Dict, now: datetime) -> float:
        """
        Calculate composite score for memory selection.
        """
        created_at = memory.get("created_at", now)
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except:
                created_at = now
        
        age_hours = (now - created_at).total_seconds() / 3600
        recency_score = math.exp(-age_hours / 168)
        
        use_count = memory.get("use_count", 0)
        frequency_score = math.log(use_count + 1) / math.log(100) if use_count > 0 else 0
        frequency_score = min(1.0, frequency_score)
        
        priority_score = memory.get("priority", 0.5)
        
        return (
            self.WEIGHT_RECENCY * recency_score +
            self.WEIGHT_FREQUENCY * frequency_score +
            self.WEIGHT_PRIORITY * priority_score
        )

    async def _compress_memories(self, memories: List[Dict]) -> Optional[str]:
        """
        Compress memories into a narrative summary using LLM.
        """
        if not self._openrouter:
            return None
        
        memory_texts = []
        for i, m in enumerate(memories, 1):
            created = m.get("created_at", datetime.now())
            if isinstance(created, datetime):
                created_str = created.strftime("%Y-%m-%d")
            else:
                created_str = str(created)[:10]
            
            memory_texts.append(
                f"{i}. [{m.get('type', 'general')}] (Priority: {m.get('priority', 0.5):.1f}, Created: {created_str})\n"
                f"   {m.get('summary', '')}"
            )
        
        prompt = f"""{MEMORY_COMPRESSION_INSTRUCTION}

# MEMORIES TO COMPRESS ({len(memories)} entries)

{chr(10).join(memory_texts)}

# OUTPUT
Generate a single coherent paragraph summarizing the user's profile, preferences, and important information:"""

        try:
            response = await self._openrouter.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=800,
                preferred_tier="tier_2"
            )
            return response.content.strip()
        except Exception as e:
            print(f"⚠️ Memory compression LLM call failed: {e}")
            return None

    async def _store_global_context(self, summary: str, source_memory_ids: List[str]) -> None:
        """Store compressed summary as global context."""
        await self._mongo.db["global_context"].update_one(
            {"_id": "current"},
            {"$set": {
                "summary": summary,
                "source_memory_ids": source_memory_ids,
                "memory_count": len(source_memory_ids),
                "created_at": datetime.now(),
                "compression_version": await self._get_next_compression_version()
            }},
            upsert=True
        )

    async def _get_next_compression_version(self) -> int:
        """Get the next compression version number."""
        doc = await self._mongo.db["global_context"].find_one({"_id": "current"})
        if doc:
            return doc.get("compression_version", 0) + 1
        return 1

    async def _mark_memories_compressed(self, memory_ids: List[str]) -> None:
        """Mark memories as compressed."""
        await self._mongo.memories.update_many(
            {"_id": {"$in": memory_ids}},
            {"$set": {"is_compressed": True, "compressed_at": datetime.now()}}
        )

    async def _log_compression(self, memory_count: int, summary: str) -> None:
        """Log compression event."""
        await self._mongo.db["compression_log"].insert_one({
            "timestamp": datetime.now(),
            "memories_processed": memory_count,
            "output_length": len(summary),
            "status": "success"
        })

    async def get_global_context(self) -> Optional[str]:
        """Get the current global context summary."""
        doc = await self._mongo.db["global_context"].find_one({"_id": "current"})
        if doc:
            return doc.get("summary")
        return None

    async def get_compression_stats(self) -> Dict[str, Any]:
        """Get memory compression statistics."""
        global_ctx = await self._mongo.db["global_context"].find_one({"_id": "current"})
        uncompressed = await self._count_uncompressed_memories()
        total_active = await self._mongo.memories.count_documents({"status": "active"})
        
        logs_cursor = self._mongo.db["compression_log"].find().sort("timestamp", -1).limit(5)
        recent_logs = await logs_cursor.to_list(length=5)
        
        return {
            "uncompressed_count": uncompressed,
            "total_active_memories": total_active,
            "compression_threshold": self.COMPRESSION_THRESHOLD,
            "global_context_exists": global_ctx is not None,
            "global_context_length": len(global_ctx.get("summary", "")) if global_ctx else 0,
            "last_compression": global_ctx.get("created_at").isoformat() if global_ctx and global_ctx.get("created_at") else None,
            "compression_version": global_ctx.get("compression_version", 0) if global_ctx else 0,
            "recent_compressions": [
                {
                    "timestamp": log.get("timestamp").isoformat() if log.get("timestamp") else None,
                    "memories_processed": log.get("memories_processed", 0),
                    "status": log.get("status", "unknown")
                }
                for log in recent_logs
            ]
        }

    async def add_schedule(
        self,
        trigger_time: datetime,
        context: str,
        priority: int = 0,
        recurring: Optional[str] = None
    ) -> str:
        """Add a new schedule."""
        conflicts = await self.detect_schedule_conflicts(trigger_time)
        if conflicts:
            await NeuralEventBus.emit("hippocampus", "prefrontal_cortex", "schedule_conflict", payload={
                "new_time": trigger_time.isoformat(),
                "conflicts": len(conflicts)
            })
        
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

    async def detect_schedule_conflicts(
        self,
        new_time: datetime,
        duration_minutes: int = 60
    ) -> List[Dict]:
        """Detect potential conflicts with existing schedules."""
        window_start = new_time - timedelta(minutes=30)
        window_end = new_time + timedelta(minutes=duration_minutes + 30)
        
        cursor = self._mongo.schedules.find({
            "status": "pending",
            "scheduled_at": {
                "$gte": window_start,
                "$lte": window_end
            }
        })
        
        conflicts = await cursor.to_list(length=10)
        return [
            {
                "id": str(c["_id"]),
                "scheduled_at": c["scheduled_at"].isoformat() if isinstance(c["scheduled_at"], datetime) else c["scheduled_at"],
                "context": c.get("context", "")
            }
            for c in conflicts
        ]

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

    async def cleanup_old_schedules(self, days_old: int = 30) -> int:
        """Clean up old executed schedules."""
        cutoff = datetime.now() - timedelta(days=days_old)
        result = await self._mongo.schedules.delete_many({
            "status": {"$in": ["executed", "cancelled"]},
            "executed_at": {"$lt": cutoff}
        })
        return result.deleted_count

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

    async def _ensure_default_persona(self) -> None:
        """Ensure at least one default persona exists."""
        count = await self._mongo.personas.count_documents({})
        if count == 0:
            from src.brainstem import DEFAULT_PERSONA_INSTRUCTION
            await self.create_persona(
                name="Default Assistant",
                instruction=DEFAULT_PERSONA_INSTRUCTION,
                temperature=0.7
            )

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
        
        await NeuralEventBus.emit("hippocampus", "dashboard", "persona_created", payload={
            "persona_id": persona_id,
            "name": name
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
        
        if result.modified_count > 0:
            persona = await self.get_active_persona()
            await NeuralEventBus.emit("hippocampus", "prefrontal_cortex", "persona_changed", payload={
                "persona_id": persona_id,
                "name": persona.get("name") if persona else "Unknown"
            })
        
        return result.modified_count > 0

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

    async def apply_memory_decay(self) -> int:
        """Apply decay to old memories while preserving important ones."""
        now = datetime.now()
        decayed_count = 0
        
        cursor = self._mongo.memories.find({
            "status": "active",
            "last_used": {"$lt": now - timedelta(days=30)}
        }).limit(100)
        
        docs = await cursor.to_list(length=100)
        
        for memory in docs:
            priority = memory.get("priority", 0.5)
            decay_factor = 0.1 * (1 - priority)
            
            new_confidence = max(0.1, memory.get("confidence", 0.5) - decay_factor)
            
            if new_confidence < 0.2:
                await self._mongo.memories.update_one(
                    {"_id": memory["_id"]},
                    {"$set": {"status": "archived", "archived_at": now}}
                )
            else:
                await self._mongo.memories.update_one(
                    {"_id": memory["_id"]},
                    {"$set": {"confidence": new_confidence}}
                )
            
            decayed_count += 1
        
        return decayed_count

    async def consolidate_memories(self) -> int:
        """Consolidate similar memories (placeholder for future implementation)."""
        return 0

    async def optimize_knowledge_graph(self) -> None:
        """Optimize knowledge graph by removing low-confidence triples."""
        cutoff = datetime.now() - timedelta(days=90)
        await self._mongo.knowledge_graph.delete_many({
            "confidence": {"$lt": 0.3},
            "last_accessed": {"$lt": cutoff}
        })

    async def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        active = await self._mongo.memories.count_documents({"status": "active"})
        archived = await self._mongo.memories.count_documents({"status": "archived"})
        triples = await self._mongo.knowledge_graph.count_documents({})
        uncompressed = await self._count_uncompressed_memories()
        return {
            "active": active,
            "archived": archived,
            "triples": triples,
            "uncompressed": uncompressed
        }

    async def _ensure_entity(self, name: str) -> None:
        """Ensure entity exists in entities collection."""
        name_lower = name.lower().strip()
        await self._mongo.entities.update_one(
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
                "last_used": datetime.now(),
                "is_compressed": False
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
        if not self._openrouter:
            return self._fallback_canonicalize(summary, memory_type)
        
        try:
            prompt = f"{CANONICALIZATION_INSTRUCTION}\n\nInput: \"{summary}\""
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.1
            )
            return self._extract_json(response) or self._fallback_canonicalize(summary, memory_type)
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
