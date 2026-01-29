import os
import uuid
import json
import hashlib
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
import asyncio
from functools import lru_cache

from pydantic import BaseModel, Field
from bson import ObjectId

from src.brain.brainstem import (
    MemoryType, MAX_RETRIEVED_MEMORIES, MIN_RELEVANCE_SCORE,
    DECAY_DAYS_EMOTION, DECAY_DAYS_GENERAL, NeuralEventBus
)
from src.brain.constants import CANONICALIZATION_INSTRUCTION, MEMORY_COMPRESSION_INSTRUCTION
from src.brain.db.mongo_client import get_mongo_client, MongoDBClient

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
    MEMBER_OF = "member_of"
    PART_OF = "part_of"
    CAUSES = "causes"
    LOCATED_IN = "located_in"

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
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.priority < 0:
            self.priority = 0.0
        elif self.priority > 1:
            self.priority = 1.0
        if self.confidence < 0:
            self.confidence = 0.0
        elif self.confidence > 1:
            self.confidence = 1.0

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
    
    def __post_init__(self):
        if self.confidence < 0:
            self.confidence = 0.0
        elif self.confidence > 1:
            self.confidence = 1.0

class AdminProfile(BaseModel):
    telegram_name: Optional[str] = None
    full_name: Optional[str] = None
    additional_info: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)

class Hippocampus:
    
    SIMILARITY_THRESHOLD: float = 0.90
    ARCHIVE_CHECK_DAYS: int = 90
    MIN_CONFIDENCE: float = 0.3
    
    COMPRESSION_THRESHOLD: int = 5
    TOP_MEMORIES_COUNT: int = 30
    
    WEIGHT_RECENCY: float = 0.3
    WEIGHT_FREQUENCY: float = 0.3
    WEIGHT_PRIORITY: float = 0.4
    
    BATCH_SIZE: int = 100
    CACHE_SIZE: int = 1000

    def __init__(self):
        self._mongo: Optional[MongoDBClient] = None
        self._admin_profile: AdminProfile = AdminProfile()
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._fingerprint_cache: Dict[str, str] = {}
        self._openrouter = None
        self._evolution_history: Dict[str, List[Dict]] = defaultdict(list)
        self._conflict_history: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._persona_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl: int = 300
        self._brain = None

    def bind_brain(self, brain) -> None:
        self._brain = brain
        if brain:
            self._openrouter = brain.openrouter

    async def initialize(self) -> None:
        self._mongo = get_mongo_client()
        await self._mongo.connect()
        await self._load_admin_profile()
        await self._ensure_default_persona()
        await self._create_indexes()

    async def close(self) -> None:
        if self._mongo:
            await self._mongo.close()
            self._mongo = None
        self._embedding_cache.clear()
        self._fingerprint_cache.clear()

    @property
    def admin_profile(self) -> AdminProfile:
        return self._admin_profile

    async def _create_indexes(self) -> None:
        try:
            await self._mongo.memories.create_index([("fingerprint", 1), ("status", 1)])
            await self._mongo.memories.create_index([("status", 1), ("priority", -1), ("last_used", -1)])
            await self._mongo.memories.create_index([("created_at", -1)])
            await self._mongo.memories.create_index([("type", 1)])
            await self._mongo.knowledge_graph.create_index([("subject", 1)])
            await self._mongo.knowledge_graph.create_index([("confidence", 1), ("last_accessed", 1)])
            await self._mongo.entities.create_index([("name", 1)], unique=True)
        except Exception:
            pass

    async def _load_admin_profile(self) -> None:
        doc = await self._mongo.admin_profile.find_one({"_id": "admin"})
        if doc:
            self._admin_profile = AdminProfile(
                telegram_name=doc.get("telegram_name"),
                full_name=doc.get("full_name"),
                additional_info=doc.get("additional_info"),
                preferences=doc.get("preferences", {}),
                last_updated=doc.get("last_updated", datetime.now())
            )

    async def update_admin_profile(
        self,
        telegram_name: Optional[str] = None,
        full_name: Optional[str] = None,
        additional_info: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> None:
        async with self._lock:
            update_fields = {"last_updated": datetime.now()}
            if telegram_name is not None:
                update_fields["telegram_name"] = telegram_name
                self._admin_profile.telegram_name = telegram_name
            if full_name is not None:
                update_fields["full_name"] = full_name
                self._admin_profile.full_name = full_name
            if additional_info is not None:
                update_fields["additional_info"] = additional_info
                self._admin_profile.additional_info = additional_info
            if preferences is not None:
                update_fields["preferences"] = preferences
                self._admin_profile.preferences = preferences
            
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
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        if not summary or not summary.strip():
            raise ValueError("Summary cannot be empty")
        
        priority = max(0.0, min(1.0, priority))
        
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
            "summary": summary.strip(),
            "type": memory_type,
            "priority": priority,
            "embedding": embedding,
            "fingerprint": canonical.get("fingerprint"),
            "entity": canonical.get("entity"),
            "relation": canonical.get("relation"),
            "value": canonical.get("value"),
            "confidence": max(0.0, min(1.0, canonical.get("confidence", 0.5))),
            "created_at": datetime.now(),
            "last_used": datetime.now(),
            "use_count": 0,
            "status": "active",
            "is_compressed": False,
            "tags": tags or []
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
        query_embedding: Optional[List[float]] = None,
        memory_type: Optional[str] = None
    ) -> List[Memory]:
        await NeuralEventBus.set_activity("hippocampus", "Recalling Memories")
        await NeuralEventBus.emit("hippocampus", "hippocampus", "recall_memory", payload={
            "query": query[:50] + "..." if len(query) > 50 else query
        })
        
        filter_query = {"status": "active"}
        if memory_type:
            filter_query["type"] = memory_type
        
        cursor = self._mongo.memories.find(filter_query).sort([
            ("priority", -1), 
            ("last_used", -1)
        ]).limit(min(limit * 3, 300))
        
        docs = await cursor.to_list(length=min(limit * 3, 300))
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
                is_compressed=doc.get("is_compressed", False),
                tags=doc.get("tags", [])
            ))
        
        if query_embedding and memories:
            query_emb = np.array(query_embedding, dtype=np.float32)
            scored_memories = []
            for mem in memories:
                if mem.embedding is not None and len(mem.embedding) > 0:
                    try:
                        similarity = self._cosine_similarity(query_emb, mem.embedding)
                        scored_memories.append((mem, similarity))
                    except Exception:
                        scored_memories.append((mem, 0.0))
                else:
                    scored_memories.append((mem, 0.0))
            
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            memories = [mem for mem, _ in scored_memories[:limit]]
        else:
            memories = memories[:limit]
        
        memory_ids = [mem.id for mem in memories]
        if memory_ids:
            await self._mongo.memories.update_many(
                {"_id": {"$in": memory_ids}},
                {"$set": {"last_used": datetime.now()}, "$inc": {"use_count": 1}}
            )
        
        await NeuralEventBus.clear_activity("hippocampus")
        return memories

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        if len(a) == 0 or len(b) == 0:
            return 0.0
        try:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0

    async def delete_memory(self, memory_id: str) -> bool:
        result = await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": {"status": "deleted", "deleted_at": datetime.now()}}
        )
        return result.modified_count > 0

    async def archive_old_memories(self, days_threshold: int = ARCHIVE_CHECK_DAYS) -> int:
        cutoff = datetime.now() - timedelta(days=days_threshold)
        result = await self._mongo.memories.update_many(
            {
                "status": "active",
                "last_used": {"$lt": cutoff},
                "priority": {"$lt": 0.3}
            },
            {"$set": {"status": "archived", "archived_at": datetime.now()}}
        )
        return result.modified_count

    async def compress_memories(self) -> int:
        if not self._openrouter:
            return 0
        
        cursor = self._mongo.memories.find({
            "status": "active",
            "type": "general",
            "is_compressed": False
        }).sort("created_at", 1).limit(self.TOP_MEMORIES_COUNT)
        
        docs = await cursor.to_list(length=self.TOP_MEMORIES_COUNT)
        if len(docs) < self.COMPRESSION_THRESHOLD:
            return 0
        
        summaries = [doc["summary"] for doc in docs]
        
        try:
            prompt = f"{MEMORY_COMPRESSION_INSTRUCTION}\n\nMemories:\n" + "\n".join(
                f"{i+1}. {s}" for i, s in enumerate(summaries)
            )
            
            compressed = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.3,
                tier="tier_1"
            )
            
            compressed_id = str(uuid.uuid4())
            await self._mongo.memories.insert_one({
                "_id": compressed_id,
                "summary": compressed.strip(),
                "type": "general",
                "priority": 0.7,
                "confidence": 0.8,
                "created_at": datetime.now(),
                "last_used": datetime.now(),
                "use_count": 0,
                "status": "active",
                "is_compressed": True,
                "source_count": len(docs),
                "tags": ["compressed"]
            })
            
            memory_ids = [doc["_id"] for doc in docs]
            await self._mongo.memories.update_many(
                {"_id": {"$in": memory_ids}},
                {"$set": {"status": "compressed", "compressed_into": compressed_id}}
            )
            
            return len(docs)
        except Exception:
            return 0

    async def _count_uncompressed_memories(self) -> int:
        return await self._mongo.memories.count_documents({
            "status": "active",
            "type": "general",
            "is_compressed": False
        })

    async def store_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source_memory_id: Optional[str] = None
    ) -> str:
        if not subject or not predicate or not obj:
            raise ValueError("Subject, predicate, and object cannot be empty")
        
        confidence = max(0.0, min(1.0, confidence))
        
        await self._ensure_entity(subject)
        await self._ensure_entity(obj)
        
        existing = await self._mongo.knowledge_graph.find_one({
            "subject": subject.lower().strip(),
            "predicate": predicate.lower().strip(),
            "object": obj.lower().strip()
        })
        
        if existing:
            new_confidence = min(1.0, existing.get("confidence", 0.5) + 0.1)
            await self._mongo.knowledge_graph.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": {
                        "confidence": new_confidence,
                        "last_accessed": datetime.now()
                    },
                    "$inc": {"access_count": 1}
                }
            )
            return str(existing["_id"])
        
        triple_id = str(uuid.uuid4())
        await self._mongo.knowledge_graph.insert_one({
            "_id": triple_id,
            "subject": subject.lower().strip(),
            "predicate": predicate.lower().strip(),
            "object": obj.lower().strip(),
            "confidence": confidence,
            "source_memory_id": source_memory_id,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0
        })
        
        return triple_id

    async def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        min_confidence: float = MIN_CONFIDENCE
    ) -> List[Triple]:
        query_filter = {"confidence": {"$gte": min_confidence}}
        
        if subject:
            query_filter["subject"] = subject.lower().strip()
        if predicate:
            query_filter["predicate"] = predicate.lower().strip()
        if obj:
            query_filter["object"] = obj.lower().strip()
        
        cursor = self._mongo.knowledge_graph.find(query_filter).sort("confidence", -1).limit(100)
        docs = await cursor.to_list(length=100)
        
        triples = []
        for doc in docs:
            triples.append(Triple(
                id=str(doc["_id"]),
                subject=doc["subject"],
                predicate=doc["predicate"],
                object=doc["object"],
                confidence=doc.get("confidence", 0.8),
                source_memory_id=doc.get("source_memory_id"),
                created_at=doc.get("created_at"),
                last_accessed=doc.get("last_accessed"),
                access_count=doc.get("access_count", 0)
            ))
        
        if triples:
            triple_ids = [t.id for t in triples]
            await self._mongo.knowledge_graph.update_many(
                {"_id": {"$in": triple_ids}},
                {
                    "$set": {"last_accessed": datetime.now()},
                    "$inc": {"access_count": 1}
                }
            )
        
        return triples

    async def get_entity_relations(self, entity: str, limit: int = 50) -> List[Triple]:
        entity_lower = entity.lower().strip()
        cursor = self._mongo.knowledge_graph.find({
            "$or": [
                {"subject": entity_lower},
                {"object": entity_lower}
            ]
        }).sort("confidence", -1).limit(limit)
        
        docs = await cursor.to_list(length=limit)
        return [Triple(
            id=str(doc["_id"]),
            subject=doc["subject"],
            predicate=doc["predicate"],
            object=doc["object"],
            confidence=doc.get("confidence", 0.8),
            source_memory_id=doc.get("source_memory_id"),
            created_at=doc.get("created_at"),
            last_accessed=doc.get("last_accessed"),
            access_count=doc.get("access_count", 0)
        ) for doc in docs]

    async def infer_knowledge(self, entity: str) -> Dict[str, List[str]]:
        relations = await self.get_entity_relations(entity)
        knowledge = defaultdict(list)
        
        for triple in relations:
            if triple.subject.lower() == entity.lower():
                knowledge[triple.predicate].append(triple.object)
            else:
                knowledge[f"inverse_{triple.predicate}"].append(triple.subject)
        
        return dict(knowledge)

    async def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3
    ) -> Optional[List[Triple]]:
        start_lower = start_entity.lower().strip()
        end_lower = end_entity.lower().strip()
        
        if start_lower == end_lower:
            return []
        
        visited = {start_lower}
        queue = [(start_lower, [])]
        
        for _ in range(max_depth):
            if not queue:
                break
            
            next_queue = []
            for current, path in queue:
                relations = await self.get_entity_relations(current, limit=20)
                
                for triple in relations:
                    next_entity = None
                    if triple.subject.lower() == current:
                        next_entity = triple.object.lower()
                    elif triple.object.lower() == current:
                        next_entity = triple.subject.lower()
                    
                    if next_entity and next_entity not in visited:
                        new_path = path + [triple]
                        
                        if next_entity == end_lower:
                            return new_path
                        
                        visited.add(next_entity)
                        next_queue.append((next_entity, new_path))
            
            queue = next_queue
        
        return None

    async def create_persona(
        self,
        name: str,
        description: str,
        traits: Dict[str, float],
        voice_tone: str = "friendly",
        identity_anchor: Optional[str] = None
    ) -> str:
        persona_id = str(uuid.uuid4())
        await self._mongo.personas.insert_one({
            "_id": persona_id,
            "name": name,
            "description": description,
            "traits": traits,
            "voice_tone": voice_tone,
            "is_active": False,
            "calibration": {
                "emotional_inertia": 0.7,
                "base_arousal": 0.0,
                "base_valence": 0.0,
                "calibration_status": False,
                "identity_anchor": identity_anchor or description
            },
            "created_at": datetime.now(),
            "last_modified": datetime.now()
        })
        self._persona_cache = None
        return persona_id

    async def update_persona(
        self,
        persona_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        traits: Optional[Dict[str, float]] = None,
        voice_tone: Optional[str] = None
    ) -> bool:
        update_fields = {"last_modified": datetime.now()}
        
        if name is not None:
            update_fields["name"] = name
        if description is not None:
            update_fields["description"] = description
        if traits is not None:
            update_fields["traits"] = traits
        if voice_tone is not None:
            update_fields["voice_tone"] = voice_tone
        
        result = await self._mongo.personas.update_one(
            {"_id": persona_id},
            {"$set": update_fields}
        )
        
        if result.modified_count > 0:
            self._persona_cache = None
        
        return result.modified_count > 0

    async def delete_persona(self, persona_id: str) -> bool:
        persona = await self._mongo.personas.find_one({"_id": persona_id})
        if not persona:
            return False
        
        if persona.get("is_active"):
            await self._ensure_default_persona()
        
        result = await self._mongo.personas.delete_one({"_id": persona_id})
        if result.deleted_count > 0:
            self._persona_cache = None
        return result.deleted_count > 0

    async def list_personas(self) -> List[Dict]:
        cursor = self._mongo.personas.find({}).sort("created_at", -1)
        docs = await cursor.to_list(length=100)
        return [{
            "id": str(doc["_id"]),
            "name": doc.get("name", ""),
            "description": doc.get("description", ""),
            "traits": doc.get("traits", {}),
            "voice_tone": doc.get("voice_tone", "friendly"),
            "is_active": doc.get("is_active", False),
            "created_at": doc.get("created_at"),
            "last_modified": doc.get("last_modified")
        } for doc in docs]

    async def get_active_persona(self) -> Optional[Dict]:
        now = datetime.now()
        if (self._persona_cache and self._cache_timestamp and 
            (now - self._cache_timestamp).total_seconds() < self._cache_ttl):
            return self._persona_cache
        
        doc = await self._mongo.personas.find_one({"is_active": True})
        if doc:
            calibration = doc.get("calibration", {})
            self._persona_cache = {
                "id": str(doc["_id"]),
                "name": doc.get("name", ""),
                "description": doc.get("description", ""),
                "traits": doc.get("traits", {}),
                "voice_tone": doc.get("voice_tone", "friendly"),
                "calibration": {
                    "emotional_inertia": calibration.get("emotional_inertia", 0.7),
                    "base_arousal": calibration.get("base_arousal", 0.0),
                    "base_valence": calibration.get("base_valence", 0.0),
                    "calibration_status": calibration.get("calibration_status", False),
                    "identity_anchor": calibration.get("identity_anchor", doc.get("description", ""))
                }
            }
            self._cache_timestamp = now
            return self._persona_cache
        return None

    async def _ensure_default_persona(self) -> None:
        existing = await self._mongo.personas.find_one({"is_active": True})
        if existing:
            return
        
        default = await self._mongo.personas.find_one({"name": "Default"})
        if not default:
            default_id = str(uuid.uuid4())
            await self._mongo.personas.insert_one({
                "_id": default_id,
                "name": "Default",
                "description": "Balanced and helpful assistant",
                "traits": {
                    "formality": 0.5,
                    "enthusiasm": 0.6,
                    "verbosity": 0.5,
                    "creativity": 0.6
                },
                "voice_tone": "friendly",
                "is_active": True,
                "calibration": {
                    "emotional_inertia": 0.7,
                    "base_arousal": 0.0,
                    "base_valence": 0.1,
                    "calibration_status": True,
                    "identity_anchor": "Balanced and helpful assistant with friendly demeanor"
                },
                "created_at": datetime.now(),
                "last_modified": datetime.now()
            })
        else:
            await self._mongo.personas.update_one(
                {"_id": default["_id"]},
                {"$set": {"is_active": True}}
            )
        self._persona_cache = None

    async def switch_persona(self, persona_id: str) -> bool:
        await self._mongo.personas.update_many({}, {"$set": {"is_active": False}})
        result = await self._mongo.personas.update_one(
            {"_id": persona_id},
            {"$set": {"is_active": True}}
        )
        
        if result.modified_count > 0:
            self._persona_cache = None
            persona = await self.get_active_persona()
            await NeuralEventBus.emit("hippocampus", "prefrontal_cortex", "persona_changed", payload={
                "persona_id": persona_id,
                "name": persona.get("name") if persona else "Unknown"
            })
        
        return result.modified_count > 0

    async def get_persona_calibration(self, persona_id: str) -> Optional[Dict]:
        doc = await self._mongo.personas.find_one({"_id": persona_id})
        if not doc:
            return None
        
        calibration = doc.get("calibration", {})
        return {
            "emotional_inertia": calibration.get("emotional_inertia", 0.7),
            "base_arousal": calibration.get("base_arousal", 0.0),
            "base_valence": calibration.get("base_valence", 0.0),
            "calibration_status": calibration.get("calibration_status", False),
            "identity_anchor": calibration.get("identity_anchor", doc.get("description", ""))
        }

    async def update_persona_calibration(
        self,
        persona_id: str,
        emotional_inertia: Optional[float] = None,
        base_arousal: Optional[float] = None,
        base_valence: Optional[float] = None,
        calibration_status: Optional[bool] = None,
        identity_anchor: Optional[str] = None
    ) -> bool:
        update_fields = {"last_modified": datetime.now()}
        
        if emotional_inertia is not None:
            update_fields["calibration.emotional_inertia"] = max(0.0, min(1.0, emotional_inertia))
        if base_arousal is not None:
            update_fields["calibration.base_arousal"] = max(-1.0, min(1.0, base_arousal))
        if base_valence is not None:
            update_fields["calibration.base_valence"] = max(-1.0, min(1.0, base_valence))
        if calibration_status is not None:
            update_fields["calibration.calibration_status"] = calibration_status
        if identity_anchor is not None:
            update_fields["calibration.identity_anchor"] = identity_anchor
        
        result = await self._mongo.personas.update_one(
            {"_id": persona_id},
            {"$set": update_fields}
        )
        
        if result.modified_count > 0:
            self._persona_cache = None
        
        return result.modified_count > 0

    async def save_emotional_state(
        self,
        mood: str,
        empathy: float,
        satisfaction: float,
        mood_history: List[Dict]
    ) -> None:
        empathy = max(0.0, min(1.0, empathy))
        satisfaction = max(0.0, min(1.0, satisfaction))
        
        await self._mongo.emotional_state.update_one(
            {"_id": "state"},
            {"$set": {
                "current_mood": mood,
                "empathy_level": empathy,
                "satisfaction_level": satisfaction,
                "last_interaction": datetime.now(),
                "mood_history": mood_history[-20:],
                "updated_at": datetime.now()
            }},
            upsert=True
        )

    async def load_emotional_state(self) -> Optional[Dict]:
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

    async def get_global_context(self) -> Optional[str]:
        cursor = self._mongo.memories.find({
            "status": "active",
            "is_compressed": True
        }).sort("created_at", -1).limit(3)
        
        docs = await cursor.to_list(length=3)
        if not docs:
            cursor = self._mongo.memories.find({
                "status": "active",
                "priority": {"$gte": 0.7}
            }).sort([("priority", -1), ("last_used", -1)]).limit(10)
            docs = await cursor.to_list(length=10)
        
        if not docs:
            return None
        
        summaries = [doc.get("summary", "") for doc in docs if doc.get("summary")]
        return "\n".join(summaries[:5]) if summaries else None

    async def get_pending_schedules(self, limit: int = 10) -> List[Dict]:
        now = datetime.now()
        cursor = self._mongo.schedules.find({
            "status": {"$in": ["pending", "active"]},
            "scheduled_at": {"$lte": now}
        }).sort("scheduled_at", 1).limit(limit)
        
        docs = await cursor.to_list(length=limit)
        return [{
            "id": str(doc["_id"]),
            "context": doc.get("context", ""),
            "scheduled_at": doc.get("scheduled_at"),
            "status": doc.get("status", "pending")
        } for doc in docs]

    async def get_upcoming_schedules(self, hours_ahead: int = 24) -> List[Dict]:
        now = datetime.now()
        future = now + timedelta(hours=hours_ahead)
        
        cursor = self._mongo.schedules.find({
            "status": {"$in": ["pending", "active"]},
            "scheduled_at": {"$gte": now, "$lte": future}
        }).sort("scheduled_at", 1).limit(20)
        
        docs = await cursor.to_list(length=20)
        return [{
            "id": str(doc["_id"]),
            "context": doc.get("context", ""),
            "scheduled_at": doc.get("scheduled_at"),
            "status": doc.get("status", "pending")
        } for doc in docs]

    async def query_entity(self, entity_name: str) -> Dict[str, Any]:
        entity_lower = entity_name.lower().strip()
        
        entity_doc = await self._mongo.entities.find_one({"name": entity_lower})
        
        triples = await self.get_entity_relations(entity_lower, limit=10)
        
        cursor = self._mongo.memories.find({
            "status": "active",
            "$or": [
                {"entity": entity_lower},
                {"summary": {"$regex": entity_name, "$options": "i"}}
            ]
        }).sort("priority", -1).limit(5)
        
        memory_docs = await cursor.to_list(length=5)
        
        return {
            "entity": entity_doc if entity_doc else {"name": entity_lower},
            "triples": [{"subject": t.subject, "predicate": t.predicate, "object": t.object} for t in triples],
            "memories": [{"id": str(doc["_id"]), "summary": doc.get("summary", ""), "type": doc.get("type", "general"), "confidence": doc.get("confidence", 0.5)} for doc in memory_docs]
        }

    async def add_schedule(
        self,
        trigger_time: datetime,
        context: str,
        repeat: Optional[str] = None
    ) -> str:
        schedule_id = str(uuid.uuid4())
        
        await self._mongo.schedules.insert_one({
            "_id": schedule_id,
            "context": context,
            "scheduled_at": trigger_time,
            "repeat": repeat,
            "status": "pending",
            "created_at": datetime.now()
        })
        
        return schedule_id

    async def check_and_compress_memories(self) -> bool:
        uncompressed_count = await self._count_uncompressed_memories()
        
        if uncompressed_count >= self.COMPRESSION_THRESHOLD:
            compressed = await self.compress_memories()
            return compressed > 0
        
        return False

    async def apply_memory_decay(self, batch_size: int = None) -> int:
        if batch_size is None:
            batch_size = self.BATCH_SIZE
        
        now = datetime.now()
        decayed_count = 0
        
        cursor = self._mongo.memories.find({
            "status": "active",
            "last_used": {"$lt": now - timedelta(days=30)}
        }).limit(batch_size)
        
        docs = await cursor.to_list(length=batch_size)
        
        bulk_operations = []
        for memory in docs:
            priority = memory.get("priority", 0.5)
            decay_factor = 0.1 * (1 - priority)
            
            new_confidence = max(0.1, memory.get("confidence", 0.5) - decay_factor)
            
            if new_confidence < 0.2:
                bulk_operations.append({
                    "update_one": {
                        "filter": {"_id": memory["_id"]},
                        "update": {"$set": {"status": "archived", "archived_at": now}}
                    }
                })
            else:
                bulk_operations.append({
                    "update_one": {
                        "filter": {"_id": memory["_id"]},
                        "update": {"$set": {"confidence": new_confidence}}
                    }
                })
            
            decayed_count += 1
        
        if bulk_operations:
            await self._mongo.memories.bulk_write(bulk_operations, ordered=False)
        
        return decayed_count

    async def consolidate_memories(self) -> int:
        return 0

    async def optimize_knowledge_graph(self) -> None:
        cutoff = datetime.now() - timedelta(days=90)
        await self._mongo.knowledge_graph.delete_many({
            "confidence": {"$lt": 0.3},
            "last_accessed": {"$lt": cutoff}
        })

    async def get_memory_stats(self) -> Dict[str, int]:
        pipeline = [
            {"$facet": {
                "active": [{"$match": {"status": "active"}}, {"$count": "count"}],
                "archived": [{"$match": {"status": "archived"}}, {"$count": "count"}],
                "compressed": [{"$match": {"is_compressed": True, "status": "active"}}, {"$count": "count"}],
                "uncompressed": [{"$match": {"type": "general", "is_compressed": False, "status": "active"}}, {"$count": "count"}]
            }}
        ]
        
        result = await self._mongo.memories.aggregate(pipeline).to_list(length=1)
        triples = await self._mongo.knowledge_graph.count_documents({})
        
        if result:
            stats = result[0]
            return {
                "active": stats["active"][0]["count"] if stats["active"] else 0,
                "archived": stats["archived"][0]["count"] if stats["archived"] else 0,
                "compressed": stats["compressed"][0]["count"] if stats["compressed"] else 0,
                "uncompressed": stats["uncompressed"][0]["count"] if stats["uncompressed"] else 0,
                "triples": triples
            }
        
        return {
            "active": 0,
            "archived": 0,
            "compressed": 0,
            "uncompressed": 0,
            "triples": triples
        }

    async def search_memories_by_tags(self, tags: List[str], limit: int = 50) -> List[Memory]:
        cursor = self._mongo.memories.find({
            "status": "active",
            "tags": {"$in": tags}
        }).sort("priority", -1).limit(limit)
        
        docs = await cursor.to_list(length=limit)
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
                is_compressed=doc.get("is_compressed", False),
                tags=doc.get("tags", [])
            ))
        
        return memories

    async def _ensure_entity(self, name: str) -> None:
        name_lower = name.lower().strip()
        if not name_lower:
            return
        
        await self._mongo.entities.update_one(
            {"name": name_lower},
            {
                "$inc": {"mention_count": 1},
                "$setOnInsert": {
                    "entity_type": "unknown",
                    "first_seen": datetime.now()
                },
                "$set": {
                    "last_seen": datetime.now()
                }
            },
            upsert=True
        )

    async def _find_by_fingerprint(self, fingerprint: str) -> Optional[Dict]:
        if fingerprint in self._fingerprint_cache:
            return await self._mongo.memories.find_one({
                "_id": self._fingerprint_cache[fingerprint],
                "status": "active"
            })
        
        doc = await self._mongo.memories.find_one({
            "fingerprint": fingerprint,
            "status": "active"
        })
        
        if doc:
            self._fingerprint_cache[fingerprint] = doc["_id"]
        
        return doc

    async def _update_memory(self, memory_id: str, data: Dict) -> None:
        await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": {
                "summary": data.get("summary"),
                "priority": max(0.0, min(1.0, data.get("priority", 0.5))),
                "confidence": max(0.0, min(1.0, data.get("confidence", 0.5))),
                "value": data.get("value"),
                "last_used": datetime.now(),
                "is_compressed": False
            }, "$inc": {"use_count": 1}}
        )

    async def _mark_memory_used(self, memory_id: str) -> None:
        await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": {"last_used": datetime.now()}, "$inc": {"use_count": 1}}
        )

    async def _canonicalize(self, summary: str, memory_type: str) -> Dict:
        cache_key = f"{memory_type}:{summary[:100]}"
        
        if not self._openrouter:
            return self._fallback_canonicalize(summary, memory_type)
        
        try:
            prompt = f"{CANONICALIZATION_INSTRUCTION}\n\nInput: \"{summary}\""
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.1
            )
            result = self._extract_json(response)
            return result if result else self._fallback_canonicalize(summary, memory_type)
        except Exception:
            return self._fallback_canonicalize(summary, memory_type)

    def _fallback_canonicalize(self, summary: str, memory_type: str) -> Dict:
        words = summary.lower().split()
        entity = words[-1] if words else "unknown"
        relation = "related_to"
        
        relation_keywords = {
            "likes": ["likes", "loves", "enjoys", "prefers", "appreciates"],
            "dislikes": ["hates", "dislikes", "avoids", "despises"],
            "is": ["is", "am", "are", "was", "were"],
            "has": ["has", "have", "owns", "possesses"],
            "works_at": ["works", "employed", "job"],
            "lives_in": ["lives", "resides", "located"]
        }
        
        for rel, keywords in relation_keywords.items():
            if any(kw in words for kw in keywords):
                relation = rel
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
        if len(self._evolution_history[fingerprint]) > 50:
            self._evolution_history[fingerprint] = self._evolution_history[fingerprint][-25:]
        
        self._evolution_history[fingerprint].append({
            "old_value": old_value,
            "new_value": new_value,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })

    def _extract_json(self, text: str) -> Optional[Dict]:
        import re
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
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

    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        doc = await self._mongo.memories.find_one({"_id": memory_id})
        if not doc:
            return None
        
        emb = None
        if doc.get("embedding"):
            try:
                emb = np.array(doc["embedding"], dtype=np.float32)
            except Exception:
                pass
        
        return Memory(
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
            is_compressed=doc.get("is_compressed", False),
            tags=doc.get("tags", [])
        )

    async def update_memory_tags(self, memory_id: str, tags: List[str]) -> bool:
        result = await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": {"tags": tags}}
        )
        return result.modified_count > 0

    async def get_related_memories(self, memory_id: str, limit: int = 10) -> List[Memory]:
        source_memory = await self.get_memory_by_id(memory_id)
        if not source_memory or source_memory.embedding is None:
            return []
        
        cursor = self._mongo.memories.find({
            "status": "active",
            "_id": {"$ne": memory_id},
            "embedding": {"$exists": True}
        }).limit(100)
        
        docs = await cursor.to_list(length=100)
        scored_memories = []
        
        for doc in docs:
            if doc.get("embedding"):
                try:
                    emb = np.array(doc["embedding"], dtype=np.float32)
                    similarity = self._cosine_similarity(source_memory.embedding, emb)
                    if similarity > 0.7:
                        scored_memories.append((doc, similarity))
                except Exception:
                    continue
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        memories = []
        for doc, _ in scored_memories[:limit]:
            emb = np.array(doc["embedding"], dtype=np.float32)
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
                is_compressed=doc.get("is_compressed", False),
                tags=doc.get("tags", [])
            ))
        
        return memories

    async def store_tool_output(
        self,
        tool_name: str,
        output_type: str,
        content: str,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        output_id = str(uuid.uuid4())
        doc = {
            "_id": output_id,
            "tool_name": tool_name,
            "output_type": output_type,
            "content": content,
            "file_path": file_path,
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "status": "active"
        }
        await self._mongo.tool_outputs.insert_one(doc)
        return output_id

    async def get_tool_output(self, output_id: str) -> Optional[Dict]:
        doc = await self._mongo.tool_outputs.find_one({"_id": output_id})
        if doc:
            return {
                "id": str(doc["_id"]),
                "tool_name": doc.get("tool_name", ""),
                "output_type": doc.get("output_type", "text"),
                "content": doc.get("content", ""),
                "file_path": doc.get("file_path"),
                "metadata": doc.get("metadata", {}),
                "created_at": doc.get("created_at")
            }
        return None

    async def get_recent_tool_outputs(
        self,
        tool_name: Optional[str] = None,
        output_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        query = {"status": "active"}
        if tool_name:
            query["tool_name"] = tool_name
        if output_type:
            query["output_type"] = output_type
        cursor = self._mongo.tool_outputs.find(query).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [{
            "id": str(doc["_id"]),
            "tool_name": doc.get("tool_name", ""),
            "output_type": doc.get("output_type", "text"),
            "content": doc.get("content", "")[:200],
            "file_path": doc.get("file_path"),
            "created_at": doc.get("created_at")
        } for doc in docs]

    async def get_recent_chat_history(self, limit: int = 25) -> List[Dict]:
        cursor = self._mongo.chat_logs.find({}).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        chats = []
        for doc in reversed(docs):
            chats.append({
                "role": doc.get("role", "user"),
                "content": doc.get("content", "")[:500],
                "timestamp": doc.get("timestamp")
            })
        return chats

    async def generate_insight(self, chat_summary: str) -> Optional[Dict]:
        if not self._openrouter:
            return None
        
        memories = await self.recall(chat_summary, limit=10)
        if not memories:
            return None
        
        memory_texts = [f"- {m.summary}" for m in memories[:5]]
        memories_context = "\n".join(memory_texts)
        
        prompt = f"""Kamu adalah Vira, AI yang sedang "melamun" dan merefleksikan percakapan terbaru dengan pemilikmu.

PERCAKAPAN TERBARU (ringkasan):
{chat_summary}

MEMORI YANG MUNGKIN TERKAIT:
{memories_context}

INSTRUKSI:
1. Cari hubungan menarik atau pola antara percakapan terbaru dan memori lama
2. Jika ada insight yang bisa berguna untuk percakapan berikutnya, buat dalam format natural
3. Jika tidak ada hubungan yang menarik, jawab dengan "TIDAK_ADA_INSIGHT"

Format output (jika ada insight):
KONEKSI: [deskripsi hubungan yang ditemukan]
INSIGHT: [kalimat natural yang bisa disampaikan ke user, seperti: "Oh iya, kemarin kamu bilang mau hemat, tapi hari ini bahas kopi terus ya hehe"]

Berikan insight yang terasa personal dan caring, bukan formal."""

        try:
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            if "TIDAK_ADA_INSIGHT" in response.upper():
                return None
            
            connection = ""
            insight_text = ""
            
            lines = response.strip().split("\n")
            for line in lines:
                if line.startswith("KONEKSI:"):
                    connection = line.replace("KONEKSI:", "").strip()
                elif line.startswith("INSIGHT:"):
                    insight_text = line.replace("INSIGHT:", "").strip()
            
            if not insight_text:
                return None
            
            memory_ids = [m.id for m in memories[:3]]
            
            return {
                "source_chat_summary": chat_summary[:500],
                "related_memory_ids": memory_ids,
                "connection": connection,
                "insight_text": insight_text,
                "relevance_score": 0.7
            }
        except Exception:
            return None

    async def store_insight(self, insight_data: Dict) -> str:
        insight_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(days=7)
        
        doc = {
            "_id": insight_id,
            "source_chat_summary": insight_data.get("source_chat_summary", ""),
            "related_memory_ids": insight_data.get("related_memory_ids", []),
            "connection": insight_data.get("connection", ""),
            "insight_text": insight_data.get("insight_text", ""),
            "relevance_score": insight_data.get("relevance_score", 0.5),
            "is_used": False,
            "created_at": now,
            "expires_at": expires_at
        }
        
        await self._mongo.insights.insert_one(doc)
        return insight_id

    async def get_relevant_insights(self, query: str, limit: int = 3) -> List[Dict]:
        cursor = self._mongo.insights.find({
            "is_used": False
        }).sort([("relevance_score", -1), ("created_at", -1)]).limit(limit * 2)
        
        docs = await cursor.to_list(length=limit * 2)
        
        query_lower = query.lower()
        scored = []
        for doc in docs:
            summary = doc.get("source_chat_summary", "").lower()
            connection = doc.get("connection", "").lower()
            
            relevance = 0.0
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3:
                    if word in summary or word in connection:
                        relevance += 0.2
            
            relevance = min(1.0, relevance + doc.get("relevance_score", 0.5) * 0.5)
            scored.append((doc, relevance))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        insights = []
        for doc, score in scored[:limit]:
            if score > 0.3:
                insights.append({
                    "id": str(doc["_id"]),
                    "insight_text": doc.get("insight_text", ""),
                    "connection": doc.get("connection", ""),
                    "relevance_score": score
                })
        
        return insights

    async def mark_insight_used(self, insight_id: str) -> bool:
        result = await self._mongo.insights.update_one(
            {"_id": insight_id},
            {"$set": {"is_used": True, "used_at": datetime.now()}}
        )
        return result.modified_count > 0

    async def run_daydream_cycle(self) -> Optional[Dict]:
        await NeuralEventBus.set_activity("hippocampus", "Daydreaming...")
        
        try:
            chat_history = await self.get_recent_chat_history(limit=25)
            if len(chat_history) < 5:
                await NeuralEventBus.clear_activity("hippocampus")
                return None
            
            user_messages = [c["content"] for c in chat_history if c["role"] == "user"]
            if not user_messages:
                await NeuralEventBus.clear_activity("hippocampus")
                return None
            
            chat_summary = " | ".join(user_messages[-10:])[:1000]
            
            insight = await self.generate_insight(chat_summary)
            
            if insight:
                insight_id = await self.store_insight(insight)
                insight["id"] = insight_id
                
                await NeuralEventBus.emit(
                    "hippocampus", 
                    "amygdala", 
                    "insight_generated",
                    payload={"insight_id": insight_id, "preview": insight.get("insight_text", "")[:100]}
                )
            
            await NeuralEventBus.clear_activity("hippocampus")
            return insight
        except Exception:
            await NeuralEventBus.clear_activity("hippocampus")
            return None