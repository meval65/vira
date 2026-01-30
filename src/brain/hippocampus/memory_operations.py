import uuid
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from .models import Memory
from src.brain.brainstem import MAX_RETRIEVED_MEMORIES, NeuralEventBus


class MemoryOperations:
    def __init__(self, mongo_client, openrouter=None):
        self._mongo = mongo_client
        self._openrouter = openrouter

    async def store(
        self,
        summary: str,
        memory_type: str,
        priority: float,
        embedding: Optional[List[float]],
        tags: Optional[List[str]],
        canonicalizer
    ) -> str:
        if not summary or not summary.strip():
            raise ValueError("Summary cannot be empty")
        
        priority = max(0.0, min(1.0, priority))
        
        await NeuralEventBus.set_activity("hippocampus", f"Storing {memory_type}")
        await NeuralEventBus.emit("hippocampus", "hippocampus", f"store_memory:{memory_type}", payload={
            "summary_len": len(summary),
            "priority": priority
        })
        
        canonical = await canonicalizer.canonicalize(summary, memory_type)
        fingerprint = canonical.get("fingerprint")
        
        if fingerprint:
            existing = await self._find_by_fingerprint(fingerprint)
            if existing:
                merged = await self._merge_memories(existing, canonical)
                await self._update_memory(existing["_id"], merged)
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
        await NeuralEventBus.clear_activity("hippocampus")
        return memory_id

    async def recall(
        self,
        query: str,
        limit: int,
        query_embedding: Optional[List[float]],
        memory_type: Optional[str]
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

    async def delete_memory(self, memory_id: str, hard_delete: bool = False) -> bool:
        if hard_delete:
            result = await self._mongo.memories.delete_one({"_id": memory_id})
            return result.deleted_count > 0
        result = await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": {"status": "deleted", "deleted_at": datetime.now()}}
        )
        return result.modified_count > 0
    
    async def list_memories(
        self,
        limit: int,
        skip: int,
        status: Optional[str],
        memory_type: Optional[str]
    ) -> List[Dict]:
        filter_query = {}
        if status:
            filter_query["status"] = status
        else:
            filter_query["status"] = "active"
        if memory_type:
            filter_query["type"] = memory_type
        
        cursor = self._mongo.memories.find(filter_query).sort([
            ("priority", -1),
            ("last_used", -1)
        ]).skip(skip).limit(limit)
        
        docs = await cursor.to_list(length=limit)
        
        return [
            {
                "id": str(d["_id"]),
                "summary": d.get("summary", ""),
                "memory_type": d.get("type", "general"),
                "priority": d.get("priority", 0.5),
                "confidence": d.get("confidence", 0.5),
                "is_compressed": d.get("is_compressed", False),
                "use_count": d.get("use_count", 0),
                "status": d.get("status", "active"),
                "fingerprint": d.get("fingerprint"),
                "tags": d.get("tags", []),
                "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
                "last_used_at": d.get("last_used").isoformat() if d.get("last_used") else None
            }
            for d in docs
        ]
    
    async def update_memory(self, memory_id: str, updates: Dict) -> bool:
        if not updates:
            return False
        
        updates["last_used"] = datetime.now()
        result = await self._mongo.memories.update_one(
            {"_id": memory_id},
            {"$set": updates}
        )
        return result.modified_count > 0

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

    async def _find_by_fingerprint(self, fingerprint: str) -> Optional[Dict]:
        return await self._mongo.memories.find_one({
            "fingerprint": fingerprint,
            "status": "active"
        })

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

    async def _merge_memories(self, existing: Dict, new: Dict) -> Dict:
        new_conf = min(1.0, existing.get("confidence", 0.5) + 0.1)
        new_priority = min(1.0, existing.get("priority", 0.5) + 0.05)
        
        return {
            "summary": new.get("summary", existing.get("summary")),
            "priority": new_priority,
            "confidence": new_conf,
            "value": new.get("value", existing.get("value"))
        }
