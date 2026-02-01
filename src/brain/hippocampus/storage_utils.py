import uuid
from datetime import datetime
from typing import Dict, Optional, List, Any


class StorageUtils:
    def __init__(self, mongo_client):
        self._mongo = mongo_client

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

    async def query_entity(self, entity_name: str, get_entity_relations_func) -> Dict[str, Any]:
        entity_lower = entity_name.lower().strip()
        
        entity_doc = await self._mongo.entities.find_one({"name": entity_lower})
        
        triples = await get_entity_relations_func(entity_lower, limit=10)
        
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

    async def store_tool_output(
        self,
        tool_name: str,
        output_type: str,
        content: str,
        file_path: Optional[str],
        metadata: Optional[Dict]
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
        tool_name: Optional[str],
        output_type: Optional[str],
        limit: int
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

    async def get_recent_chat_history(self, limit: int) -> List[Dict]:
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

    async def get_related_memories(
        self,
        memory_id: str,
        get_memory_by_id_func,
        cosine_similarity_func
    ) -> List:
        import numpy as np
        from models import Memory
        
        source_memory = await get_memory_by_id_func(memory_id)
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
                    similarity = cosine_similarity_func(source_memory.embedding, emb)
                    if similarity > 0.7:
                        scored_memories.append((doc, similarity))
                except Exception:
                    continue
        
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        memories = []
        for doc, _ in scored_memories[:10]:
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


