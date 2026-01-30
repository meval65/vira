import uuid
import numpy as np
from datetime import datetime
from typing import List, Optional
from .models import Memory
from src.brain.brainstem import NeuralEventBus


class VisualMemory:
    def __init__(self, mongo_client):
        self._mongo = mongo_client

    async def store_visual_memory(
        self,
        image_description: str,
        image_path: Optional[str],
        embedding: Optional[List[float]],
        additional_context: Optional[str],
        priority: float
    ) -> str:
        if not image_description or not image_description.strip():
            raise ValueError("Image description cannot be empty")
        
        await NeuralEventBus.set_activity("hippocampus", "Storing Visual Memory")
        await NeuralEventBus.emit("hippocampus", "hippocampus", "store_visual_memory", payload={
            "description_len": len(image_description),
            "has_path": image_path is not None
        })
        
        summary = image_description.strip()
        if additional_context:
            summary = f"{summary}\n\nContext: {additional_context}"
        
        tags = ["visual_memory"]
        if image_path:
            if "." in image_path:
                ext = image_path.rsplit(".", 1)[-1].lower()
                if ext in ["jpg", "jpeg", "png", "gif", "webp", "bmp"]:
                    tags.append(f"image_{ext}")
        
        memory_id = str(uuid.uuid4())
        doc = {
            "_id": memory_id,
            "summary": summary,
            "type": "visual",
            "priority": max(0.0, min(1.0, priority)),
            "embedding": embedding,
            "image_path": image_path,
            "confidence": 0.8,
            "created_at": datetime.now(),
            "last_used": datetime.now(),
            "use_count": 0,
            "status": "active",
            "is_compressed": False,
            "tags": tags
        }
        
        await self._mongo.memories.insert_one(doc)
        
        await NeuralEventBus.clear_activity("hippocampus")
        return memory_id

    async def search_by_tag(
        self,
        tag: str,
        limit: int,
        query_embedding: Optional[List[float]],
        cosine_similarity_func
    ) -> List[Memory]:
        await NeuralEventBus.set_activity("hippocampus", f"Searching by tag: {tag}")
        
        filter_query = {
            "status": "active",
            "tags": tag
        }
        
        cursor = self._mongo.memories.find(filter_query).sort([
            ("priority", -1),
            ("last_used", -1)
        ]).limit(limit * 2 if query_embedding else limit)
        
        docs = await cursor.to_list(length=limit * 2 if query_embedding else limit)
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
                        similarity = cosine_similarity_func(query_emb, mem.embedding)
                        scored_memories.append((mem, similarity))
                    except Exception:
                        scored_memories.append((mem, 0.0))
                else:
                    scored_memories.append((mem, 0.0))
            
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            memories = [mem for mem, _ in scored_memories[:limit]]
        else:
            memories = memories[:limit]
        
        await NeuralEventBus.clear_activity("hippocampus")
        return memories

    async def search_visual_memories(
        self,
        query: str,
        query_embedding: Optional[List[float]],
        limit: int,
        cosine_similarity_func
    ) -> List[Memory]:
        return await self.search_by_tag(
            tag="visual_memory",
            limit=limit,
            query_embedding=query_embedding,
            cosine_similarity_func=cosine_similarity_func
        )

    async def search_memories_by_tags(
        self,
        tags: List[str],
        limit: int
    ) -> List[Memory]:
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
