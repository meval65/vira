import uuid
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict
from .models import Triple


class KnowledgeGraph:
    MIN_CONFIDENCE = 0.3

    def __init__(self, mongo_client):
        self._mongo = mongo_client

    async def store_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float,
        source_memory_id: Optional[str]
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
        subject: Optional[str],
        predicate: Optional[str],
        obj: Optional[str],
        min_confidence: float
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

    async def list_triples(self, limit: int, skip: int) -> List[Dict]:
        cursor = self._mongo.knowledge_graph.find({}).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [
            {
                "id": str(d["_id"]),
                "subject": d.get("subject"),
                "predicate": d.get("predicate"),
                "object": d.get("object"),
                "confidence": d.get("confidence", 0.8),
                "source_memory_id": d.get("source_memory_id"),
                "access_count": d.get("access_count", 0),
                "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
                "last_accessed": d.get("last_accessed").isoformat() if d.get("last_accessed") else None
            }
            for d in docs
        ]

    async def update_triple(self, triple_id: str, updates: Dict) -> bool:
        if not updates:
            return False
        
        if "_id" in updates:
            del updates["_id"]
            
        result = await self._mongo.knowledge_graph.update_one(
            {"_id": triple_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    async def delete_triple(self, triple_id: str) -> bool:
        result = await self._mongo.knowledge_graph.delete_one({"_id": triple_id})
        return result.deleted_count > 0

    async def get_entity_relations(self, entity: str, limit: int) -> List[Triple]:
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
        relations = await self.get_entity_relations(entity, limit=50)
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
        max_depth: int
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


