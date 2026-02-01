from fastapi import APIRouter
from src.brain.infrastructure.mongo_client import get_mongo_client
from src.brain.brainstem import get_brain

router = APIRouter(prefix="/api/search", tags=["search"])


def get_mongo():
    return get_mongo_client()


@router.get("/memories")
async def search_memories(query: str, limit: int = 10):
    brain = await get_brain()
    
    if not brain or not brain.hippocampus:
        return []
    
    memories = await brain.hippocampus.recall(query, limit=limit)
    
    result = []
    for m in memories:
        result.append({
            "id": m.id,
            "summary": m.summary,
            "memory_type": m.memory_type,
            "confidence": m.confidence
        })
    
    return result


@router.get("/entities")
async def search_entities(query: str, limit: int = 10):
    mongo = get_mongo()
    
    cursor = mongo.entities.find({
        "$or": [
            {"name": {"$regex": query, "$options": "i"}},
            {"aliases": {"$regex": query, "$options": "i"}}
        ]
    }).limit(limit)
    
    entities = await cursor.to_list(length=limit)
    
    result = []
    for e in entities:
        result.append({
            "name": e.get("name", ""),
            "entity_type": e.get("entity_type", "unknown"),
            "aliases": e.get("aliases", []),
            "mention_count": e.get("mention_count", 0)
        })
    
    return result


