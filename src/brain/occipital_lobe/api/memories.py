from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query, Body, Depends
from src.brain.brainstem import get_brain
from src.brain.occipital_lobe.types import MemoryCreate, MemoryUpdate
from src.brain.occipital_lobe.state import manager
from src.brain.db.mongo_client import get_mongo_client

router = APIRouter(prefix="/api/memories", tags=["memories"])

def get_mongo():
    return get_mongo_client()

@router.get("")
async def get_memories(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    memory_type: Optional[str] = Query(None)
):
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        return []
    return await brain.hippocampus.list_memories(
        limit=limit,
        skip=skip,
        status=status,
        memory_type=memory_type
    )

@router.get("/{memory_id}")
async def get_memory_by_id(memory_id: str):
    mongo = get_mongo()
    doc = await mongo.memories.find_one({"_id": memory_id})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    return {
        "id": str(doc["_id"]),
        "summary": doc.get("summary", ""),
        "memory_type": doc.get("type", "general"),
        "priority": doc.get("priority", 0.5),
        "confidence": doc.get("confidence", 0.5),
        "is_compressed": doc.get("is_compressed", False),
        "use_count": doc.get("use_count", 0),
        "status": doc.get("status", "active"),
        "fingerprint": doc.get("fingerprint"),
        "entity": doc.get("entity"),
        "relation": doc.get("relation"),
        "value": doc.get("value"),
        "embedding": doc.get("embedding"),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "last_used_at": doc.get("last_used").isoformat() if doc.get("last_used") else None
    }

@router.post("")
async def create_memory(memory: MemoryCreate):
    try:
        brain = await get_brain()
        if not brain or not brain.hippocampus:
            raise HTTPException(status_code=503, detail="Brain not initialized")
        memory_id = await brain.hippocampus.store(
            summary=memory.summary,
            memory_type=memory.memory_type,
            priority=memory.priority,
            embedding=memory.embedding
        )
        return {"id": memory_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{memory_id}")
async def update_memory(memory_id: str, memory: MemoryUpdate):
    updates = memory.dict(exclude_unset=True)
    if "memory_type" in updates:
        updates["type"] = updates.pop("memory_type")
        
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    success = await brain.hippocampus.update_memory(memory_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or update failed")
    return {"status": "updated", "id": memory_id}

@router.delete("/{memory_id}")
async def delete_memory(memory_id: str, hard_delete: bool = Query(False)):
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    success = await brain.hippocampus.delete_memory(memory_id, hard_delete=hard_delete)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "id": memory_id}

@router.post("/bulk-delete")
async def bulk_delete_memories(memory_ids: List[str] = Body(...), hard_delete: bool = Body(False)):
    mongo = get_mongo()
    
    if hard_delete:
        result = await mongo.memories.delete_many({"_id": {"$in": memory_ids}})
        deleted_count = result.deleted_count
        action = "hard_deleted"
    else:
        result = await mongo.memories.update_many(
            {"_id": {"$in": memory_ids}},
            {"$set": {"status": "archived"}}
        )
        deleted_count = result.modified_count
        action = "archived"
    
    await manager.broadcast("memory_update", {"action": f"bulk_{action}", "count": deleted_count})
    return {"status": f"bulk_{action}", "count": deleted_count}
