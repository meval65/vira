from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.brain.infrastructure.mongo_client import get_mongo_client
from src.brain.brainstem import get_brain

router = APIRouter(prefix="/api/maintenance", tags=["maintenance"])


def get_mongo():
    return get_mongo_client()


class CompressRequest(BaseModel):
    force: bool = False


@router.post("/trigger")
async def trigger_maintenance():
    brain = await get_brain()
    
    results = {}
    
    if brain and brain.hippocampus:
        try:
            await brain.hippocampus.run_maintenance()
            results["hippocampus"] = "maintenance_triggered"
        except Exception as e:
            results["hippocampus"] = f"error: {str(e)}"
    
    return {"status": "triggered", "results": results}


@router.post("/compress-memories")
async def compress_memories(data: CompressRequest):
    brain = await get_brain()
    
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    mongo = get_mongo()
    
    query = {"status": "active"}
    if not data.force:
        query["$or"] = [
            {"is_compressed": False},
            {"is_compressed": {"$exists": False}}
        ]
    
    cursor = mongo.memories.find(query).limit(100)
    memories = await cursor.to_list(length=100)
    
    compressed_count = 0
    
    for mem in memories:
        try:
            summary = mem.get("summary", "")
            if len(summary) > 200:
                compressed_summary = summary[:200] + "..."
                await mongo.memories.update_one(
                    {"_id": mem["_id"]},
                    {"$set": {
                        "summary": compressed_summary,
                        "original_summary": summary,
                        "is_compressed": True,
                        "compressed_at": datetime.now()
                    }}
                )
                compressed_count += 1
        except Exception:
            pass
    
    return {"status": "completed", "compressed": compressed_count}


@router.post("/optimize-graph")
async def optimize_graph():
    brain = await get_brain()
    
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    mongo = get_mongo()
    
    try:
        cursor = mongo.knowledge_graph.aggregate([
            {"$group": {
                "_id": {"subject": "$subject", "predicate": "$predicate", "object": "$object"},
                "count": {"$sum": 1},
                "ids": {"$push": "$_id"}
            }},
            {"$match": {"count": {"$gt": 1}}}
        ])
        
        duplicates = await cursor.to_list(length=100)
        removed = 0
        
        for dup in duplicates:
            ids_to_remove = dup["ids"][1:]
            for oid in ids_to_remove:
                await mongo.knowledge_graph.delete_one({"_id": oid})
                removed += 1
        
        return {"status": "optimized", "duplicates_removed": removed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


