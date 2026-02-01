from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from src.brain.occipital_lobe.types import EntityCreate, EntityUpdate
from src.brain.occipital_lobe.state import manager
from src.brain.infrastructure.mongo_client import get_mongo_client

router = APIRouter(prefix="/api/entities", tags=["entities"])

def get_mongo():
    return get_mongo_client()

@router.get("")
async def get_entities(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    entity_type: Optional[str] = Query(None)
):
    mongo = get_mongo()
    
    filter_query = {}
    if entity_type:
        filter_query["entity_type"] = entity_type
    
    cursor = mongo.entities.find(filter_query).sort("mention_count", -1).skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d.get("_id", d.get("name"))),
            "name": d.get("name"),
            "entity_type": d.get("entity_type", "unknown"),
            "mention_count": d.get("mention_count", 0),
            "metadata": d.get("metadata", {}),
            "first_seen": d.get("first_seen").isoformat() if d.get("first_seen") else None
        }
        for d in docs
    ]

@router.get("/{entity_name}")
async def get_entity_by_name(entity_name: str):
    mongo = get_mongo()
    doc = await mongo.entities.find_one({"name": entity_name.lower()})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return {
        "id": str(doc.get("_id", doc.get("name"))),
        "name": doc.get("name"),
        "entity_type": doc.get("entity_type", "unknown"),
        "mention_count": doc.get("mention_count", 0),
        "metadata": doc.get("metadata", {}),
        "first_seen": doc.get("first_seen").isoformat() if doc.get("first_seen") else None
    }

@router.post("")
async def create_entity(entity: EntityCreate):
    mongo = get_mongo()
    entity_name = entity.name.lower().strip()
    
    existing = await mongo.entities.find_one({"name": entity_name})
    if existing:
        raise HTTPException(status_code=400, detail="Entity already exists")
    
    await mongo.entities.insert_one({
        "name": entity_name,
        "entity_type": entity.entity_type,
        "mention_count": 0,
        "metadata": entity.metadata or {},
        "first_seen": datetime.now()
    })
    
    await manager.broadcast("entity_update", {"action": "created", "name": entity_name})
    return {"name": entity_name, "status": "created"}

@router.put("/{entity_name}")
async def update_entity(entity_name: str, entity: EntityUpdate):
    mongo = get_mongo()
    
    update_data = {}
    if entity.name:
        update_data["name"] = entity.name.lower().strip()
    if entity.entity_type:
        update_data["entity_type"] = entity.entity_type
    if entity.metadata is not None:
        update_data["metadata"] = entity.metadata
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    result = await mongo.entities.update_one(
        {"name": entity_name.lower()},
        {"$set": update_data}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    await manager.broadcast("entity_update", {"action": "updated", "name": entity_name})
    return {"status": "updated", "modified": result.modified_count}

@router.delete("/{entity_name}")
async def delete_entity(entity_name: str):
    mongo = get_mongo()
    result = await mongo.entities.delete_one({"name": entity_name.lower()})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    await manager.broadcast("entity_update", {"action": "deleted", "name": entity_name})
    return {"status": "deleted"}


