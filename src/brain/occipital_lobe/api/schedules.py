from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from src.brain.brainstem import get_brain
from src.brain.occipital_lobe.types import ScheduleCreate, ScheduleUpdate
from src.brain.db.mongo_client import get_mongo_client

router = APIRouter(prefix="/api/schedules", tags=["schedules"])

def get_mongo():
    return get_mongo_client()

@router.get("")
async def get_schedules(
    limit: int = Query(50, ge=1, le=500),
    upcoming: bool = Query(True),
    status: Optional[str] = Query(None)
):
    if upcoming:
        brain = await get_brain()
        if not brain or not brain.hippocampus:
            return []
        return await brain.hippocampus.get_pending_schedules(limit=limit)
    
    mongo = get_mongo()
    filter_query = {}
    if status:
        filter_query["status"] = status
        
    cursor = mongo.schedules.find(filter_query).sort("scheduled_at", 1).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d["_id"]),
            "context": d.get("context"),
            "scheduled_at": d.get("scheduled_at"),
            "priority": d.get("priority", 0),
            "status": d.get("status", "pending"),
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None
        }
        for d in docs
    ]

@router.get("/{schedule_id}")
async def get_schedule_by_id(schedule_id: str):
    mongo = get_mongo()
    doc = await mongo.schedules.find_one({"_id": schedule_id})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    return {
        "id": str(doc["_id"]),
        "context": doc.get("context"),
        "scheduled_at": doc.get("scheduled_at"),
        "priority": doc.get("priority", 0),
        "status": doc.get("status", "pending"),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None
    }

@router.post("")
async def create_schedule(schedule: ScheduleCreate):
    try:
        brain = await get_brain()
        if not brain or not brain.hippocampus:
            raise HTTPException(status_code=503, detail="Brain not initialized")
        schedule_id = await brain.hippocampus.add_schedule(
            trigger_time=datetime.fromisoformat(schedule.scheduled_at),
            context=schedule.context,
            priority=schedule.priority
        )
        return {"id": schedule_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{schedule_id}")
async def update_schedule(schedule_id: str, schedule: ScheduleUpdate):
    updates = schedule.dict(exclude_unset=True)
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    success = await brain.hippocampus.update_schedule(schedule_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found or update failed")
    return {"status": "updated", "id": schedule_id}

@router.delete("/{schedule_id}")
async def delete_schedule(schedule_id: str):
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    success = await brain.hippocampus.delete_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"status": "deleted", "id": schedule_id}
