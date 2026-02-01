import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from src.brain.brainstem import NeuralEventBus


class ScheduleManager:
    def __init__(self, mongo_client):
        self._mongo = mongo_client

    async def add_schedule(
        self,
        trigger_time: datetime,
        context: str,
        priority: int,
        repeat: Optional[str]
    ) -> str:
        window_start = trigger_time - timedelta(minutes=30)
        window_end = trigger_time + timedelta(minutes=30)
        
        conflict = await self._mongo.schedules.find_one({
            "status": "pending",
            "scheduled_at": {"$gte": window_start, "$lte": window_end}
        })
        
        if conflict and conflict.get("priority", 0) >= priority:
            pass
            
        schedule_id = str(uuid.uuid4())
        doc = {
            "_id": schedule_id,
            "scheduled_at": trigger_time,
            "context": context,
            "priority": priority,
            "repeat": repeat,
            "status": "pending",
            "created_at": datetime.now()
        }
        
        await self._mongo.schedules.insert_one(doc)
        
        await NeuralEventBus.emit("hippocampus", "thalamus", "schedule_created", {
            "schedule_id": schedule_id,
            "trigger_time": trigger_time.isoformat()
        })
        
        return schedule_id

    async def update_schedule(self, schedule_id: str, updates: Dict) -> bool:
        if not updates:
            return False
            
        if "scheduled_at" in updates and isinstance(updates["scheduled_at"], str):
            try:
                updates["scheduled_at"] = datetime.fromisoformat(updates["scheduled_at"])
            except ValueError:
                pass
                
        result = await self._mongo.schedules.update_one(
            {"_id": schedule_id},
            {"$set": updates}
        )
        
        if result.modified_count > 0:
            await NeuralEventBus.emit("hippocampus", "thalamus", "schedule_updated", {
                "schedule_id": schedule_id
            })
            
        return result.modified_count > 0

    async def delete_schedule(self, schedule_id: str) -> bool:
        result = await self._mongo.schedules.delete_one({"_id": schedule_id})
        
        if result.deleted_count > 0:
            await NeuralEventBus.emit("hippocampus", "thalamus", "schedule_deleted", {
                "schedule_id": schedule_id
            })
            
        return result.deleted_count > 0

    async def get_pending_schedules(self, limit: int) -> List[Dict]:
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

    async def get_upcoming_schedules(self, hours_ahead: int) -> List[Dict]:
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


