import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from src.brain.occipital_lobe.types import ChatLogCreate
from src.brain.occipital_lobe.state import manager
from src.brain.infrastructure.mongo_client import get_mongo_client

router = APIRouter(prefix="/api/chat-logs", tags=["chat_logs"])

def get_mongo():
    return get_mongo_client()

@router.get("")
async def get_chat_logs(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    session_id: Optional[str] = Query(None)
):
    mongo = get_mongo()
    
    filter_query = {}
    if session_id:
        filter_query["session_id"] = session_id
    
    cursor = mongo.chat_logs.find(filter_query).sort("timestamp", -1).skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d["_id"]),
            "session_id": d.get("session_id"),
            "role": d.get("role"),
            "content": d.get("content"),
            "metadata": d.get("metadata", {}),
            "timestamp": d.get("timestamp").isoformat() if d.get("timestamp") else None
        }
        for d in docs
    ]

@router.get("/sessions")
async def get_chat_sessions():
    mongo = get_mongo()
    
    sessions = await mongo.chat_logs.distinct("session_id")
    return {"sessions": sessions, "count": len(sessions)}

@router.get("/{log_id}")
async def get_chat_log_by_id(log_id: str):
    mongo = get_mongo()
    doc = await mongo.chat_logs.find_one({"_id": log_id})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Chat log not found")
    
    return {
        "id": str(doc["_id"]),
        "session_id": doc.get("session_id"),
        "role": doc.get("role"),
        "content": doc.get("content"),
        "metadata": doc.get("metadata", {}),
        "timestamp": doc.get("timestamp").isoformat() if doc.get("timestamp") else None
    }

@router.post("")
async def create_chat_log(log: ChatLogCreate):
    mongo = get_mongo()
    log_id = str(uuid.uuid4())
    
    await mongo.chat_logs.insert_one({
        "_id": log_id,
        "session_id": log.session_id,
        "role": log.role,
        "content": log.content,
        "metadata": log.metadata or {},
        "timestamp": datetime.now()
    })
    
    await manager.broadcast("chat_log_update", {"action": "created", "id": log_id})
    return {"id": log_id, "status": "created"}

@router.delete("/{log_id}")
async def delete_chat_log(log_id: str):
    mongo = get_mongo()
    result = await mongo.chat_logs.delete_one({"_id": log_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat log not found")
    
    await manager.broadcast("chat_log_update", {"action": "deleted", "id": log_id})
    return {"status": "deleted"}

@router.delete("/session/{session_id}")
async def delete_chat_session(session_id: str):
    mongo = get_mongo()
    result = await mongo.chat_logs.delete_many({"session_id": session_id})
    
    await manager.broadcast("chat_log_update", {"action": "session_deleted", "session_id": session_id, "count": result.deleted_count})
    return {"status": "deleted", "count": result.deleted_count}


