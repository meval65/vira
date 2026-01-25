"""
Occipital Lobe Module - Dashboard and API for Vira Personal Life OS.

This module provides:
- Web dashboard for system visualization
- REST API for CRUD operations on memories, schedules, entities
- WebSocket for real-time neural activity monitoring
- Chat logs API for conversation history

Refactored to use MongoDB instead of SQLite.
"""

import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, Set, Dict, List, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.brainstem import ADMIN_ID, API_ROTATOR, PERSONA_INSTRUCTION, NeuralEventBus
from src.db.mongo_client import get_mongo_client, MongoDBClient

logger = logging.getLogger(__name__)


# ==================== PYDANTIC MODELS ====================

class MemoryCreate(BaseModel):
    summary: str
    memory_type: str = "general"
    priority: float = 0.5


class MemoryUpdate(BaseModel):
    summary: str


class ScheduleCreate(BaseModel):
    context: str
    scheduled_at: str
    priority: int = 0


class AdminProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    additional_info: Optional[str] = None


class CustomInstructionUpdate(BaseModel):
    instruction: str = Field(..., min_length=10, max_length=5000)
    name: Optional[str] = "Custom Override"


class PersonaCreate(BaseModel):
    name: str
    instruction: str
    temperature: float = 0.7


class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    instruction: Optional[str] = None
    temperature: Optional[float] = None


class ChatLogEntry(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None


# ==================== GLOBAL STATE ====================

_custom_instruction_override: Optional[str] = None
_custom_instruction_name: str = "Default Persona"


def get_active_instruction() -> tuple[str, str]:
    """Get current active instruction and its name."""
    global _custom_instruction_override, _custom_instruction_name
    if _custom_instruction_override:
        return (_custom_instruction_override, _custom_instruction_name)
    return (PERSONA_INSTRUCTION, "Default Persona")


# ==================== WEBSOCKET MANAGER ====================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, event_type: str, data: dict):
        message = {"type": event_type, "data": data, "timestamp": datetime.now().isoformat()}
        async with self._lock:
            for connection in list(self.active_connections):
                try:
                    await connection.send_json(message)
                except Exception:
                    self.active_connections.discard(connection)


manager = ConnectionManager()


# ==================== MONGODB HELPER ====================

def get_mongo() -> MongoDBClient:
    """Get MongoDB client instance."""
    return get_mongo_client()


# ==================== FASTAPI APP ====================

app = FastAPI(
    title="Vira Dashboard API",
    description="Complete System Dashboard for Vira Personal Life OS (MongoDB Edition)",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")


# ==================== CORE ENDPOINTS ====================

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    html_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Dashboard not found. Please create src/dashboard/index.html</h1>"


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)


@app.get("/api/health")
async def get_health():
    try:
        mongo = get_mongo()
        await mongo.db.command('ping')
        return {"status": "healthy", "database": "mongodb", "timestamp": datetime.now().isoformat(), "admin_id": ADMIN_ID}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/api/stats")
async def get_stats():
    mongo = get_mongo()
    try:
        mem_count = await mongo.memories.count_documents({"status": "active"})
        triple_count = await mongo.knowledge_graph.count_documents({})
        sched_count = await mongo.schedules.count_documents({"status": "pending"})
        entity_count = await mongo.entities.count_documents({})
        chat_count = await mongo.chat_logs.count_documents({})

        return {
            "memories": mem_count,
            "triples": triple_count,
            "pending_schedules": sched_count,
            "entities": entity_count,
            "chat_logs": chat_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== MEMORY ENDPOINTS ====================

@app.get("/api/memories")
async def get_memories(limit: int = 50, skip: int = 0):
    mongo = get_mongo()
    cursor = mongo.memories.find(
        {"status": "active"}
    ).sort("last_used", -1).skip(skip).limit(limit)
    
    docs = await cursor.to_list(length=limit)
    return [
        {
            "id": str(d["_id"]),
            "summary": d.get("summary", ""),
            "memory_type": d.get("type", "general"),
            "priority": d.get("priority", 0.5),
            "confidence": d.get("confidence", 0.5),
            "created_at": d.get("created_at").isoformat() if d.get("created_at") else None,
            "last_used_at": d.get("last_used").isoformat() if d.get("last_used") else None
        }
        for d in docs
    ]


@app.post("/api/memories")
async def create_memory(memory: MemoryCreate):
    mongo = get_mongo()
    memory_id = str(uuid.uuid4())
    
    await mongo.memories.insert_one({
        "_id": memory_id,
        "summary": memory.summary,
        "type": memory.memory_type,
        "priority": memory.priority,
        "confidence": 0.5,
        "status": "active",
        "created_at": datetime.now(),
        "last_used": datetime.now(),
        "use_count": 0
    })
    
    await manager.broadcast("memory_update", {"action": "created", "id": memory_id})
    return {"id": memory_id, "status": "created"}


@app.put("/api/memories/{memory_id}")
async def update_memory(memory_id: str, data: MemoryUpdate):
    mongo = get_mongo()
    
    # Generate new embedding if brain is available
    emb = None
    try:
        from src.brainstem import get_brain
        brain = get_brain()
        if brain and brain.prefrontal_cortex:
            emb = await brain.prefrontal_cortex.generate_embedding(data.summary)
    except Exception:
        pass
    
    update_doc = {"summary": data.summary, "last_used": datetime.now()}
    if emb:
        update_doc["embedding"] = emb
    
    result = await mongo.memories.update_one(
        {"_id": memory_id},
        {"$set": update_doc}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    await manager.broadcast("memory_update", {"action": "updated", "id": memory_id})
    return {"status": "updated"}


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    mongo = get_mongo()
    result = await mongo.memories.update_one(
        {"_id": memory_id},
        {"$set": {"status": "deleted"}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Memory not found")
    
    await manager.broadcast("memory_update", {"action": "deleted", "id": memory_id})
    return {"status": "deleted"}


# ==================== SCHEDULE ENDPOINTS ====================

@app.get("/api/schedules")
async def get_schedules(limit: int = 50, status: Optional[str] = None):
    mongo = get_mongo()
    query = {}
    if status:
        query["status"] = status
    
    cursor = mongo.schedules.find(query).sort("scheduled_at", 1).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d["_id"]),
            "scheduled_at": d["scheduled_at"].isoformat() if isinstance(d["scheduled_at"], datetime) else d["scheduled_at"],
            "context": d.get("context", ""),
            "priority": d.get("priority", 0),
            "status": d.get("status", "pending"),
            "recurrence": d.get("recurrence")
        }
        for d in docs
    ]


@app.post("/api/schedules")
async def create_schedule(schedule: ScheduleCreate):
    from dateutil import parser
    mongo = get_mongo()
    
    trigger_time = parser.parse(schedule.scheduled_at)
    schedule_id = str(uuid.uuid4())
    
    await mongo.schedules.insert_one({
        "_id": schedule_id,
        "scheduled_at": trigger_time,
        "context": schedule.context,
        "priority": schedule.priority,
        "status": "pending",
        "created_at": datetime.now()
    })
    
    await manager.broadcast("schedule_update", {"action": "created", "id": schedule_id})
    return {"id": schedule_id, "status": "created"}


@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    mongo = get_mongo()
    result = await mongo.schedules.update_one(
        {"_id": schedule_id},
        {"$set": {"status": "cancelled"}}
    )
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Schedule not found")
    
    await manager.broadcast("schedule_update", {"action": "cancelled", "id": schedule_id})
    return {"status": "cancelled"}


# ==================== CHAT LOGS ENDPOINTS (NEW) ====================

@app.get("/api/chat-logs")
async def get_chat_logs(limit: int = 50, skip: int = 0):
    """Get chat history with pagination."""
    mongo = get_mongo()
    cursor = mongo.chat_logs.find().sort("timestamp", -1).skip(skip).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d["_id"]),
            "role": d.get("role", "user"),
            "content": d.get("content", ""),
            "timestamp": d["timestamp"].isoformat() if d.get("timestamp") else None,
            "has_embedding": d.get("embedding") is not None
        }
        for d in reversed(docs)  # Chronological order
    ]


@app.get("/api/chat-logs/search")
async def search_chat_logs(q: str = Query(..., min_length=2), limit: int = 20):
    """Search chat logs by content (text search)."""
    mongo = get_mongo()
    
    # Simple regex search (case-insensitive)
    cursor = mongo.chat_logs.find({
        "content": {"$regex": q, "$options": "i"}
    }).sort("timestamp", -1).limit(limit)
    
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d["_id"]),
            "role": d.get("role", "user"),
            "content": d.get("content", ""),
            "timestamp": d["timestamp"].isoformat() if d.get("timestamp") else None
        }
        for d in docs
    ]


@app.delete("/api/chat-logs/{log_id}")
async def delete_chat_log(log_id: str):
    """Delete a specific chat log entry."""
    mongo = get_mongo()
    from bson import ObjectId
    
    try:
        result = await mongo.chat_logs.delete_one({"_id": ObjectId(log_id)})
    except Exception:
        result = await mongo.chat_logs.delete_one({"_id": log_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat log not found")
    
    await manager.broadcast("chat_update", {"action": "deleted", "id": log_id})
    return {"status": "deleted"}


@app.delete("/api/chat-logs")
async def clear_chat_logs():
    """Clear all chat logs (use with caution)."""
    mongo = get_mongo()
    result = await mongo.chat_logs.delete_many({})
    await manager.broadcast("chat_update", {"action": "cleared", "count": result.deleted_count})
    return {"status": "cleared", "deleted_count": result.deleted_count}


# ==================== KNOWLEDGE GRAPH ENDPOINTS ====================

@app.get("/api/triples")
async def get_triples(limit: int = 50):
    mongo = get_mongo()
    cursor = mongo.knowledge_graph.find().sort("access_count", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d["_id"]),
            "subject": d["subject"],
            "predicate": d["predicate"],
            "object": d["object"],
            "confidence": d.get("confidence", 0.8),
            "access_count": d.get("access_count", 0)
        }
        for d in docs
    ]


@app.get("/api/entities")
async def get_entities(limit: int = 50):
    mongo = get_mongo()
    cursor = mongo.entities.find().sort("mention_count", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    return [
        {
            "id": str(d["_id"]),
            "name": d["name"],
            "entity_type": d.get("entity_type", "unknown"),
            "mention_count": d.get("mention_count", 0)
        }
        for d in docs
    ]


# ==================== PROFILE ENDPOINTS ====================

@app.get("/api/profile")
async def get_profile():
    mongo = get_mongo()
    doc = await mongo.admin_profile.find_one({"_id": "admin"})
    if doc:
        return {
            "telegram_name": doc.get("telegram_name"),
            "full_name": doc.get("full_name"),
            "additional_info": doc.get("additional_info")
        }
    return {"telegram_name": None, "full_name": None, "additional_info": None}


@app.put("/api/profile")
async def update_profile(profile: AdminProfileUpdate):
    mongo = get_mongo()
    update_doc = {"last_updated": datetime.now()}
    if profile.full_name:
        update_doc["full_name"] = profile.full_name
    if profile.additional_info:
        update_doc["additional_info"] = profile.additional_info
    
    await mongo.admin_profile.update_one(
        {"_id": "admin"},
        {"$set": update_doc},
        upsert=True
    )
    return {"status": "updated"}


# ==================== EMOTIONAL STATE ENDPOINTS ====================

@app.get("/api/emotional-state")
async def get_emotional_state():
    mongo = get_mongo()
    doc = await mongo.emotional_state.find_one({"_id": "state"})
    if doc:
        return {
            "mood": doc.get("current_mood", "neutral"),
            "empathy": doc.get("empathy_level", 0.5),
            "satisfaction": doc.get("satisfaction_level", 0.0)
        }
    return {"mood": "neutral", "empathy": 0.5, "satisfaction": 0.0}


# ==================== NEURAL STATUS ENDPOINTS ====================

@app.on_event("startup")
async def startup_event():
    NeuralEventBus.subscribe(broadcast_neural_event)


@app.on_event("shutdown")
async def shutdown_event():
    NeuralEventBus.unsubscribe(broadcast_neural_event)


async def broadcast_neural_event(event: dict):
    """Broadcast neural event to all connected clients."""
    await manager.broadcast("neural_activity", event)


@app.get("/api/neural-status")
async def get_neural_status():
    """Get current activity state of all brain modules."""
    return NeuralEventBus.get_module_states()


# ==================== SYSTEM STATUS ENDPOINTS ====================

@app.get("/api/system-status")
async def get_system_status():
    """Get comprehensive system status including API health and active model."""
    rotator_status = API_ROTATOR.get_status()
    instruction, instruction_name = get_active_instruction()

    mongo = get_mongo()
    try:
        mem_count = await mongo.memories.count_documents({"status": "active"})
        sched_count = await mongo.schedules.count_documents({"status": "pending"})

        return {
            "status": "online",
            "timestamp": datetime.now().isoformat(),
            "database": "mongodb",
            "api": {
                "current_key_index": rotator_status["current_key_idx"],
                "current_model": rotator_status["current_model"],
                "failed_combinations": rotator_status["failed_count"],
                "total_combinations": rotator_status["total_combinations"],
                "health": "healthy" if rotator_status["failed_count"] < rotator_status["total_combinations"] else "degraded"
            },
            "instruction": {
                "name": instruction_name,
                "is_custom": instruction_name != "Default Persona",
                "length": len(instruction)
            },
            "database_stats": {
                "active_memories": mem_count,
                "pending_schedules": sched_count
            },
            "admin_id": ADMIN_ID
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ==================== PERSONA ENDPOINTS ====================

@app.get("/api/personas")
async def get_personas():
    """Get all saved personas."""
    from src.brainstem import get_brain
    brain = get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return await brain.hippocampus.get_personas()


@app.post("/api/personas")
async def create_persona(data: PersonaCreate):
    """Create a new persona profile."""
    from src.brainstem import get_brain
    brain = get_brain()
    try:
        pid = await brain.hippocampus.create_persona(data.name, data.instruction, data.temperature)
        await manager.broadcast("personas_update", {"action": "created", "id": pid})
        return {"id": pid, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/personas/{persona_id}/activate")
async def activate_persona(persona_id: str):
    """Switch the active persona."""
    from src.brainstem import get_brain
    brain = get_brain()
    
    success = await brain.hippocampus.set_active_persona(persona_id)
    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")
        
    global _custom_instruction_override, _custom_instruction_name
    _custom_instruction_override = None
    _custom_instruction_name = "Default Persona"
    
    persona = await brain.hippocampus.get_active_persona()
    
    await manager.broadcast("instruction_update", {
        "name": persona["name"],
        "is_custom": True,
        "temperature": persona["temperature"]
    })
    
    return {"status": "activated", "persona": persona}


@app.put("/api/personas/{persona_id}")
async def update_persona(persona_id: str, data: PersonaUpdate):
    """Update persona settings."""
    from src.brainstem import get_brain
    brain = get_brain()
    
    updates = {k: v for k, v in data.dict().items() if v is not None}
    success = await brain.hippocampus.update_persona(persona_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")
        
    await manager.broadcast("personas_update", {"action": "updated", "id": persona_id})
    return {"status": "updated"}


@app.delete("/api/personas/{persona_id}")
async def delete_persona(persona_id: str):
    """Delete a persona profile."""
    from src.brainstem import get_brain
    brain = get_brain()
    
    try:
        success = await brain.hippocampus.delete_persona(persona_id)
        if not success:
            raise HTTPException(status_code=404, detail="Persona not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    await manager.broadcast("personas_update", {"action": "deleted", "id": persona_id})
    return {"status": "deleted"}


# ==================== CUSTOM INSTRUCTION ENDPOINTS ====================

@app.get("/api/custom-instruction")
async def get_custom_instruction():
    """Get current custom instruction."""
    instruction, name = get_active_instruction()
    return {
        "name": name,
        "instruction": instruction,
        "is_custom": name != "Default Persona",
        "default_available": True
    }


@app.put("/api/custom-instruction")
async def update_custom_instruction(data: CustomInstructionUpdate):
    """Override the persona instruction."""
    global _custom_instruction_override, _custom_instruction_name
    _custom_instruction_override = data.instruction
    _custom_instruction_name = data.name or "Custom Override"

    await manager.broadcast("instruction_update", {
        "name": _custom_instruction_name,
        "is_custom": True
    })

    return {
        "status": "updated",
        "name": _custom_instruction_name,
        "length": len(data.instruction)
    }


@app.post("/api/reset-instruction")
async def reset_custom_instruction():
    """Reset to default persona instruction."""
    global _custom_instruction_override, _custom_instruction_name
    _custom_instruction_override = None
    _custom_instruction_name = "Default Persona"

    await manager.broadcast("instruction_update", {
        "name": "Default Persona",
        "is_custom": False
    })

    return {
        "status": "reset",
        "name": "Default Persona"
    }


# ==================== RUN ====================

def run_dashboard(host: str = "0.0.0.0", port: int = 5000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
