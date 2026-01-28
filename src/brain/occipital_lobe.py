import os
import uuid
import asyncio
import logging
from datetime import datetime
from typing import Optional, Set, Dict, List, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.brain.brainstem import ADMIN_ID, NeuralEventBus
from src.brain.constants import DEFAULT_PERSONA_INSTRUCTION
from src.brain.db.mongo_client import get_mongo_client, MongoDBClient

logger = logging.getLogger(__name__)

# ==================== PYDANTIC MODELS ====================

class MemoryCreate(BaseModel):
    summary: str
    memory_type: str = "general"
    priority: float = 0.5
    embedding: Optional[List[float]] = None

class MemoryUpdate(BaseModel):
    summary: Optional[str] = None
    memory_type: Optional[str] = None
    priority: Optional[float] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    is_compressed: Optional[bool] = None

class TripleCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_memory_id: Optional[str] = None

class TripleUpdate(BaseModel):
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: Optional[float] = None

class ScheduleCreate(BaseModel):
    context: str
    scheduled_at: str
    priority: int = 0
    status: str = "pending"

class ScheduleUpdate(BaseModel):
    context: Optional[str] = None
    scheduled_at: Optional[str] = None
    priority: Optional[int] = None
    status: Optional[str] = None

class EntityCreate(BaseModel):
    name: str
    entity_type: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None

class EntityUpdate(BaseModel):
    name: Optional[str] = None
    entity_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AdminProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    telegram_name: Optional[str] = None
    additional_info: Optional[str] = None

class CustomInstructionUpdate(BaseModel):
    instruction: str = Field(..., min_length=10, max_length=5000)
    name: Optional[str] = "Custom Override"

class PersonaCreate(BaseModel):
    name: str
    instruction: str
    temperature: float = 0.7
    description: Optional[str] = None

class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    instruction: Optional[str] = None
    temperature: Optional[float] = None
    description: Optional[str] = None

class ChatLogEntry(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatLogCreate(BaseModel):
    session_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class GlobalContextUpdate(BaseModel):
    context_text: str
    metadata: Optional[Dict[str, Any]] = None

class SystemConfigUpdate(BaseModel):
    chat_model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    proactive_check_interval: Optional[int] = None

from collections import deque

class CompressionTriggerRequest(BaseModel):
    force: bool = False

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    source: str = "system"

_custom_instruction_override: Optional[str] = None
_custom_instruction_name: str = "Default Persona"
LOG_BUFFER: deque = deque(maxlen=200)

def get_active_instruction() -> tuple[str, str]:
    """Get current active instruction and its name."""
    global _custom_instruction_override, _custom_instruction_name
    if _custom_instruction_override:
        return (_custom_instruction_override, _custom_instruction_name)
    return (DEFAULT_PERSONA_INSTRUCTION, "Default Persona")

# ==================== CONNECTION MANAGER ====================

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

class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                "level": record.levelname,
                "message": self.format(record),
                "source": record.name
            }
            LOG_BUFFER.append(log_entry)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(manager.broadcast("log", log_entry))
            except RuntimeError:
                pass
        except Exception:
            self.handleError(record)

from src.brain.hippocampus import Hippocampus
from src.brain.brainstem import BrainStem

manager = ConnectionManager()
hippocampus = Hippocampus()

# ==================== EVENT BRIDGING ====================
async def bridge_neural_events(event: dict):
    """Forward neural events to dashboard via WebSocket."""
    # Add a flag to prevent infinite loops if dashboard emits events back
    event_data = event.copy()
    event_type = f"{event['source']}_event"
    
    # Specific dashboard event mapping
    if event['type'] == 'memory_update':
        event_type = 'memory_update'
    elif event['type'] == 'triple_update':
        event_type = 'triple_update'
    elif event['type'] == 'schedule_update':
        event_type = 'schedule_update'
    elif event['type'] == 'entity_update':
        event_type = 'entity_update'
    
    await manager.broadcast(event_type, event_data)


    
def get_mongo() -> MongoDBClient:
    """Get MongoDB client instance."""
    # Use hippocampus's client to ensure single connection pool if possible, 
    # or just use the global one (they are singletons usually)
    return get_mongo_client()

# ==================== FASTAPI APP ====================

app = FastAPI(title="Vira Occipital Lobe", description="Visual Processing & UI Backend - Complete CRUD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "..", "dashboard")
DASHBOARD_DIST = os.path.join(DASHBOARD_DIR, "dist")

# ==================== DASHBOARD & WEBSOCKET ====================

if os.path.exists(os.path.join(DASHBOARD_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DASHBOARD_DIST, "assets")), name="assets")

@app.on_event("startup")
async def startup_event():
    await hippocampus.initialize()
    NeuralEventBus.subscribe(bridge_neural_events)
    logger.info("🧠 Occipital Lobe connected to Hippocampus & Neural Event Bus")

@app.on_event("shutdown")
async def shutdown_event():
    await hippocampus.close()

@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    html_path = os.path.join(DASHBOARD_DIST, "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Dashboard not built. Please run 'npm run build' in src/dashboard</h1>"

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

# ==================== SYSTEM ENDPOINTS ====================

@app.get("/api/logs/history")
async def get_log_history():
    """Get recent system logs."""
    return list(LOG_BUFFER)

@app.get("/api/health")
async def get_health():
    """Health check endpoint."""
    try:
        mongo = get_mongo()
        await mongo.db.command('ping')
        return {
            "status": "healthy",
            "database": "mongodb",
            "timestamp": datetime.now().isoformat(),
            "admin_id": ADMIN_ID
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/api/stats")
async def get_stats():
    """Get comprehensive database statistics."""
    mongo = get_mongo()
    try:
        mem_count = await mongo.memories.count_documents({"status": "active"})
        triple_count = await mongo.knowledge_graph.count_documents({})
        sched_count = await mongo.schedules.count_documents({"status": "pending"})
        entity_count = await mongo.entities.count_documents({})
        chat_count = await mongo.chat_logs.count_documents({})
        
        uncompressed_count = await mongo.memories.count_documents({
            "status": "active",
            "$or": [{"is_compressed": False}, {"is_compressed": {"$exists": False}}]
        })
        global_ctx = await mongo.db["global_context"].find_one({"_id": "current"})
        persona_count = await mongo.db["personas"].count_documents({})

        return {
            "memories": mem_count,
            "triples": triple_count,
            "pending_schedules": sched_count,
            "entities": entity_count,
            "chat_logs": chat_count,
            "uncompressed_memories": uncompressed_count,
            "personas": persona_count,
            "has_global_context": global_ctx is not None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==================== MEMORIES CRUD ====================

@app.get("/api/memories")
async def get_memories(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    status: Optional[str] = Query(None),
    memory_type: Optional[str] = Query(None)
):
    """Get memories with filtering and pagination via Hippocampus."""
    return await hippocampus.list_memories(
        limit=limit,
        skip=skip,
        status=status,
        memory_type=memory_type
    )

@app.get("/api/memories/{memory_id}")
async def get_memory_by_id(memory_id: str):
    """Get a specific memory by ID."""
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

@app.post("/api/memories")
async def create_memory(memory: MemoryCreate):
    """Create a new memory via Hippocampus (triggers embedding & canonicalization)."""
    try:
        memory_id = await hippocampus.store(
            summary=memory.summary,
            memory_type=memory.memory_type,
            priority=memory.priority,
            embedding=memory.embedding
        )
        return {"id": memory_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/memories/{memory_id}")
async def update_memory(memory_id: str, memory: MemoryUpdate):
    """Update a memory via Hippocampus."""
    updates = memory.dict(exclude_unset=True)
    # Map memory_type to type for compatibility
    if "memory_type" in updates:
        updates["type"] = updates.pop("memory_type")
        
    success = await hippocampus.update_memory(memory_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found or update failed")
    return {"status": "updated", "id": memory_id}

@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str, hard_delete: bool = Query(False)):
    """Delete (soft/hard) a memory via Hippocampus."""
    success = await hippocampus.delete_memory(memory_id, hard_delete=hard_delete)
    if not success:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"status": "deleted", "id": memory_id}

@app.post("/api/memories/bulk-delete")
async def bulk_delete_memories(memory_ids: List[str] = Body(...), hard_delete: bool = Body(False)):
    """Bulk delete multiple memories."""
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

# ==================== KNOWLEDGE GRAPH / TRIPLES CRUD ====================

@app.get("/api/triples")
async def get_triples(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    subject: Optional[str] = Query(None),
    predicate: Optional[str] = Query(None)
):
    """Get knowledge graph triples with filtering."""
    mongo = get_mongo()
    
    filter_query = {}
    if subject:
        filter_query["subject"] = {"$regex": subject, "$options": "i"}
    if predicate:
        filter_query["predicate"] = predicate
    
    cursor = mongo.knowledge_graph.find(filter_query).sort("last_accessed", -1).skip(skip).limit(limit)
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

@app.get("/api/triples/{triple_id}")
async def get_triple_by_id(triple_id: str):
    """Get a specific triple by ID."""
    mongo = get_mongo()
    doc = await mongo.knowledge_graph.find_one({"_id": triple_id})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Triple not found")
    
    return {
        "id": str(doc["_id"]),
        "subject": doc.get("subject"),
        "predicate": doc.get("predicate"),
        "object": doc.get("object"),
        "confidence": doc.get("confidence", 0.8),
        "source_memory_id": doc.get("source_memory_id"),
        "access_count": doc.get("access_count", 0),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
        "last_accessed": doc.get("last_accessed").isoformat() if doc.get("last_accessed") else None
    }

@app.post("/api/triples")
async def create_triple(triple: TripleCreate):
    """Create a new knowledge graph triple via Hippocampus."""
    try:
        triple_id = await hippocampus.add_triple(
            subject=triple.subject,
            predicate=triple.predicate,
            obj=triple.object,
            confidence=triple.confidence,
            source_memory_id=triple.source_memory_id
        )
        return {"id": triple_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/triples/{triple_id}")
async def update_triple(triple_id: str, triple: TripleUpdate):
    """Update triple via Hippocampus."""
    updates = triple.dict(exclude_unset=True)
    success = await hippocampus.update_triple(triple_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Triple not found")
    return {"status": "updated", "id": triple_id}

@app.delete("/api/triples/{triple_id}")
async def delete_triple(triple_id: str):
    """Delete triple via Hippocampus."""
    success = await hippocampus.delete_triple(triple_id)
    if not success:
        raise HTTPException(status_code=404, detail="Triple not found")
    return {"status": "deleted", "id": triple_id}

@app.get("/api/triples")
async def get_triples(limit: int = 100, skip: int = 0):
    """Get triples via Hippocampus."""
    return await hippocampus.list_triples(limit=limit, skip=skip)

@app.get("/api/triples/query")
async def query_triples(entity: str, limit: int = 50):
    """Query triples where entity is subject or object."""
    # This falls back to direct mongo as hippocampus doesn't have a specific regex search helper yet
    # or we can use hippocampus.query_entity but that returns a dict structure, not a list of triples directly.
    # Let's check hippocampus.query_entity structure.
    # For now, let's implement the search logic here using hippocampus's mongo client to match previous behavior
    # but using the shared mongo definition.
    mongo = get_mongo()
    cursor = mongo.knowledge_graph.find({
        "$or": [
            {"subject": {"$regex": entity, "$options": "i"}},
            {"object": {"$regex": entity, "$options": "i"}}
        ]
    }).limit(limit)
    docs = await cursor.to_list(length=limit)
    return [
        {
            "id": str(d["_id"]),
            "subject": d.get("subject"),
            "predicate": d.get("predicate"),
            "object": d.get("object"),
            "confidence": d.get("confidence", 0.8)
        }
        for d in docs
    ]

# ==================== SCHEDULES CRUD ====================

@app.get("/api/schedules")
async def get_schedules(
    limit: int = Query(50, ge=1, le=500),
    upcoming: bool = Query(True),
    status: Optional[str] = Query(None)
):
    """Get schedules with filtering via Hippocampus."""
    if upcoming:
        # Use hippocampus helper for upcoming (pending) schedules
        return await hippocampus.get_pending_schedules(limit=limit)
    
    # Fallback to general list if not just upcoming or if status specified
    # Using direct mongo access as Hippocampus might not expose full generic history list yet
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

@app.get("/api/schedules/{schedule_id}")
async def get_schedule_by_id(schedule_id: str):
    """Get a specific schedule by ID."""
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

@app.post("/api/schedules")
async def create_schedule(schedule: ScheduleCreate):
    """Create a new schedule via Hippocampus (checks conflicts)."""
    try:
        schedule_id = await hippocampus.add_schedule(
            trigger_time=datetime.fromisoformat(schedule.scheduled_at),
            context=schedule.context,
            priority=schedule.priority
        )
        return {"id": schedule_id, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/schedules/{schedule_id}")
async def update_schedule(schedule_id: str, schedule: ScheduleUpdate):
    """Update a schedule via Hippocampus."""
    updates = schedule.dict(exclude_unset=True)
    success = await hippocampus.update_schedule(schedule_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found or update failed")
    return {"status": "updated", "id": schedule_id}

@app.delete("/api/schedules/{schedule_id}")
async def delete_schedule(schedule_id: str):
    """Delete a schedule via Hippocampus."""
    success = await hippocampus.delete_schedule(schedule_id)
    if not success:
        raise HTTPException(status_code=404, detail="Schedule not found")
    return {"status": "deleted", "id": schedule_id}

# ==================== ENTITIES CRUD ====================

@app.get("/api/entities")
async def get_entities(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    entity_type: Optional[str] = Query(None)
):
    """Get entities with filtering."""
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

@app.get("/api/entities/{entity_name}")
async def get_entity_by_name(entity_name: str):
    """Get a specific entity by name."""
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

@app.post("/api/entities")
async def create_entity(entity: EntityCreate):
    """Create a new entity."""
    mongo = get_mongo()
    entity_name = entity.name.lower().strip()
    
    # Check if entity already exists
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

@app.put("/api/entities/{entity_name}")
async def update_entity(entity_name: str, entity: EntityUpdate):
    """Update an existing entity."""
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

@app.delete("/api/entities/{entity_name}")
async def delete_entity(entity_name: str):
    """Delete an entity."""
    mongo = get_mongo()
    result = await mongo.entities.delete_one({"name": entity_name.lower()})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    await manager.broadcast("entity_update", {"action": "deleted", "name": entity_name})
    return {"status": "deleted"}

# ==================== CHAT LOGS CRUD ====================

@app.get("/api/chat-logs")
async def get_chat_logs(
    limit: int = Query(50, ge=1, le=500),
    skip: int = Query(0, ge=0),
    session_id: Optional[str] = Query(None)
):
    """Get chat logs with filtering."""
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

@app.get("/api/chat-logs/sessions")
async def get_chat_sessions():
    """Get list of unique chat sessions."""
    mongo = get_mongo()
    
    sessions = await mongo.chat_logs.distinct("session_id")
    return {"sessions": sessions, "count": len(sessions)}

@app.get("/api/chat-logs/{log_id}")
async def get_chat_log_by_id(log_id: str):
    """Get a specific chat log by ID."""
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

@app.post("/api/chat-logs")
async def create_chat_log(log: ChatLogCreate):
    """Create a new chat log entry."""
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

@app.delete("/api/chat-logs/{log_id}")
async def delete_chat_log(log_id: str):
    """Delete a chat log entry."""
    mongo = get_mongo()
    result = await mongo.chat_logs.delete_one({"_id": log_id})
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Chat log not found")
    
    await manager.broadcast("chat_log_update", {"action": "deleted", "id": log_id})
    return {"status": "deleted"}

@app.delete("/api/chat-logs/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete all chat logs for a session."""
    mongo = get_mongo()
    result = await mongo.chat_logs.delete_many({"session_id": session_id})
    
    await manager.broadcast("chat_log_update", {"action": "session_deleted", "session_id": session_id, "count": result.deleted_count})
    return {"status": "deleted", "count": result.deleted_count}

# ==================== ADMIN PROFILE CRUD ====================

@app.get("/api/admin-profile")
async def get_admin_profile():
    """Get admin profile."""
    mongo = get_mongo()
    doc = await mongo.admin_profile.find_one({"_id": "admin"})
    
    if not doc:
        return {
            "telegram_name": None,
            "full_name": None,
            "additional_info": None,
            "last_updated": None
        }
    
    return {
        "telegram_name": doc.get("telegram_name"),
        "full_name": doc.get("full_name"),
        "additional_info": doc.get("additional_info"),
        "last_updated": doc.get("last_updated").isoformat() if doc.get("last_updated") else None
    }

@app.put("/api/admin-profile")
async def update_admin_profile(profile: AdminProfileUpdate):
    """Update admin profile."""
    mongo = get_mongo()
    
    update_data = {"last_updated": datetime.now()}
    if profile.telegram_name:
        update_data["telegram_name"] = profile.telegram_name
    if profile.full_name:
        update_data["full_name"] = profile.full_name
    if profile.additional_info:
        update_data["additional_info"] = profile.additional_info
    
    await mongo.admin_profile.update_one(
        {"_id": "admin"},
        {"$set": update_data},
        upsert=True
    )
    
    await manager.broadcast("admin_profile_update", {"action": "updated"})
    return {"status": "updated"}

@app.delete("/api/admin-profile")
async def delete_admin_profile():
    """Delete admin profile (reset to default)."""
    mongo = get_mongo()
    result = await mongo.admin_profile.delete_one({"_id": "admin"})
    
    await manager.broadcast("admin_profile_update", {"action": "deleted"})
    return {"status": "deleted", "deleted": result.deleted_count > 0}

# ==================== PERSONAS CRUD ====================

@app.get("/api/personas")
async def get_personas():
    """Get all saved personas."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return await brain.hippocampus.get_personas()

@app.get("/api/personas/active")
async def get_active_persona():
    """Get the currently active persona."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    persona = await brain.hippocampus.get_active_persona()
    if not persona:
        return {"status": "none", "persona": None}
    
    return {"status": "active", "persona": persona}

@app.get("/api/personas/{persona_id}")
async def get_persona_by_id(persona_id: str):
    """Get a specific persona by ID."""
    mongo = get_mongo()
    doc = await mongo.db["personas"].find_one({"id": persona_id})
    
    if not doc:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    return {
        "id": doc.get("id"),
        "name": doc.get("name"),
        "instruction": doc.get("instruction"),
        "temperature": doc.get("temperature", 0.7),
        "description": doc.get("description"),
        "is_active": doc.get("is_active", False),
        "created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None
    }

@app.post("/api/personas")
async def create_persona(data: PersonaCreate):
    """Create a new persona profile."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    try:
        pid = await brain.hippocampus.create_persona(
            data.name, 
            data.instruction, 
            data.temperature,
            data.description
        )
        await manager.broadcast("personas_update", {"action": "created", "id": pid})
        
        await NeuralEventBus.emit(
            "occipital_lobe", "hippocampus", "persona_created",
            payload={"persona_id": pid, "name": data.name}
        )
        
        return {"id": pid, "status": "created"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/personas/{persona_id}/activate")
async def activate_persona(persona_id: str):
    """Switch the active persona."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    success = await brain.hippocampus.set_active_persona(persona_id)
    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")
        
    global _custom_instruction_override, _custom_instruction_name
    _custom_instruction_override = None
    _custom_instruction_name = "Default Persona"
    
    persona = await brain.hippocampus.get_active_persona()
    
    if brain.prefrontal_cortex:
        await brain.prefrontal_cortex.switch_persona(persona_id)
    
    await manager.broadcast("instruction_update", {
        "name": persona["name"],
        "is_custom": True,
        "temperature": persona.get("temperature", 0.7)
    })
    
    await NeuralEventBus.emit(
        "occipital_lobe", "prefrontal_cortex", "persona_switched",
        payload={"persona_id": persona_id, "name": persona["name"]}
    )
    
    return {"status": "activated", "persona": persona}

@app.put("/api/personas/{persona_id}")
async def update_persona(persona_id: str, data: PersonaUpdate):
    """Update persona settings."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    updates = {k: v for k, v in data.dict().items() if v is not None}
    success = await brain.hippocampus.update_persona(persona_id, updates)
    if not success:
        raise HTTPException(status_code=404, detail="Persona not found")
        
    await manager.broadcast("personas_update", {"action": "updated", "id": persona_id})
    return {"status": "updated"}

@app.delete("/api/personas/{persona_id}")
async def delete_persona(persona_id: str):
    """Delete a persona profile."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    try:
        success = await brain.hippocampus.delete_persona(persona_id)
        if not success:
            raise HTTPException(status_code=404, detail="Persona not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    await manager.broadcast("personas_update", {"action": "deleted", "id": persona_id})
    return {"status": "deleted"}

# ==================== GLOBAL CONTEXT CRUD ====================

@app.get("/api/global-context")
async def get_global_context():
    """Get global context."""
    mongo = get_mongo()
    doc = await mongo.db["global_context"].find_one({"_id": "current"})
    
    if not doc:
        return {
            "context_text": None,
            "metadata": {},
            "last_updated": None
        }
    
    return {
        "context_text": doc.get("context_text"),
        "metadata": doc.get("metadata", {}),
        "last_updated": doc.get("last_updated").isoformat() if doc.get("last_updated") else None
    }

@app.put("/api/global-context")
async def update_global_context(context: GlobalContextUpdate):
    """Update global context."""
    mongo = get_mongo()
    
    await mongo.db["global_context"].update_one(
        {"_id": "current"},
        {"$set": {
            "context_text": context.context_text,
            "metadata": context.metadata or {},
            "last_updated": datetime.now()
        }},
        upsert=True
    )
    
    await manager.broadcast("global_context_update", {"action": "updated"})
    return {"status": "updated"}

@app.delete("/api/global-context")
async def delete_global_context():
    """Delete global context."""
    mongo = get_mongo()
    result = await mongo.db["global_context"].delete_one({"_id": "current"})
    
    await manager.broadcast("global_context_update", {"action": "deleted"})
    return {"status": "deleted", "deleted": result.deleted_count > 0}

# ==================== SYSTEM CONFIG CRUD ====================

@app.get("/api/system-config")
async def get_system_config():
    """Get system configuration."""
    mongo = get_mongo()
    doc = await mongo.db["system_config"].find_one({"_id": "config"})
    
    if not doc:
        return {
            "chat_model": "openai/gpt-4",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_output_tokens": 512,
            "proactive_check_interval": 1800
        }
    
    return {
        "chat_model": doc.get("chat_model"),
        "temperature": doc.get("temperature"),
        "top_p": doc.get("top_p"),
        "max_output_tokens": doc.get("max_output_tokens"),
        "proactive_check_interval": doc.get("proactive_check_interval"),
        "last_updated": doc.get("last_updated").isoformat() if doc.get("last_updated") else None
    }

@app.put("/api/system-config")
async def update_system_config(config: SystemConfigUpdate):
    """Update system configuration."""
    mongo = get_mongo()
    
    update_data = {"last_updated": datetime.now()}
    if config.chat_model:
        update_data["chat_model"] = config.chat_model
    if config.temperature is not None:
        update_data["temperature"] = config.temperature
    if config.top_p is not None:
        update_data["top_p"] = config.top_p
    if config.max_output_tokens is not None:
        update_data["max_output_tokens"] = config.max_output_tokens
    if config.proactive_check_interval is not None:
        update_data["proactive_check_interval"] = config.proactive_check_interval
    
    await mongo.db["system_config"].update_one(
        {"_id": "config"},
        {"$set": update_data},
        upsert=True
    )
    
    await manager.broadcast("system_config_update", {"action": "updated"})
    # Notify BrainStem and other modules via NeuralEventBus
    await NeuralEventBus.emit("occipital", "all", "system_config_update", payload=update_data)
    
    return {"status": "updated"}

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

# ==================== CONTEXT & ANALYSIS ENDPOINTS ====================

@app.get("/api/context/stats")
async def get_context_stats():
    """Get context building statistics."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    
    if not brain or not brain.thalamus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    return await brain.thalamus.get_context_stats()

@app.get("/api/neural-events")
async def get_neural_events(limit: int = Query(20, ge=1, le=100)):
    """Get recent neural activity events."""
    events = NeuralEventBus.get_recent_events(limit)
    return {"events": events, "count": len(events)}

@app.get("/api/module-states")
async def get_module_states():
    """Get current state of all brain modules."""
    states = NeuralEventBus.get_module_states()
    return {"modules": states, "timestamp": datetime.now().isoformat()}

# ==================== MAINTENANCE ENDPOINTS ====================

@app.post("/api/maintenance/trigger")
async def trigger_maintenance():
    """Manually trigger maintenance tasks."""
    from src.brain.cerebellum import trigger_maintenance_manual
    from src.brain.brainstem import get_brain
    
    brain = get_brain()
    if not brain:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    result = await trigger_maintenance_manual(brain)
    
    if result["status"] == "success":
        await manager.broadcast("maintenance_update", {
            "action": "completed",
            "results": result.get("results", {})
        })
    
    return result

@app.post("/api/maintenance/compress-memories")
async def trigger_memory_compression(request: CompressionTriggerRequest):
    """Trigger memory compression."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    compressed = await brain.hippocampus.compress_memories(force=request.force)
    
    await manager.broadcast("compression_update", {
        "action": "completed",
        "compressed_count": compressed
    })
    
    return {"status": "completed", "compressed": compressed}

@app.post("/api/maintenance/optimize-graph")
async def optimize_knowledge_graph():
    """Optimize knowledge graph by removing low-confidence triples."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    await brain.hippocampus.optimize_knowledge_graph()
    
    await manager.broadcast("graph_update", {"action": "optimized"})
    return {"status": "optimized"}

# ==================== OPENROUTER ENDPOINTS ====================

@app.get("/api/openrouter/models")
async def get_openrouter_models():
    """Get available OpenRouter models organized by tier."""
    from src.brain.brainstem import OPENROUTER_MODELS
    return OPENROUTER_MODELS

@app.get("/api/openrouter/health")
async def get_model_health():
    """Get model health scores."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    
    if not brain or not brain.openrouter:
        raise HTTPException(status_code=503, detail="OpenRouter client not initialized")
    
    health = brain.openrouter.get_model_health()
    
    from src.brain.brainstem import OPENROUTER_MODELS
    for model, stats in health.items():
        for tier, models in OPENROUTER_MODELS.items():
            if model in models:
                stats["tier"] = tier
                break
        if "tier" not in stats:
            stats["tier"] = "custom"
            
    return health

@app.post("/api/openrouter/reset-health")
async def reset_model_health():
    """Reset model health scores."""
    from src.brain.brainstem import get_brain
    brain = get_brain()
    if not brain or not brain.openrouter:
        raise HTTPException(status_code=503, detail="OpenRouter client not initialized")
    
    brain.openrouter.reset_health()
    await manager.broadcast("openrouter_update", {"action": "health_reset"})
    return {"status": "reset"}

# ==================== DATABASE MANAGEMENT ENDPOINTS ====================

@app.get("/api/database/collections")
async def get_collections():
    """Get list of all database collections."""
    mongo = get_mongo()
    collections = await mongo.db.list_collection_names()
    return {"collections": collections, "count": len(collections)}

@app.get("/api/database/collection/{collection_name}/count")
async def get_collection_count(collection_name: str):
    """Get document count for a specific collection."""
    mongo = get_mongo()
    count = await mongo.db[collection_name].count_documents({})
    return {"collection": collection_name, "count": count}

@app.delete("/api/database/collection/{collection_name}")
async def drop_collection(collection_name: str, confirm: bool = Query(False)):
    """Drop an entire collection (requires confirmation)."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Collection drop requires confirmation. Add ?confirm=true to the request."
        )
    
    mongo = get_mongo()
    await mongo.db[collection_name].drop()
    
    await manager.broadcast("database_update", {
        "action": "collection_dropped",
        "collection": collection_name
    })
    
    return {"status": "dropped", "collection": collection_name}

# ==================== SEARCH & QUERY ENDPOINTS ====================

@app.get("/api/search/memories")
async def search_memories(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Search memories by text query."""
    mongo = get_mongo()
    
    cursor = mongo.memories.find({
        "summary": {"$regex": query, "$options": "i"},
        "status": "active"
    }).limit(limit)
    
    docs = await cursor.to_list(length=limit)
    return [
        {
            "id": str(d["_id"]),
            "summary": d.get("summary", ""),
            "memory_type": d.get("type", "general"),
            "priority": d.get("priority", 0.5),
            "confidence": d.get("confidence", 0.5)
        }
        for d in docs
    ]

@app.get("/api/search/entities")
async def search_entities(
    query: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50)
):
    """Search entities by name."""
    mongo = get_mongo()
    
    cursor = mongo.entities.find({
        "name": {"$regex": query, "$options": "i"}
    }).limit(limit)
    
    docs = await cursor.to_list(length=limit)
    return [
        {
            "name": d.get("name"),
            "entity_type": d.get("entity_type", "unknown"),
            "mention_count": d.get("mention_count", 0)
        }
        for d in docs
    ]

# ==================== EXPORT & IMPORT ENDPOINTS ====================

@app.get("/api/export/memories")
async def export_memories(limit: int = Query(1000, ge=1, le=10000)):
    """Export memories to JSON format."""
    mongo = get_mongo()
    cursor = mongo.memories.find({"status": "active"}).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    export_data = []
    for d in docs:
        d.pop("_id", None)
        if "created_at" in d:
            d["created_at"] = d["created_at"].isoformat()
        if "last_used" in d:
            d["last_used"] = d["last_used"].isoformat()
        export_data.append(d)
    
    return {
        "export_date": datetime.now().isoformat(),
        "count": len(export_data),
        "data": export_data
    }

@app.get("/api/export/knowledge-graph")
async def export_knowledge_graph(limit: int = Query(1000, ge=1, le=10000)):
    """Export knowledge graph triples to JSON format."""
    mongo = get_mongo()
    cursor = mongo.knowledge_graph.find({}).limit(limit)
    docs = await cursor.to_list(length=limit)
    
    export_data = []
    for d in docs:
        d.pop("_id", None)
        if "created_at" in d:
            d["created_at"] = d["created_at"].isoformat()
        if "last_accessed" in d:
            d["last_accessed"] = d["last_accessed"].isoformat()
        export_data.append(d)
    
    return {
        "export_date": datetime.now().isoformat(),
        "count": len(export_data),
        "data": export_data
    }

# ==================== RUN SERVER ====================

def run_dashboard(host: str = "0.0.0.0", port: int = 5000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    run_dashboard()