import os
import asyncio
import logging
from datetime import datetime
from typing import Optional, Set, Dict, List, Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import aiosqlite

from src.brainstem import DB_PATH, ADMIN_ID

logger = logging.getLogger(__name__)


class MemoryCreate(BaseModel):
    summary: str
    memory_type: str = "general"
    priority: float = 0.5


class ScheduleCreate(BaseModel):
    context: str
    scheduled_at: str
    priority: int = 0


class AdminProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    additional_info: Optional[str] = None


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


async def get_db():
    db = await aiosqlite.connect(DB_PATH, timeout=30.0)
    db.row_factory = aiosqlite.Row
    return db


app = FastAPI(
    title="Vira Dashboard API",
    description="Complete System Dashboard for Vira Personal Life OS",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


DASHBOARD_DIR = os.path.join(os.path.dirname(__file__), "dashboard")


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
        db = await get_db()
        await db.execute("SELECT 1")
        await db.close()
        return {"status": "healthy", "timestamp": datetime.now().isoformat(), "admin_id": ADMIN_ID}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/api/stats")
async def get_stats():
    db = await get_db()
    try:
        memories = await db.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'")
        mem_count = (await memories.fetchone())[0]

        triples = await db.execute("SELECT COUNT(*) FROM triples")
        triple_count = (await triples.fetchone())[0]

        schedules = await db.execute("SELECT COUNT(*) FROM schedules WHERE status = 'pending'")
        sched_count = (await schedules.fetchone())[0]

        entities = await db.execute("SELECT COUNT(*) FROM entities")
        entity_count = (await entities.fetchone())[0]

        return {
            "memories": mem_count,
            "triples": triple_count,
            "pending_schedules": sched_count,
            "entities": entity_count
        }
    finally:
        await db.close()


@app.get("/api/emotional-state")
async def get_emotional_state():
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT current_mood, empathy_level, satisfaction_level FROM emotional_state WHERE id = 1"
        )
        row = await cursor.fetchone()
        if row:
            return {"mood": row[0], "empathy": row[1], "satisfaction": row[2]}
        return {"mood": "neutral", "empathy": 0.5, "satisfaction": 0.0}
    except Exception:
        return {"mood": "neutral", "empathy": 0.5, "satisfaction": 0.0}
    finally:
        await db.close()


@app.get("/api/memories")
async def get_memories(limit: int = 50):
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT id, summary, memory_type, priority, confidence, created_at, last_used_at
            FROM memories WHERE status = 'active'
            ORDER BY last_used_at DESC LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


@app.post("/api/memories")
async def create_memory(memory: MemoryCreate):
    import uuid
    db = await get_db()
    try:
        memory_id = str(uuid.uuid4())
        await db.execute("""
            INSERT INTO memories (id, summary, memory_type, priority, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (memory_id, memory.summary, memory.memory_type, memory.priority, datetime.now()))
        await db.commit()
        await manager.broadcast("memory_update", {"action": "created", "id": memory_id})
        return {"id": memory_id, "status": "created"}
    finally:
        await db.close()


@app.delete("/api/memories/{memory_id}")
async def delete_memory(memory_id: str):
    db = await get_db()
    try:
        await db.execute("UPDATE memories SET status = 'deleted' WHERE id = ?", (memory_id,))
        await db.commit()
        await manager.broadcast("memory_update", {"action": "deleted", "id": memory_id})
        return {"status": "deleted"}
    finally:
        await db.close()


@app.get("/api/schedules")
async def get_schedules(limit: int = 50):
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT id, scheduled_at, context, priority, status, recurrence
            FROM schedules ORDER BY scheduled_at ASC LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


@app.post("/api/schedules")
async def create_schedule(schedule: ScheduleCreate):
    from dateutil import parser
    db = await get_db()
    try:
        trigger_time = parser.parse(schedule.scheduled_at)
        cursor = await db.execute("""
            INSERT INTO schedules (scheduled_at, context, priority, created_at)
            VALUES (?, ?, ?, ?)
        """, (trigger_time, schedule.context, schedule.priority, datetime.now()))
        await db.commit()
        await manager.broadcast("schedule_update", {"action": "created", "id": cursor.lastrowid})
        return {"id": cursor.lastrowid, "status": "created"}
    finally:
        await db.close()


@app.delete("/api/schedules/{schedule_id}")
async def cancel_schedule(schedule_id: int):
    db = await get_db()
    try:
        await db.execute("UPDATE schedules SET status = 'cancelled' WHERE id = ?", (schedule_id,))
        await db.commit()
        await manager.broadcast("schedule_update", {"action": "cancelled", "id": schedule_id})
        return {"status": "cancelled"}
    finally:
        await db.close()


@app.get("/api/triples")
async def get_triples(limit: int = 50):
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT id, subject, predicate, object, confidence, access_count
            FROM triples ORDER BY access_count DESC LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


@app.get("/api/profile")
async def get_profile():
    db = await get_db()
    try:
        cursor = await db.execute(
            "SELECT telegram_name, full_name, additional_info FROM admin_profile WHERE id = 1"
        )
        row = await cursor.fetchone()
        if row:
            return {"telegram_name": row[0], "full_name": row[1], "additional_info": row[2]}
        return {"telegram_name": None, "full_name": None, "additional_info": None}
    except Exception:
        return {"telegram_name": None, "full_name": None, "additional_info": None}
    finally:
        await db.close()


@app.put("/api/profile")
async def update_profile(profile: AdminProfileUpdate):
    db = await get_db()
    try:
        await db.execute("""
            INSERT INTO admin_profile (id, full_name, additional_info, last_updated)
            VALUES (1, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                full_name = COALESCE(excluded.full_name, full_name),
                additional_info = COALESCE(excluded.additional_info, additional_info),
                last_updated = excluded.last_updated
        """, (profile.full_name, profile.additional_info, datetime.now()))
        await db.commit()
        return {"status": "updated"}
    finally:
        await db.close()


@app.get("/api/entities")
async def get_entities(limit: int = 50):
    db = await get_db()
    try:
        cursor = await db.execute("""
            SELECT id, name, entity_type, mention_count
            FROM entities ORDER BY mention_count DESC LIMIT ?
        """, (limit,))
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        await db.close()


def run_dashboard(host: str = "0.0.0.0", port: int = 5000):
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_dashboard()
