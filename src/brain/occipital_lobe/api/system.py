import os
from datetime import datetime
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from src.brain.brainstem import ADMIN_ID, get_brain
from src.brain.db.mongo_client import get_mongo_client
from src.brain.occipital_lobe.types import AdminProfileUpdate, LogEntry
from src.brain.occipital_lobe.state import LOG_BUFFER, manager

router = APIRouter(tags=["system"])

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "dashboard")

def get_mongo():
    return get_mongo_client()

@router.get("/", response_class=HTMLResponse)
async def dashboard_page():
    html_path = os.path.join(DASHBOARD_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "<h1>Dashboard not found</h1>"

@router.get("/styles.css")
async def styles_css():
    css_path = os.path.join(DASHBOARD_DIR, "styles.css")
    if os.path.exists(css_path):
        return FileResponse(css_path, media_type="text/css")
    raise HTTPException(status_code=404)

@router.get("/app.js")
async def app_js():
    js_path = os.path.join(DASHBOARD_DIR, "app.js")
    if os.path.exists(js_path):
        return FileResponse(js_path, media_type="application/javascript")
    raise HTTPException(status_code=404)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

@router.get("/api/logs/history")
async def get_log_history():
    return list(LOG_BUFFER)

@router.get("/api/health")
async def get_health():
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
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@router.get("/api/stats")
async def get_stats():
    mongo = get_mongo()
    
    memory_count = await mongo.memories.count_documents({})
    schedule_count = await mongo.schedules.count_documents({})
    persona_count = await mongo.personas.count_documents({})
    entity_count = await mongo.entities.count_documents({})
    triple_count = await mongo.knowledge_graph.count_documents({})
    chat_count = await mongo.chat_logs.count_documents({})
    
    return {
        "memories": memory_count,
        "schedules": schedule_count,
        "personas": persona_count,
        "entities": entity_count,
        "triples": triple_count,
        "chat_logs": chat_count
    }

@router.get("/api/admin/profile")
async def get_admin_profile():
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    profile = brain.hippocampus.admin_profile
    return {
        "telegram_name": profile.telegram_name,
        "additional_info": profile.additional_info
    }

@router.put("/api/admin/profile")
async def update_admin_profile(data: AdminProfileUpdate):
    brain = await get_brain()
    if not brain or not brain.hippocampus:
        raise HTTPException(status_code=503, detail="Brain not initialized")
    
    brain.hippocampus.update_admin_profile(
        name=data.telegram_name,
        info=data.additional_info
    )
    
    return {"status": "updated"}
