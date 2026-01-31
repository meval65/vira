import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from src.brain.infrastructure.neural_event_bus import NeuralEventBus
from src.brain.brainstem import get_brain
from src.brain.occipital_lobe.state import manager
from src.brain.occipital_lobe.api import (
    memories, triples, schedules, entities, chat_logs, system,
    personas, config, maintenance, openrouter, neural_events, search
)

logger = logging.getLogger(__name__)

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


def _sanitize_for_json(obj):
    """Convert BSON/ObjectId and other non-JSON types so FastAPI can serialize."""
    if ObjectId is not None and isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


_original_jsonable_encoder = jsonable_encoder


def _custom_jsonable_encoder(obj, **kwargs):
    return _original_jsonable_encoder(_sanitize_for_json(obj), **kwargs)


app = FastAPI(title="Vira Occipital Lobe", description="Visual Processing & UI Backend - Complete CRUD API")

# Ensure JSON responses can serialize MongoDB ObjectId
import fastapi.encoders
fastapi.encoders.jsonable_encoder = _custom_jsonable_encoder

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "dashboard"
DASHBOARD_DIST = DASHBOARD_DIR / "dist"
DASHBOARD_INDEX = DASHBOARD_DIST / "index.html"
ASSETS_DIR = DASHBOARD_DIST / "assets"

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

# Serve SPA index first so "/" returns dashboard (not overridden by routers)
@app.get("/", include_in_schema=False)
def serve_dashboard():
    if DASHBOARD_INDEX.exists():
        return FileResponse(
            str(DASHBOARD_INDEX),
            media_type="text/html",
        )
    return {"message": "Dashboard not built. Run: cd src/dashboard && npm install && npm run build"}

app.include_router(memories.router)
app.include_router(triples.router)
app.include_router(schedules.router)
app.include_router(entities.router)
app.include_router(chat_logs.router)
app.include_router(system.router)
app.include_router(personas.router)
app.include_router(config.router)
app.include_router(maintenance.router)
app.include_router(openrouter.router)
app.include_router(neural_events.router)
app.include_router(search.router)


async def bridge_neural_events(event: dict):
    event_data = event.copy()
    event_type = f"{event['source']}_event"
    
    if event['type'] == 'memory_update':
        event_type = 'memory_update'
    elif event['type'] == 'triple_update':
        event_type = 'triple_update'
    elif event['type'] == 'schedule_update':
        event_type = 'schedule_update'
    elif event['type'] == 'entity_update':
        event_type = 'entity_update'
    
    await manager.broadcast(event_type, event_data)

@app.on_event("startup")
async def startup_event():
    await get_brain()
    await NeuralEventBus.subscribe(bridge_neural_events)
    logger.info("🧠 Occipital Lobe connected to Hippocampus & Neural Event Bus")

@app.on_event("shutdown")
async def shutdown_event():
    brain = await get_brain()
    if brain:
        await brain.shutdown()
