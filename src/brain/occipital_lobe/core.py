import os
import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.brain.infrastructure.neural_event_bus import NeuralEventBus
from src.brain.brainstem import get_brain
from src.brain.occipital_lobe.state import manager
from src.brain.occipital_lobe.api import (
    memories, triples, schedules, entities, chat_logs, system,
    personas, config, maintenance, openrouter, neural_events, search
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Vira Occipital Lobe", description="Visual Processing & UI Backend - Complete CRUD API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dashboard")
DASHBOARD_DIST = os.path.join(DASHBOARD_DIR, "dist")
if os.path.exists(os.path.join(DASHBOARD_DIST, "assets")):
    app.mount("/assets", StaticFiles(directory=os.path.join(DASHBOARD_DIST, "assets")), name="assets")

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
