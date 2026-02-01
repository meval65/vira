"""
Brain State API - Real-time status of all neural modules.

Provides comprehensive view of the brain's current state for dashboard visualization.
"""

from fastapi import APIRouter
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.brain.brainstem import get_brain, MODEL_LIST, VISION_MODEL_LIST
from src.brain.infrastructure.neural_event_bus import NeuralEventBus

router = APIRouter(prefix="/api/brain-state", tags=["brain_state"])


# Module definitions with their descriptions
BRAIN_MODULES = {
    "brainstem": {
        "name": "Brainstem",
        "description": "Central coordinator and message processing",
        "color": "#ef4444"
    },
    "hippocampus": {
        "name": "Hippocampus",
        "description": "Long-term memory storage and retrieval",
        "color": "#3b82f6"
    },
    "amygdala": {
        "name": "Amygdala",
        "description": "Emotional state and personality management",
        "color": "#f59e0b"
    },
    "prefrontal_cortex": {
        "name": "Prefrontal Cortex",
        "description": "Reasoning, planning, and decision making",
        "color": "#8b5cf6"
    },
    "thalamus": {
        "name": "Thalamus",
        "description": "Session management and context relay",
        "color": "#22c55e"
    },
    "motor_cortex": {
        "name": "Motor Cortex",
        "description": "Response generation and output",
        "color": "#ec4899"
    },
    "parietal_lobe": {
        "name": "Parietal Lobe",
        "description": "Tool execution and reflexes",
        "color": "#06b6d4"
    },
    "occipital_lobe": {
        "name": "Occipital Lobe",
        "description": "Dashboard and visual interface",
        "color": "#a855f7"
    },
    "cerebellum": {
        "name": "Cerebellum",
        "description": "Background tasks and scheduling",
        "color": "#14b8a6"
    }
}

# Neural connections between modules
NEURAL_CONNECTIONS = [
    {"source": "brainstem", "target": "hippocampus", "type": "bidirectional"},
    {"source": "brainstem", "target": "amygdala", "type": "bidirectional"},
    {"source": "brainstem", "target": "prefrontal_cortex", "type": "bidirectional"},
    {"source": "brainstem", "target": "thalamus", "type": "bidirectional"},
    {"source": "brainstem", "target": "motor_cortex", "type": "output"},
    {"source": "brainstem", "target": "parietal_lobe", "type": "bidirectional"},
    {"source": "brainstem", "target": "occipital_lobe", "type": "output"},
    {"source": "brainstem", "target": "cerebellum", "type": "bidirectional"},
    {"source": "hippocampus", "target": "prefrontal_cortex", "type": "memory_flow"},
    {"source": "amygdala", "target": "prefrontal_cortex", "type": "emotional_context"},
    {"source": "prefrontal_cortex", "target": "motor_cortex", "type": "response_plan"},
    {"source": "thalamus", "target": "hippocampus", "type": "session_context"},
    {"source": "parietal_lobe", "target": "hippocampus", "type": "tool_result"},
]


@router.get("")
async def get_brain_state():
    """Get complete real-time brain state."""
    brain = await get_brain()
    
    # Get current activity from NeuralEventBus
    current_activity = NeuralEventBus.get_current_activity()
    recent_events = NeuralEventBus.get_recent_events(limit=20)
    
    # Build module states
    modules = {}
    for module_id, info in BRAIN_MODULES.items():
        activity = current_activity.get(module_id, "")
        status = "active" if activity else "idle"
        
        module_data = {
            "id": module_id,
            "name": info["name"],
            "description": info["description"],
            "color": info["color"],
            "status": status,
            "activity": activity if activity else None
        }
        
        # Add module-specific data
        if brain:
            if module_id == "hippocampus" and brain.hippocampus:
                try:
                    stats = await brain.hippocampus.get_stats()
                    module_data["data"] = {
                        "memory_count": stats.get("total_memories", 0),
                        "entity_count": stats.get("total_entities", 0)
                    }
                except:
                    pass
            
            elif module_id == "amygdala" and brain.amygdala:
                try:
                    emotion_data = brain.amygdala.get_emotional_state()
                    module_data["data"] = {
                        "current_emotion": emotion_data.get("primary_emotion", "neutral"),
                        "intensity": emotion_data.get("intensity", 0.5),
                        "persona": emotion_data.get("active_persona", "default")
                    }
                except:
                    pass
            
            elif module_id == "thalamus" and brain.thalamus:
                try:
                    session_data = brain.thalamus.get_session_summary()
                    module_data["data"] = {
                        "active_sessions": session_data.get("active_count", 0),
                        "messages_today": session_data.get("messages_today", 0)
                    }
                except:
                    pass
                    
            elif module_id == "brainstem" and brain.openrouter:
                try:
                    status_data = brain.openrouter.get_status()
                    module_data["data"] = {
                        "api_configured": status_data.get("api_configured", False),
                        "primary_model": MODEL_LIST[0] if MODEL_LIST else "none"
                    }
                except:
                    pass
        
        modules[module_id] = module_data
    
    # Get system health
    system_health = {
        "status": "healthy",
        "brain_initialized": brain is not None,
        "timestamp": datetime.now().isoformat()
    }
    
    if brain:
        system_health["openrouter_api"] = brain.openrouter.get_status().get("api_configured", False) if brain.openrouter else False
    
    return {
        "modules": modules,
        "connections": NEURAL_CONNECTIONS,
        "recent_events": recent_events,
        "current_activity": current_activity,
        "system_health": system_health,
        "model_list": MODEL_LIST,
        "vision_model_list": VISION_MODEL_LIST
    }


@router.get("/modules/{module_id}")
async def get_module_state(module_id: str):
    """Get detailed state of a specific module."""
    brain = await get_brain()
    
    if module_id not in BRAIN_MODULES:
        return {"error": f"Unknown module: {module_id}"}
    
    info = BRAIN_MODULES[module_id]
    current_activity = NeuralEventBus.get_current_activity()
    activity = current_activity.get(module_id, "")
    
    module_data = {
        "id": module_id,
        "name": info["name"],
        "description": info["description"],
        "color": info["color"],
        "status": "active" if activity else "idle",
        "activity": activity if activity else None,
        "connections": [c for c in NEURAL_CONNECTIONS if c["source"] == module_id or c["target"] == module_id]
    }
    
    # Add module-specific detailed data
    if brain:
        if module_id == "hippocampus" and brain.hippocampus:
            stats = await brain.hippocampus.get_stats()
            module_data["details"] = stats
            
        elif module_id == "amygdala" and brain.amygdala:
            module_data["details"] = {
                "emotional_state": brain.amygdala.get_emotional_state(),
                "persona_info": brain.amygdala.get_persona_info() if hasattr(brain.amygdala, 'get_persona_info') else {}
            }
            
        elif module_id == "thalamus" and brain.thalamus:
            module_data["details"] = brain.thalamus.get_session_summary() if hasattr(brain.thalamus, 'get_session_summary') else {}
            
        elif module_id == "parietal_lobe" and brain.parietal_lobe:
            module_data["details"] = {
                "available_tools": len(brain.parietal_lobe.reflexes) if hasattr(brain.parietal_lobe, 'reflexes') else 0
            }
    
    return module_data


@router.get("/activity")
async def get_current_activity():
    """Get current activity of all modules."""
    return {
        "activity": NeuralEventBus.get_current_activity(),
        "timestamp": datetime.now().isoformat()
    }


@router.get("/events/stream")
async def get_events_stream(limit: int = 50):
    """Get recent events for streaming updates."""
    events = NeuralEventBus.get_recent_events(limit=limit)
    return {
        "events": events,
        "count": len(events),
        "timestamp": datetime.now().isoformat()
    }
