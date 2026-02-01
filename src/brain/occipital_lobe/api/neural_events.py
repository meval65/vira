from fastapi import APIRouter
from src.brain.infrastructure.neural_event_bus import NeuralEventBus

router = APIRouter(prefix="/api/neural-events", tags=["neural_events"])


@router.get("")
async def get_neural_events(limit: int = 50):
    events = NeuralEventBus.get_recent_events(limit=limit)
    
    return {
        "events": events,
        "count": len(events)
    }


