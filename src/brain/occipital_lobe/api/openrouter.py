from fastapi import APIRouter
from src.brain.brainstem import get_brain, MODEL_LIST, VISION_MODEL_LIST

router = APIRouter(prefix="/api/openrouter", tags=["openrouter"])


@router.get("/models")
async def get_models():
    """Get list of available models."""
    brain = await get_brain()
    
    models = {
        "text": [{"id": m, "name": m.split("/")[-1]} for m in MODEL_LIST],
        "vision": [{"id": m, "name": m.split("/")[-1]} for m in VISION_MODEL_LIST],
    }
    
    return {"models": models, "categories": ["text", "vision"]}


@router.get("/health")
async def get_health():
    brain = await get_brain()
    
    if not brain or not brain.openrouter:
        return {
            "api_configured": False,
            "status": "not_initialized",
            "model_health": {}
        }
    
    status = brain.openrouter.get_status()
    
    return {
        "api_configured": status.get("api_configured", False),
        "status": "healthy" if status.get("api_configured") else "no_api_key",
        "model_health": status.get("health_scores", {}),
        "total_models": len(MODEL_LIST) + len(VISION_MODEL_LIST)
    }


@router.post("/reset-health")
async def reset_health():
    brain = await get_brain()
    
    if not brain or not brain.openrouter:
        return {"status": "not_initialized"}
    
    brain.openrouter.health_scores.clear()
    
    return {"status": "reset"}
