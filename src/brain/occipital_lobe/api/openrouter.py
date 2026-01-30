from fastapi import APIRouter
from src.brain.brainstem import get_brain, OPENROUTER_MODELS

router = APIRouter(prefix="/api/openrouter", tags=["openrouter"])


@router.get("/models")
async def get_models():
    brain = await get_brain()
    
    models = {}
    for tier, model_list in OPENROUTER_MODELS.items():
        models[tier] = [{"id": m, "name": m.split("/")[-1]} for m in model_list]
    
    return {"models": models, "tiers": list(OPENROUTER_MODELS.keys())}


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
        "model_health": status.get("model_health", {}),
        "active_tier": status.get("active_tier"),
        "total_requests": status.get("total_requests", 0)
    }


@router.post("/reset-health")
async def reset_health():
    brain = await get_brain()
    
    if not brain or not brain.openrouter:
        return {"status": "not_initialized"}
    
    brain.openrouter._model_health.clear()
    
    return {"status": "reset"}
