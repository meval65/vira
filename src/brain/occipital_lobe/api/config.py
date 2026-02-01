from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
from src.brain.infrastructure.mongo_client import get_mongo_client
from src.brain.brainstem import get_brain, SYSTEM_CONFIG

router = APIRouter(tags=["config"])


def get_mongo():
    return get_mongo_client()


class GlobalContextUpdate(BaseModel):
    context_text: str


class SystemConfigUpdate(BaseModel):
    chat_model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None


@router.get("/api/global-context")
async def get_global_context():
    mongo = get_mongo()
    doc = await mongo.db["global_context"].find_one({"_id": "current"})
    
    if not doc:
        return {
            "context_text": "",
            "metadata": {},
            "last_updated": None
        }
    
    return {
        "context_text": doc.get("context_text", ""),
        "metadata": doc.get("metadata", {}),
        "last_updated": doc.get("last_updated").isoformat() if doc.get("last_updated") else None
    }


@router.put("/api/global-context")
async def update_global_context(data: GlobalContextUpdate):
    mongo = get_mongo()
    
    await mongo.db["global_context"].update_one(
        {"_id": "current"},
        {"$set": {
            "context_text": data.context_text,
            "last_updated": datetime.now()
        }},
        upsert=True
    )
    
    brain = await get_brain()
    if brain and brain.hippocampus:
        await brain.hippocampus.update_global_context(data.context_text)
    
    return {"status": "updated"}


@router.get("/api/system-config")
async def get_system_config():
    mongo = get_mongo()
    doc = await mongo.db["system_config"].find_one({"_id": "config"})
    
    if not doc:
        return {
            "chat_model": SYSTEM_CONFIG.chat_model,
            "temperature": SYSTEM_CONFIG.temperature,
            "top_p": SYSTEM_CONFIG.top_p,
            "max_output_tokens": SYSTEM_CONFIG.max_output_tokens
        }
    
    return {
        "chat_model": doc.get("chat_model", SYSTEM_CONFIG.chat_model),
        "temperature": doc.get("temperature", 0.7),
        "top_p": doc.get("top_p", 0.95),
        "max_output_tokens": doc.get("max_output_tokens", 512)
    }


@router.put("/api/system-config")
async def update_system_config(data: SystemConfigUpdate):
    mongo = get_mongo()
    
    update_data = {}
    if data.chat_model is not None:
        update_data["chat_model"] = data.chat_model
    if data.temperature is not None:
        update_data["temperature"] = data.temperature
    if data.top_p is not None:
        update_data["top_p"] = data.top_p
    if data.max_output_tokens is not None:
        update_data["max_output_tokens"] = data.max_output_tokens
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    update_data["updated_at"] = datetime.now()
    
    await mongo.db["system_config"].update_one(
        {"_id": "config"},
        {"$set": update_data},
        upsert=True
    )
    
    return {"status": "updated"}


