from datetime import datetime
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List, Union
from src.brain.db.mongo_client import get_mongo_client
from src.brain.brainstem import get_brain

router = APIRouter(prefix="/api/personas", tags=["personas"])


def get_mongo():
    return get_mongo_client()


def parse_id(id_str: str) -> Union[ObjectId, str]:
    try:
        return ObjectId(id_str)
    except InvalidId:
        return id_str


def build_id_query(id_str: str) -> dict:
    try:
        return {"_id": ObjectId(id_str)}
    except InvalidId:
        return {"$or": [{"_id": id_str}, {"id": id_str}]}


class PersonaCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    instruction: str
    temperature: Optional[float] = 0.7
    traits: Optional[Dict] = {}
    voice_tone: Optional[str] = "friendly"


class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    instruction: Optional[str] = None
    temperature: Optional[float] = None
    traits: Optional[Dict] = None
    voice_tone: Optional[str] = None
    is_active: Optional[bool] = None


@router.get("")
async def get_personas():
    mongo = get_mongo()
    cursor = mongo.personas.find()
    personas = await cursor.to_list(length=100)
    
    result = []
    for p in personas:
        result.append({
            "id": str(p["_id"]),
            "name": p.get("name", ""),
            "description": p.get("description", ""),
            "instruction": p.get("instruction", ""),
            "temperature": p.get("temperature", 0.7),
            "traits": p.get("traits", {}),
            "voice_tone": p.get("voice_tone", "friendly"),
            "is_active": p.get("is_active", False),
            "calibration": p.get("calibration", {}),
            "created_at": p.get("created_at").isoformat() if p.get("created_at") else None
        })
    return result


@router.get("/active")
async def get_active_persona():
    mongo = get_mongo()
    doc = await mongo.personas.find_one({"is_active": True})
    
    if not doc:
        return {"status": "none", "persona": None}
    
    return {
        "status": "active",
        "persona": {
            "id": str(doc["_id"]),
            "name": doc.get("name", ""),
            "description": doc.get("description", ""),
            "instruction": doc.get("instruction", ""),
            "temperature": doc.get("temperature", 0.7),
            "traits": doc.get("traits", {}),
            "voice_tone": doc.get("voice_tone", "friendly"),
            "is_active": True,
            "calibration": doc.get("calibration", {})
        }
    }


@router.post("")
async def create_persona(data: PersonaCreate):
    mongo = get_mongo()
    
    existing = await mongo.personas.find_one({"name": data.name})
    if existing:
        raise HTTPException(status_code=400, detail=f"Persona with name '{data.name}' already exists")
    
    persona_doc = {
        "name": data.name,
        "description": data.description,
        "instruction": data.instruction,
        "temperature": data.temperature,
        "traits": data.traits,
        "voice_tone": data.voice_tone,
        "is_active": False,
        "calibration": {"calibration_status": False},
        "created_at": datetime.now()
    }
    
    result = await mongo.personas.insert_one(persona_doc)
    persona_doc["id"] = str(result.inserted_id)
    
    return persona_doc


@router.put("/{persona_id}")
async def update_persona(persona_id: str, data: PersonaUpdate):
    mongo = get_mongo()
    query = build_id_query(persona_id)
    
    if data.name is not None:
        existing = await mongo.personas.find_one({"name": data.name})
        if existing and str(existing["_id"]) != persona_id:
            raise HTTPException(status_code=400, detail=f"Persona with name '{data.name}' already exists")
    
    update_data = {}
    if data.name is not None:
        update_data["name"] = data.name
    if data.description is not None:
        update_data["description"] = data.description
    if data.instruction is not None:
        update_data["instruction"] = data.instruction
        update_data["calibration.calibration_status"] = False
    if data.temperature is not None:
        update_data["temperature"] = data.temperature
    if data.traits is not None:
        update_data["traits"] = data.traits
    if data.voice_tone is not None:
        update_data["voice_tone"] = data.voice_tone
    if data.is_active is not None:
        update_data["is_active"] = data.is_active
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    update_data["updated_at"] = datetime.now()
    
    result = await mongo.personas.update_one(query, {"$set": update_data})
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    return {"status": "updated", "id": persona_id}


@router.delete("/{persona_id}")
async def delete_persona(persona_id: str):
    mongo = get_mongo()
    query = build_id_query(persona_id)
    
    result = await mongo.personas.delete_one(query)
    
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    return {"status": "deleted"}


@router.post("/{persona_id}/activate")
async def activate_persona(persona_id: str):
    mongo = get_mongo()
    query = build_id_query(persona_id)
    
    await mongo.personas.update_many({}, {"$set": {"is_active": False}})
    
    result = await mongo.personas.update_one(query, {"$set": {"is_active": True}})
    
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Persona not found")
    
    doc = await mongo.personas.find_one(query)
    
    brain = await get_brain()
    if brain and brain.prefrontal_cortex:
        try:
            await brain.prefrontal_cortex.switch_persona(persona_id)
        except Exception:
            pass
    
    return {
        "status": "activated",
        "persona": {
            "id": str(doc["_id"]),
            "name": doc.get("name", ""),
            "is_active": True
        }
    }
