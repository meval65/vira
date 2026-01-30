import uuid
import asyncio
from datetime import datetime
from typing import Dict, Optional, List, Union, Any
from bson import ObjectId
from bson.errors import InvalidId
from src.brain.infrastructure.neural_event_bus import NeuralEventBus


class PersonaManager:
    def __init__(self, mongo_client):
        self._mongo = mongo_client
        self._persona_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl: int = 300

    def _build_id_query(self, id_str: str) -> Dict[str, Any]:
        """Build a query that matches either ObjectId or string ID."""
        try:
            return {"_id": ObjectId(id_str)}
        except InvalidId:
            return {"_id": id_str}

    async def create_persona(
        self,
        name: str,
        description: str,
        traits: Dict[str, float],
        voice_tone: str,
        identity_anchor: Optional[str]
    ) -> str:
        # We stick to UUID for internal creation to avoid confusion, 
        # but API might create ObjectIds.
        persona_id = str(uuid.uuid4())
        await self._mongo.personas.insert_one({
            "_id": persona_id,
            "name": name,
            "description": description,
            "traits": traits,
            "voice_tone": voice_tone,
            "is_active": False,
            "calibration": {
                "emotional_inertia": 0.7,
                "base_arousal": 0.0,
                "base_valence": 0.0,
                "calibration_status": False,
                "identity_anchor": identity_anchor or description
            },
            "created_at": datetime.now(),
            "last_modified": datetime.now()
        })
        self._persona_cache = None
        return persona_id

    async def update_persona(
        self,
        persona_id: str,
        name: Optional[str],
        description: Optional[str],
        traits: Optional[Dict[str, float]],
        voice_tone: Optional[str]
    ) -> bool:
        update_fields = {"last_modified": datetime.now()}
        
        if name is not None:
            update_fields["name"] = name
        if description is not None:
            update_fields["description"] = description
        if traits is not None:
            update_fields["traits"] = traits
        if voice_tone is not None:
            update_fields["voice_tone"] = voice_tone
        
        query = self._build_id_query(persona_id)
        result = await self._mongo.personas.update_one(
            query,
            {"$set": update_fields}
        )
        
        if result.modified_count > 0:
            self._persona_cache = None
        
        return result.modified_count > 0

    async def delete_persona(self, persona_id: str) -> bool:
        query = self._build_id_query(persona_id)
        persona = await self._mongo.personas.find_one(query)
        if not persona:
            return False
        
        if persona.get("is_active"):
            await self._ensure_default_persona()
        
        result = await self._mongo.personas.delete_one(query)
        if result.deleted_count > 0:
            self._persona_cache = None
        return result.deleted_count > 0

    async def list_personas(self) -> List[Dict]:
        cursor = self._mongo.personas.find({})
        
        personas = []
        async for doc in cursor:
            personas.append({
                "id": str(doc["_id"]),
                "name": doc.get("name"),
                "instruction": doc.get("instruction"),
                "temperature": doc.get("temperature", 0.7),
                "description": doc.get("description"),
                "is_active": doc.get("is_active", False)
            })
        return personas

    async def get_active_persona(self) -> Optional[Dict]:
        now = datetime.now()
        if (self._persona_cache and self._cache_timestamp and 
            (now - self._cache_timestamp).total_seconds() < self._cache_ttl):
            return self._persona_cache
        
        doc = await self._mongo.personas.find_one({"is_active": True})
        if doc:
            calibration = doc.get("calibration", {})
            self._persona_cache = {
                "id": str(doc["_id"]),
                "name": doc.get("name", ""),
                "description": doc.get("description", ""),
                "traits": doc.get("traits", {}),
                "voice_tone": doc.get("voice_tone", "friendly"),
                "calibration": {
                    "emotional_inertia": calibration.get("emotional_inertia", 0.7),
                    "base_arousal": calibration.get("base_arousal", 0.0),
                    "base_valence": calibration.get("base_valence", 0.0),
                    "calibration_status": calibration.get("calibration_status", False),
                    "identity_anchor": calibration.get("identity_anchor", doc.get("description", ""))
                }
            }
            self._cache_timestamp = now
            return self._persona_cache
        return None

    async def _ensure_default_persona(self) -> None:
        existing = await self._mongo.personas.find_one({"is_active": True})
        if existing:
            return
        
        default = await self._mongo.personas.find_one({"name": "Default"})
        if not default:
            default_id = str(uuid.uuid4())
            await self._mongo.personas.insert_one({
                "_id": default_id,
                "name": "Default",
                "description": "Balanced and helpful assistant",
                "traits": {
                    "formality": 0.5,
                    "enthusiasm": 0.6,
                    "verbosity": 0.5,
                    "creativity": 0.6
                },
                "voice_tone": "friendly",
                "is_active": True,
                "calibration": {
                    "emotional_inertia": 0.7,
                    "base_arousal": 0.0,
                    "base_valence": 0.1,
                    "calibration_status": True,
                    "identity_anchor": "Balanced and helpful assistant with friendly demeanor"
                },
                "created_at": datetime.now(),
                "last_modified": datetime.now()
            })
        else:
            await self._mongo.personas.update_one(
                {"_id": default["_id"]},
                {"$set": {"is_active": True}}
            )
        self._persona_cache = None

    async def switch_persona(self, persona_id: str) -> bool:
        query = self._build_id_query(persona_id)
        
        # Verify persona exists first
        target = await self._mongo.personas.find_one(query)
        if not target:
            return False
            
        await self._mongo.personas.update_many({}, {"$set": {"is_active": False}})
        result = await self._mongo.personas.update_one(
            query,
            {"$set": {"is_active": True}}
        )
        
        if result.modified_count > 0:
            self._persona_cache = None
            persona = await self.get_active_persona()
            await NeuralEventBus.emit("hippocampus", "prefrontal_cortex", "persona_changed", payload={
                "persona_id": persona_id,
                "name": persona.get("name") if persona else "Unknown"
            })
        
        return result.modified_count > 0

    async def get_persona_calibration(self, persona_id: str) -> Optional[Dict]:
        query = self._build_id_query(persona_id)
        doc = await self._mongo.personas.find_one(query)
        if not doc:
            return None
        
        calibration = doc.get("calibration", {})
        return {
            "emotional_inertia": calibration.get("emotional_inertia", 0.7),
            "base_arousal": calibration.get("base_arousal", 0.0),
            "base_valence": calibration.get("base_valence", 0.0),
            "calibration_status": calibration.get("calibration_status", False),
            "identity_anchor": calibration.get("identity_anchor", doc.get("description", ""))
        }

    async def update_persona_calibration(
        self,
        persona_id: str,
        emotional_inertia: Optional[float],
        base_arousal: Optional[float],
        base_valence: Optional[float],
        calibration_status: Optional[bool],
        identity_anchor: Optional[str]
    ) -> bool:
        update_fields = {"last_modified": datetime.now()}
        
        if emotional_inertia is not None:
            update_fields["calibration.emotional_inertia"] = max(0.0, min(1.0, emotional_inertia))
        if base_arousal is not None:
            update_fields["calibration.base_arousal"] = max(-1.0, min(1.0, base_arousal))
        if base_valence is not None:
            update_fields["calibration.base_valence"] = max(-1.0, min(1.0, base_valence))
        if calibration_status is not None:
            update_fields["calibration.calibration_status"] = calibration_status
        if identity_anchor is not None:
            update_fields["calibration.identity_anchor"] = identity_anchor
        
        query = self._build_id_query(persona_id)
        result = await self._mongo.personas.update_one(
            query,
            {"$set": update_fields}
        )
        
        if result.modified_count > 0:
            self._persona_cache = None
        
        return result.modified_count > 0
