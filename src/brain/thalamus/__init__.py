import uuid
import datetime
from typing import List, Dict, Optional, Any

from src.brain.db.mongo_client import get_mongo_client, MongoDBClient

from .types import (
    InsightType, InsightPriority, ContextPriority,
    ProactiveInsight, SessionMessage
)
from .session import SessionManagerMixin
from .context import ContextBuilderMixin
from .insights import InsightsManagerMixin
from .weather import WeatherMixin


class Thalamus(SessionManagerMixin, ContextBuilderMixin, InsightsManagerMixin, WeatherMixin):
    LONG_TERM_TOP_K: int = 5

    def __init__(self):
        self._brain = None
        self._session_id = str(uuid.uuid4())
        self._context_window: List[Dict] = []
        self._max_context_length = 15
        self._system_prompt_cache = None
        self._last_weather_update = None
        self._weather_cache = None
        
        self._mongo: Optional[MongoDBClient] = None
        self._metadata: Dict[str, Any] = {
            "summary": "",
            "memory_summary": "",
            "schedule_summary": "",
            "last_analysis_count": 0,
            "last_interaction": None
        }
        self._insight_cache: Dict[str, datetime.datetime] = {}
        self._weather_cache: Dict[str, Any] = {}
        self._weather_timestamp: Optional[datetime.datetime] = None
        self._context_cache: Dict[str, Any] = {}
        self._global_context_cache: Optional[str] = None
        self._global_context_timestamp: Optional[datetime.datetime] = None

    def bind_brain(self, brain) -> None:
        self._brain = brain

    @property
    def hippocampus(self):
        return self._brain.hippocampus if self._brain else None

    @property
    def amygdala(self):
        return self._brain.amygdala if self._brain else None

    async def initialize(self) -> None:
        self._mongo = get_mongo_client()
        await self._load_metadata()

    async def _load_metadata(self) -> None:
        doc = await self._mongo.db["session_metadata"].find_one({"_id": "admin"})
        if doc:
            self._metadata = {
                "summary": doc.get("summary", ""),
                "memory_summary": doc.get("memory_summary", ""),
                "schedule_summary": doc.get("schedule_summary", ""),
                "last_analysis_count": doc.get("last_analysis_count", 0),
                "last_interaction": doc.get("last_interaction")
            }
            for k, v in doc.get("insight_cache", {}).items():
                try:
                    self._insight_cache[k] = datetime.datetime.fromisoformat(v)
                except Exception:
                    pass

    async def _save_metadata(self) -> None:
        await self._mongo.db["session_metadata"].update_one(
            {"_id": "admin"},
            {"$set": {
                "summary": self._metadata.get("summary", ""),
                "memory_summary": self._metadata.get("memory_summary", ""),
                "schedule_summary": self._metadata.get("schedule_summary", ""),
                "last_analysis_count": self._metadata.get("last_analysis_count", 0),
                "last_interaction": self._metadata.get("last_interaction"),
                "insight_cache": {k: v.isoformat() for k, v in self._insight_cache.items()}
            }},
            upsert=True
        )


__all__ = [
    "Thalamus",
    "InsightType",
    "InsightPriority", 
    "ContextPriority",
    "ProactiveInsight",
    "SessionMessage"
]
