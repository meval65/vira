import os
import json
import asyncio
import datetime
import hashlib
import httpx
from collections import deque
from typing import List, Dict, Optional, Any, Deque
from dataclasses import dataclass, field
from enum import Enum

import PIL.Image
from google.genai import types

from src.brainstem import METEOSOURCE_API_KEY


class InsightType(str, Enum):
    REMINDER = "reminder"
    FOLLOW_UP = "follow_up"
    PATTERN = "pattern"
    ANNIVERSARY = "anniversary"
    INACTIVITY = "inactivity"
    KNOWLEDGE = "knowledge"
    WELLNESS = "wellness"


class InsightPriority(int, Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ProactiveInsight:
    insight_type: InsightType
    priority: InsightPriority
    message: str
    context: Dict = field(default_factory=dict)
    trigger_time: Optional[datetime.datetime] = None
    expires_at: Optional[datetime.datetime] = None
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    delivered: bool = False


@dataclass
class SessionMessage:
    role: str
    content: str
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    image_path: Optional[str] = None


class Thalamus:
    MAX_HISTORY: int = 40
    INACTIVITY_THRESHOLD_HOURS: int = 72
    COOLDOWN_MINUTES: int = 120
    SESSION_DIR: str = "storage/sessions"
    DEFAULT_LAT: float = -7.6398581
    DEFAULT_LON: float = 112.2395766
    WEATHER_CACHE_TTL: int = 1800

    def __init__(self, hippocampus):
        self._hippocampus = hippocampus
        self._session: Deque[SessionMessage] = deque(maxlen=self.MAX_HISTORY)
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

    async def initialize(self) -> None:
        os.makedirs(self.SESSION_DIR, exist_ok=True)
        await self._load_session()

    async def _load_session(self) -> None:
        session_file = os.path.join(self.SESSION_DIR, "admin_session.json")
        if os.path.exists(session_file):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for msg in data.get("history", []):
                        self._session.append(SessionMessage(
                            role=msg.get("role", "user"),
                            content=msg.get("content", ""),
                            timestamp=datetime.datetime.fromisoformat(msg.get("timestamp", datetime.datetime.now().isoformat())),
                            image_path=msg.get("image_path")
                        ))
                    self._metadata = data.get("metadata", self._metadata)
            except Exception:
                pass

    async def _save_session(self) -> None:
        session_file = os.path.join(self.SESSION_DIR, "admin_session.json")
        try:
            data = {
                "history": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp.isoformat(),
                        "image_path": msg.image_path
                    }
                    for msg in self._session
                ],
                "metadata": self._metadata
            }
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_session(self) -> List[SessionMessage]:
        return list(self._session)

    def get_history_for_model(self) -> List[types.Content]:
        history = []
        for msg in self._session:
            parts = []
            if msg.content:
                parts.append(types.Part.from_text(text=msg.content))
            if msg.image_path and os.path.exists(msg.image_path):
                try:
                    img = PIL.Image.open(msg.image_path)
                    parts.append(types.Part.from_image(image=img))
                except Exception:
                    pass
            if parts:
                history.append(types.Content(role=msg.role, parts=parts))
        return history

    async def update_session(
        self,
        user_text: str,
        ai_response: str,
        image_path: Optional[str] = None
    ) -> None:
        self._session.append(SessionMessage(
            role="user",
            content=user_text,
            image_path=image_path
        ))
        self._session.append(SessionMessage(
            role="model",
            content=ai_response
        ))
        self._metadata["last_interaction"] = datetime.datetime.now().isoformat()
        await self._save_session()

    def clear_session(self) -> None:
        self._session.clear()
        self._metadata = {
            "summary": "",
            "memory_summary": "",
            "schedule_summary": "",
            "last_analysis_count": 0,
            "last_interaction": None
        }

    async def cleanup_session(self) -> None:
        await self._save_session()

    def get_summary(self) -> str:
        return self._metadata.get("summary", "")

    def get_memory_summary(self) -> str:
        return self._metadata.get("memory_summary", "")

    def update_summary(self, summary: str) -> None:
        self._metadata["summary"] = summary

    def update_memory_summary(self, summary: str) -> None:
        self._metadata["memory_summary"] = summary

    def update_schedule_summary(self, summary: str) -> None:
        self._metadata["schedule_summary"] = summary

    def should_run_memory_analysis(self, current_count: int) -> bool:
        last_count = self._metadata.get("last_analysis_count", 0)
        return current_count >= last_count + 5

    def mark_memory_analysis_done(self, count: int) -> None:
        self._metadata["last_analysis_count"] = count

    async def build_context(
        self,
        relevant_memories: List[Dict],
        schedule_context: Optional[str] = None,
        intent_data: Optional[Dict] = None,
        user_metrics: Optional[Dict] = None
    ) -> str:
        sections = []

        now = datetime.datetime.now()
        sections.append(f"[SYSTEM TIME]\n{now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})")

        weather = await self._get_weather()
        if weather:
            sections.append(f"[WEATHER]\n{weather}")

        profile = self._hippocampus.admin_profile
        if profile.telegram_name or profile.additional_info:
            profile_parts = []
            if profile.telegram_name:
                profile_parts.append(f"Name: {profile.telegram_name}")
            if profile.additional_info:
                profile_parts.append(f"Info: {profile.additional_info}")
            sections.append(f"[ADMIN PROFILE]\n" + "\n".join(profile_parts))

        last_interaction = self._metadata.get("last_interaction")
        if last_interaction:
            try:
                last_dt = datetime.datetime.fromisoformat(last_interaction)
                gap = now - last_dt
                if gap.total_seconds() > 3600:
                    hours = int(gap.total_seconds() / 3600)
                    sections.append(f"[TIME GAP]\nLast interaction: {hours} hours ago")
            except Exception:
                pass

        if schedule_context:
            sections.append(f"[SCHEDULES]\n{schedule_context}")

        if relevant_memories:
            mem_lines = []
            for m in relevant_memories[:5]:
                summary = m.get("summary", "")[:100]
                mem_type = m.get("memory_type", "general")
                mem_lines.append(f"- [{mem_type}] {summary}")
            sections.append(f"[RELEVANT MEMORIES]\n" + "\n".join(mem_lines))

        if intent_data:
            intent_parts = []
            if intent_data.get("intent_type"):
                intent_parts.append(f"Intent: {intent_data['intent_type']}")
            if intent_data.get("entities"):
                intent_parts.append(f"Entities: {', '.join(intent_data['entities'][:3])}")
            if intent_data.get("sentiment"):
                intent_parts.append(f"Sentiment: {intent_data['sentiment']}")
            if intent_parts:
                sections.append(f"[INTENT ANALYSIS]\n" + "\n".join(intent_parts))

        summary = self._metadata.get("summary", "")
        if summary:
            sections.append(f"[CONVERSATION CONTEXT]\n{summary[:300]}")

        return "\n\n".join(sections)

    async def _get_weather(self) -> Optional[str]:
        if self._is_weather_cached():
            return self._weather_cache.get("formatted")

        if not METEOSOURCE_API_KEY:
            return None

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"https://www.meteosource.com/api/v1/free/point"
                params = {
                    "lat": self.DEFAULT_LAT,
                    "lon": self.DEFAULT_LON,
                    "sections": "current",
                    "key": METEOSOURCE_API_KEY
                }
                response = await client.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    current = data.get("current", {})
                    temp = current.get("temperature", "?")
                    summary = current.get("summary", "Unknown")
                    formatted = f"{summary}, {temp}°C"
                    self._weather_cache = {"formatted": formatted, "raw": data}
                    self._weather_timestamp = datetime.datetime.now()
                    return formatted
        except Exception:
            pass
        return None

    def _is_weather_cached(self) -> bool:
        if not self._weather_timestamp:
            return False
        age = (datetime.datetime.now() - self._weather_timestamp).total_seconds()
        return age < self.WEATHER_CACHE_TTL

    async def check_proactive_triggers(self) -> Optional[str]:
        insights = await self._gather_insights()
        if not insights:
            return None

        best = max(insights, key=lambda x: x.priority.value)

        if not self._can_send_insight(best.insight_type):
            return None

        self._mark_insight_sent(best.insight_type)
        return best.message

    async def _gather_insights(self) -> List[ProactiveInsight]:
        insights = []

        inactivity = await self._check_inactivity()
        if inactivity:
            insights.append(inactivity)

        reminders = await self._check_scheduled_reminders()
        insights.extend(reminders)

        knowledge = await self._check_knowledge_gaps()
        if knowledge:
            insights.append(knowledge)

        return insights

    async def _check_inactivity(self) -> Optional[ProactiveInsight]:
        last_str = self._metadata.get("last_interaction")
        if not last_str:
            return None

        try:
            last_dt = datetime.datetime.fromisoformat(last_str)
            hours_since = (datetime.datetime.now() - last_dt).total_seconds() / 3600

            if hours_since >= self.INACTIVITY_THRESHOLD_HOURS:
                return ProactiveInsight(
                    insight_type=InsightType.INACTIVITY,
                    priority=InsightPriority.MEDIUM,
                    message="Hei, lu kemana aja? Udah lama gak ngobrol nih. Semua baik-baik aja kan?",
                    context={"hours_inactive": hours_since}
                )
        except Exception:
            pass

        return None

    async def _check_scheduled_reminders(self) -> List[ProactiveInsight]:
        insights = []
        pending = await self._hippocampus.get_pending_schedules(limit=5)

        for schedule in pending:
            scheduled_at = schedule.get("scheduled_at")
            if isinstance(scheduled_at, str):
                try:
                    scheduled_at = datetime.datetime.fromisoformat(scheduled_at)
                except Exception:
                    continue

            if scheduled_at <= datetime.datetime.now():
                insights.append(ProactiveInsight(
                    insight_type=InsightType.REMINDER,
                    priority=InsightPriority.HIGH,
                    message=f"⏰ Pengingat: {schedule.get('context', 'Ada yang perlu dilakukan')}",
                    context={"schedule_id": schedule.get("id")}
                ))

        return insights

    async def _check_knowledge_gaps(self) -> Optional[ProactiveInsight]:
        stats = await self._hippocampus.get_memory_stats()
        if stats.get("active", 0) < 5:
            return ProactiveInsight(
                insight_type=InsightType.KNOWLEDGE,
                priority=InsightPriority.LOW,
                message="Gw masih belum kenal lu lebih jauh nih. Ceritain sesuatu tentang diri lu dong!",
                context={"memory_count": stats.get("active", 0)}
            )
        return None

    def _can_send_insight(self, insight_type: InsightType) -> bool:
        key = insight_type.value
        last_sent = self._insight_cache.get(key)
        if not last_sent:
            return True

        minutes_since = (datetime.datetime.now() - last_sent).total_seconds() / 60
        return minutes_since >= self.COOLDOWN_MINUTES

    def _mark_insight_sent(self, insight_type: InsightType) -> None:
        self._insight_cache[insight_type.value] = datetime.datetime.now()

    def get_last_interaction(self) -> Optional[datetime.datetime]:
        last_str = self._metadata.get("last_interaction")
        if last_str:
            try:
                return datetime.datetime.fromisoformat(last_str)
            except Exception:
                pass
        return None

    async def should_initiate_contact(self) -> bool:
        last = self.get_last_interaction()
        if not last:
            return False

        hours_since = (datetime.datetime.now() - last).total_seconds() / 3600
        if hours_since >= self.INACTIVITY_THRESHOLD_HOURS:
            return self._can_send_insight(InsightType.INACTIVITY)

        pending = await self._hippocampus.get_pending_schedules(limit=1)
        if pending:
            return True

        return False

    def format_time_gap(self, last_time: Optional[datetime.datetime]) -> str:
        if not last_time:
            return "First interaction"

        gap = datetime.datetime.now() - last_time
        hours = gap.total_seconds() / 3600

        if hours < 1:
            return "Just now"
        elif hours < 24:
            return f"{int(hours)} hours ago"
        else:
            days = int(hours / 24)
            return f"{days} days ago"
