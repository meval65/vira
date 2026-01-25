"""
Thalamus Module - Session and Context Management for Vira.

This module handles:
- Chat session management with MongoDB chat_logs
- Hybrid context retrieval (short-term + vector-based long-term)
- Proactive insight generation
- Weather context integration

Refactored to use MongoDB for session storage with automatic TTL expiry.
"""

import os
import json
import asyncio
import datetime
import hashlib
import httpx
import math
import random
import numpy as np
from collections import deque
from typing import List, Dict, Optional, Any, Deque
from dataclasses import dataclass, field
from enum import Enum

import PIL.Image
from google.genai import types

from src.brainstem import METEOSOURCE_API_KEY, NeuralEventBus
from src.db.mongo_client import get_mongo_client, MongoDBClient


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
    embedding: Optional[List[float]] = None


class Thalamus:
    """
    Session and context manager using MongoDB.
    
    Implements hybrid retrieval:
    - Short-term: Last 20 messages from chat_logs
    - Long-term: 3-5 semantically similar messages via NumPy vector search
    """
    
    MAX_SHORT_TERM: int = 20
    LONG_TERM_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.75
    INACTIVITY_THRESHOLD_HOURS: int = 72
    COOLDOWN_MINUTES: int = 120
    DEFAULT_LAT: float = -7.6398581
    DEFAULT_LON: float = 112.2395766
    WEATHER_CACHE_TTL: int = 1800

    INACTIVITY_PROMPTS = [
        "Hei, lu kemana aja? Udah lama gak ngobrol nih. Semua baik-baik aja kan?",
        "Sepi banget nih gak ada lu. Lagi sibuk apa sekarang?",
        "Woy, masih idup kan? 😂 Canda deng. Muncul dong!",
        "Kangen deh ngobrol sama lu. Lagi ngerjain apa?",
        "Eh, ada cerita seru apa nih akhir-akhir ini? Share dong!"
    ]
    
    KNOWLEDGE_PROMPTS = [
        "Gw masih belum kenal lu lebih jauh nih. Ceritain sesuatu tentang diri lu dong!",
        "Gw penasaran, hobi lu sebenernya apa sih selain yang biasa lu ceritain?",
        "Coba kasih tau gw satu fakta unik tentang diri lu yang jarang orang tau.",
        "Kalo lu bisa pergi ke mana aja sekarang, lu mau ke mana? Biar gw lebih tau selera lu.",
        "Apa sih mimpi terbesar lu saat ini? Gw pengen tau lebih banyak soal ambisi lu."
    ]

    def __init__(self, hippocampus):
        self._hippocampus = hippocampus
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

    async def initialize(self) -> None:
        """Initialize MongoDB connection for session storage."""
        self._mongo = get_mongo_client()
        # Connection already established by hippocampus
        await self._load_metadata()

    async def _load_metadata(self) -> None:
        """Load session metadata from MongoDB."""
        doc = await self._mongo.db["session_metadata"].find_one({"_id": "admin"})
        if doc:
            self._metadata = {
                "summary": doc.get("summary", ""),
                "memory_summary": doc.get("memory_summary", ""),
                "schedule_summary": doc.get("schedule_summary", ""),
                "last_analysis_count": doc.get("last_analysis_count", 0),
                "last_interaction": doc.get("last_interaction")
            }
            # Load insight cache
            for k, v in doc.get("insight_cache", {}).items():
                try:
                    self._insight_cache[k] = datetime.datetime.fromisoformat(v)
                except Exception:
                    pass

    async def _save_metadata(self) -> None:
        """Save session metadata to MongoDB."""
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

    async def get_session(self, limit: int = 40) -> List[SessionMessage]:
        """Get recent session messages from MongoDB."""
        cursor = self._mongo.chat_logs.find().sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        messages = []
        for doc in reversed(docs):  # Reverse to get chronological order
            messages.append(SessionMessage(
                role=doc.get("role", "user"),
                content=doc.get("content", ""),
                timestamp=doc.get("timestamp", datetime.datetime.now()),
                image_path=doc.get("image_path"),
                embedding=doc.get("embedding")
            ))
        return messages

    def get_history_for_model(self) -> List[types.Content]:
        """Get session history formatted for Gemini model (async wrapper needed)."""
        # This will be called from async context, return empty for sync access
        # The actual implementation should be called via async method
        return []

    async def get_history_for_model_async(self) -> List[types.Content]:
        """Get session history formatted for Gemini model."""
        messages = await self.get_session(limit=self.MAX_SHORT_TERM)
        history = []
        
        for msg in messages:
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
        image_path: Optional[str] = None,
        user_embedding: Optional[List[float]] = None,
        ai_embedding: Optional[List[float]] = None
    ) -> None:
        """Store new messages in MongoDB chat_logs."""
        now = datetime.datetime.now()
        
        # Store user message
        await self._mongo.chat_logs.insert_one({
            "role": "user",
            "content": user_text,
            "timestamp": now,
            "image_path": image_path,
            "embedding": user_embedding
        })
        
        # Store AI response
        await self._mongo.chat_logs.insert_one({
            "role": "model",
            "content": ai_response,
            "timestamp": now + datetime.timedelta(milliseconds=1),
            "embedding": ai_embedding
        })
        
        self._metadata["last_interaction"] = now.isoformat()
        await self._save_metadata()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors using NumPy."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        
        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)
        
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def get_hybrid_context(
        self,
        query_embedding: Optional[List[float]],
        short_term_limit: int = 20,
        long_term_limit: int = 5
    ) -> Dict[str, List[SessionMessage]]:
        """
        Retrieve hybrid context: short-term recent + long-term relevant.
        
        Returns:
            {
                "short_term": [last 20 messages],
                "long_term": [3-5 relevant older messages]
            }
        """
        await NeuralEventBus.set_activity("thalamus", "Building Hybrid Context")
        
        # Short-term: Last N messages
        short_term_cursor = self._mongo.chat_logs.find().sort("timestamp", -1).limit(short_term_limit)
        short_term_docs = await short_term_cursor.to_list(length=short_term_limit)
        
        short_term = []
        oldest_short_term = None
        for doc in reversed(short_term_docs):
            short_term.append(SessionMessage(
                role=doc.get("role", "user"),
                content=doc.get("content", ""),
                timestamp=doc.get("timestamp", datetime.datetime.now()),
                image_path=doc.get("image_path"),
                embedding=doc.get("embedding")
            ))
            if oldest_short_term is None or doc.get("timestamp", datetime.datetime.now()) < oldest_short_term:
                oldest_short_term = doc.get("timestamp")
        
        # Long-term: Semantically relevant older messages
        long_term = []
        if query_embedding and oldest_short_term:
            # Fetch older messages with embeddings
            long_term_cursor = self._mongo.chat_logs.find({
                "timestamp": {"$lt": oldest_short_term},
                "embedding": {"$exists": True, "$ne": None}
            }).limit(100)  # Limit to last 100 for performance
            
            long_term_docs = await long_term_cursor.to_list(length=100)
            
            # Score by similarity
            scored = []
            for doc in long_term_docs:
                emb = doc.get("embedding")
                if emb:
                    sim = self._cosine_similarity(query_embedding, emb)
                    if sim >= self.SIMILARITY_THRESHOLD:
                        scored.append((doc, sim))
            
            # Sort by similarity descending
            scored.sort(key=lambda x: x[1], reverse=True)
            
            for doc, _ in scored[:long_term_limit]:
                long_term.append(SessionMessage(
                    role=doc.get("role", "user"),
                    content=doc.get("content", ""),
                    timestamp=doc.get("timestamp", datetime.datetime.now()),
                    image_path=doc.get("image_path"),
                    embedding=doc.get("embedding")
                ))
        
        await NeuralEventBus.clear_activity("thalamus")
        
        return {
            "short_term": short_term,
            "long_term": long_term
        }

    def get_relevant_history(
        self,
        query_embedding: Optional[List[float]],
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[SessionMessage]:
        """
        Synchronous version for backward compatibility.
        Note: This returns empty list in sync context. Use get_hybrid_context instead.
        """
        return []

    async def get_relevant_history_async(
        self,
        query_embedding: Optional[List[float]],
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[SessionMessage]:
        """Retrieve semantically similar messages via NumPy vector search."""
        if not query_embedding:
            return []
        
        # Fetch messages with embeddings
        cursor = self._mongo.chat_logs.find({
            "embedding": {"$exists": True, "$ne": None}
        }).limit(200)
        
        docs = await cursor.to_list(length=200)
        
        scored = []
        for doc in docs:
            emb = doc.get("embedding")
            if emb:
                sim = self._cosine_similarity(query_embedding, emb)
                if sim >= min_similarity:
                    scored.append((doc, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, _ in scored[:top_k]:
            result.append(SessionMessage(
                role=doc.get("role", "user"),
                content=doc.get("content", ""),
                timestamp=doc.get("timestamp", datetime.datetime.now()),
                image_path=doc.get("image_path"),
                embedding=doc.get("embedding")
            ))
        
        return result

    async def clear_session(self) -> None:
        """Clear all chat logs (use with caution)."""
        await self._mongo.chat_logs.delete_many({})
        self._metadata = {
            "summary": "",
            "memory_summary": "",
            "schedule_summary": "",
            "last_analysis_count": 0,
            "last_interaction": None
        }
        await self._save_metadata()

    async def cleanup_session(self) -> None:
        """Cleanup hook - metadata is auto-saved after each update."""
        await self._save_metadata()

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
        user_metrics: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> str:
        """Build comprehensive context for response generation."""
        await NeuralEventBus.emit("thalamus", "prefrontal_cortex", "context_ready")
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

        # Add hybrid context (long-term relevant messages)
        if query_embedding:
            hybrid = await self.get_hybrid_context(query_embedding, short_term_limit=0, long_term_limit=5)
            if hybrid.get("long_term"):
                lt_lines = []
                for msg in hybrid["long_term"][:3]:
                    time_str = msg.timestamp.strftime("%d/%m %H:%M")
                    role = "User" if msg.role == "user" else "Vira"
                    content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    lt_lines.append(f"[{time_str}] {role}: {content}")
                sections.append(f"[RELEVANT PAST CONVERSATIONS]\n" + "\n".join(lt_lines))

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
                    message=random.choice(self.INACTIVITY_PROMPTS),
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
                message=random.choice(self.KNOWLEDGE_PROMPTS),
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
