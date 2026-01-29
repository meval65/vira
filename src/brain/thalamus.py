import os
import json
import uuid
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
import tiktoken

import PIL.Image
from google.genai import types

from src.brain.brainstem import METEOSOURCE_API_KEY, NeuralEventBus
from src.brain.db.mongo_client import get_mongo_client, MongoDBClient


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


class ContextPriority(int, Enum):
    GLOBAL_CONTEXT = 100
    PERSONA_CONTEXT = 90
    SCHEDULE_CONTEXT = 80
    RELEVANT_MEMORIES = 70
    RELEVANT_HISTORY = 60
    INTENT_ANALYSIS = 50
    TIME_CONTEXT = 40
    WEATHER_CONTEXT = 30


class Thalamus:
    MAX_SHORT_TERM: int = 20
    LONG_TERM_TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.75
    INACTIVITY_THRESHOLD_HOURS: int = 72
    COOLDOWN_MINUTES: int = 120
    DEFAULT_LAT: float = -7.6398581
    DEFAULT_LON: float = 112.2395766
    WEATHER_CACHE_TTL: int = 1800
    
    MAX_CONTEXT_TOKENS: int = 1500

    def __init__(self):
        self._brain = None
        self._session_id = str(uuid.uuid4())
        self._context_window: List[Dict] = []
        self._max_context_length = 15  # Default messages to keep
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

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

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
        return []

    async def get_history_for_model_async(self) -> List[types.Content]:
        """Get session history formatted for model."""
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
            long_term_cursor = self._mongo.chat_logs.find({
                "timestamp": {"$lt": oldest_short_term},
                "embedding": {"$exists": True, "$ne": None}
            }).limit(100)
            
            long_term_docs = await long_term_cursor.to_list(length=100)
            
            scored = []
            for doc in long_term_docs:
                emb = doc.get("embedding")
                if emb:
                    sim = self._cosine_similarity(query_embedding, emb)
                    if sim >= self.SIMILARITY_THRESHOLD:
                        scored.append((doc, sim))
            
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
        """Synchronous version - returns empty list. Use async version."""
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

    async def _get_global_context(self) -> Optional[str]:
        if (self._global_context_cache and self._global_context_timestamp and
            (datetime.datetime.now() - self._global_context_timestamp).total_seconds() < 300):
            return self._global_context_cache

        global_ctx = await self.hippocampus.get_global_context()
        if global_ctx:
            self._global_context_cache = global_ctx
            self._global_context_timestamp = datetime.datetime.now()

        return global_ctx

    async def _get_system_config(self) -> Dict[str, Any]:
        doc = await self._mongo.db["system_config"].find_one({"_id": "config"})
        if doc:
            return {
                "chat_model": doc.get("chat_model"),
                "temperature": doc.get("temperature", 0.7),
                "top_p": doc.get("top_p", 0.95),
                "max_output_tokens": doc.get("max_output_tokens", 512)
            }
        return {"temperature": 0.7, "top_p": 0.95, "max_output_tokens": 512}

    async def _get_active_persona(self) -> Optional[Dict[str, Any]]:
        try:
            doc = await self._mongo.personas.find_one({"is_active": True})
            if doc:
                return {
                    "id": str(doc["_id"]),
                    "name": doc.get("name", "Default"),
                    "instruction": doc.get("instruction", ""),
                    "temperature": doc.get("temperature", 0.7),
                    "description": doc.get("description", "")
                }
        except Exception:
            pass
        return None

    async def build_context(
        self,
        relevant_memories: List[Dict],
        schedule_context: Optional[str] = None,
        intent_data: Optional[Dict] = None,
        user_metrics: Optional[Dict] = None,
        query_embedding: Optional[List[float]] = None
    ) -> str:
        await NeuralEventBus.emit("thalamus", "prefrontal_cortex", "context_ready")
        
        # Collect context sections with priorities
        context_sections: List[Dict[str, Any]] = []
        
        # 1. GLOBAL CONTEXT (Highest Priority)
        global_ctx = await self._get_global_context()
        if global_ctx:
            context_sections.append({
                "priority": ContextPriority.GLOBAL_CONTEXT.value,
                "header": "[GLOBAL CONTEXT - USER PROFILE]",
                "content": global_ctx
            })
        
        # 2. ACTIVE PERSONA (Hot-reload from DB)
        persona = await self._get_active_persona()
        if persona and persona.get("instruction"):
            context_sections.append({
                "priority": ContextPriority.PERSONA_CONTEXT.value,
                "header": f"[ACTIVE PERSONA: {persona.get('name', 'Default')}]",
                "content": persona.get("instruction", "")
            })
        
        # 3. SYSTEM TIME
        now = datetime.datetime.now()
        context_sections.append({
            "priority": ContextPriority.TIME_CONTEXT.value,
            "header": "[SYSTEM TIME]",
            "content": f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"
        })
        
        # 4. WEATHER
        weather = await self._get_weather()
        if weather:
            context_sections.append({
                "priority": ContextPriority.WEATHER_CONTEXT.value,
                "header": "[WEATHER]",
                "content": weather
            })
        
        # 4. ADMIN PROFILE
        profile = self.hippocampus.admin_profile
        if profile.telegram_name or profile.additional_info:
            profile_parts = []
            if profile.telegram_name:
                profile_parts.append(f"Name: {profile.telegram_name}")
            if profile.additional_info:
                profile_parts.append(f"Info: {profile.additional_info}")
            context_sections.append({
                "priority": ContextPriority.PERSONA_CONTEXT.value - 5,  # Slightly lower than persona
                "header": "[ADMIN PROFILE]",
                "content": "\n".join(profile_parts)
            })
        
        # 5. TIME GAP (if significant)
        last_interaction = self._metadata.get("last_interaction")
        if last_interaction:
            try:
                last_dt = datetime.datetime.fromisoformat(last_interaction)
                gap = now - last_dt
                if gap.total_seconds() > 3600:
                    hours = int(gap.total_seconds() / 3600)
                    context_sections.append({
                        "priority": ContextPriority.TIME_CONTEXT.value - 5,
                        "header": "[TIME GAP]",
                        "content": f"Last interaction: {hours} hours ago"
                    })
            except Exception:
                pass
        
        # 6. SCHEDULES
        if schedule_context:
            context_sections.append({
                "priority": ContextPriority.SCHEDULE_CONTEXT.value,
                "header": "[UPCOMING SCHEDULES]",
                "content": schedule_context
            })
        
        # 7. RELEVANT MEMORIES
        if relevant_memories:
            mem_lines = []
            for m in relevant_memories[:5]:
                summary = m.get("summary", "")[:100]
                mem_type = m.get("memory_type", "general")
                mem_lines.append(f"- [{mem_type}] {summary}")
            context_sections.append({
                "priority": ContextPriority.RELEVANT_MEMORIES.value,
                "header": "[RELEVANT MEMORIES]",
                "content": "\n".join(mem_lines)
            })
        
        # 8. INTENT ANALYSIS
        if intent_data:
            intent_parts = []
            if intent_data.get("intent_type"):
                intent_parts.append(f"Intent: {intent_data['intent_type']}")
            if intent_data.get("entities"):
                intent_parts.append(f"Entities: {', '.join(intent_data['entities'][:3])}")
            if intent_data.get("sentiment"):
                intent_parts.append(f"Sentiment: {intent_data['sentiment']}")
            if intent_parts:
                context_sections.append({
                    "priority": ContextPriority.INTENT_ANALYSIS.value,
                    "header": "[INTENT ANALYSIS]",
                    "content": "\n".join(intent_parts)
                })
        
        # 9. RELEVANT PAST CONVERSATIONS (Hybrid context)
        if query_embedding:
            hybrid = await self.get_hybrid_context(query_embedding, short_term_limit=0, long_term_limit=5)
            if hybrid.get("long_term"):
                lt_lines = []
                for msg in hybrid["long_term"][:3]:
                    time_str = msg.timestamp.strftime("%d/%m %H:%M")
                    role = "User" if msg.role == "user" else "AI"
                    content = msg.content[:150] + "..." if len(msg.content) > 150 else msg.content
                    lt_lines.append(f"[{time_str}] {role}: {content}")
                context_sections.append({
                    "priority": ContextPriority.RELEVANT_HISTORY.value,
                    "header": "[RELEVANT PAST CONVERSATIONS]",
                    "content": "\n".join(lt_lines)
                })
        
        # 10. CONVERSATION SUMMARY
        summary = self._metadata.get("summary", "")
        if summary:
            context_sections.append({
                "priority": ContextPriority.INTENT_ANALYSIS.value - 5,
                "header": "[CONVERSATION CONTEXT]",
                "content": summary[:300]
            })
        
        context_sections.sort(key=lambda x: x["priority"], reverse=True)
        
        final_sections = []
        estimated_tokens = 0
        
        for section in context_sections:
            section_text = f"{section['header']}\n{section['content']}"
            section_tokens = self.estimate_tokens(section_text)
            
            if estimated_tokens + section_tokens <= self.MAX_CONTEXT_TOKENS:
                final_sections.append(section_text)
                estimated_tokens += section_tokens
            else:
                remaining_tokens = self.MAX_CONTEXT_TOKENS - estimated_tokens
                if remaining_tokens > 30:
                    remaining_chars = remaining_tokens * 3
                    truncated = section_text[:remaining_chars] + "..."
                    final_sections.append(truncated)
                break
        
        return "\n\n".join(final_sections)

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

        if best.insight_type == InsightType.FOLLOW_UP and "schedule_id" in best.context:
            self._mark_insight_sent_with_id(f"followup_{best.context['schedule_id']}")

        await self._mongo.chat_logs.insert_one({
            "role": "model",
            "content": best.message,
            "timestamp": datetime.datetime.now(),
            "proactive": True,
            "insight_type": best.insight_type.value
        })
        
        self._metadata["last_interaction"] = datetime.datetime.now().isoformat()
        await self._save_metadata()

        return best.message

    async def _gather_insights(self) -> List[ProactiveInsight]:
        """Gather all potential proactive insights."""
        insights = []

        # 1. Check recent memory events (High Priority)
        memory_events = await self._check_memory_events()
        insights.extend(memory_events)

        # 2. Check scheduled reminders
        reminders = await self._check_scheduled_reminders()
        insights.extend(reminders)

        # 3. Check inactivity (Medium Priority)
        inactivity = await self._check_inactivity()
        if inactivity:
            insights.append(inactivity)

        # 4. Check knowledge gaps (Low Priority)
        knowledge = await self._check_knowledge_gaps()
        if knowledge:
            insights.append(knowledge)

        return insights

    async def _check_memory_events(self) -> List[ProactiveInsight]:
        """Check for relevant past events to follow up on."""
        insights = []
        try:
            # Check for schedules explicitly marked as executed/completed today
            today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0)
            today_end = today_start + datetime.timedelta(days=1)
            
            recent_schedules = await self._mongo.schedules.find({
                "scheduled_at": {"$gte": today_start, "$lt": today_end},
                "status": "executed"
            }).to_list(10)
            
            for schedule in recent_schedules:
                context = schedule.get("context", "")
                # Avoid follow-up if already done
                if await self._is_insight_sent(f"followup_{schedule['_id']}"):
                    continue
                    
                # Simple keyword heuristic for now, better with LLM analysis later
                if any(w in context.lower() for w in ["ujian", "tes", "meeting", "rapat", "dokter", "janji"]):
                    msg = f"Gimana {context}-nya tadi? Lancar kan?"
                    insights.append(ProactiveInsight(
                        insight_type=InsightType.FOLLOW_UP,
                        priority=InsightPriority.HIGH,
                        message=msg,
                        context={"schedule_id": str(schedule["_id"]), "original_context": context}
                    ))
                    # Mark as candidate (actual send will mark as sent)
                    
        except Exception:
            pass
            
        return insights

    async def _check_inactivity(self) -> Optional[ProactiveInsight]:
        last_str = self._metadata.get("last_interaction")
        if not last_str:
            return None

        try:
            last_dt = datetime.datetime.fromisoformat(last_str)
            hours_since = (datetime.datetime.now() - last_dt).total_seconds() / 3600

            if hours_since >= self.INACTIVITY_THRESHOLD_HOURS:
                message = await self._generate_proactive_message(hours_since)
                return ProactiveInsight(
                    insight_type=InsightType.INACTIVITY,
                    priority=InsightPriority.MEDIUM,
                    message=message,
                    context={"hours_inactive": hours_since, "ai_generated": True}
                )
        except Exception:
            pass

        return None

    async def _generate_proactive_message(self, hours_inactive: float) -> str:
        try:
            persona = await self._get_active_persona()
            config = await self._get_system_config()
            
            base_temp = persona.get("temperature", 0.7) if persona else config.get("temperature", 0.7)
            persona_instruction = persona.get("instruction", "") if persona else ""
            
            global_ctx = await self._get_global_context()
            
            days_inactive = int(hours_inactive / 24)
            time_desc = f"{days_inactive} hari" if days_inactive > 0 else f"{int(hours_inactive)} jam"
            
            context_info = ""
            if global_ctx:
                context_info = f"\n\nInformasi tentang user:\n{global_ctx[:500]}"
            
            persona_context = ""
            if persona_instruction:
                persona_context = f"\n\nKarakter persona:\n{persona_instruction[:300]}"
            
            prompt = f"""Kamu adalah Vira, AI assistant yang akrab dengan user. User sudah tidak chat selama {time_desc}.

Buat pesan singkat (1-2 kalimat) untuk menyapa user dengan gaya santai dan akrab. Gunakan bahasa gaul Indonesia.
Jangan terlalu formal. Bisa tanyakan kabar atau apa yang sedang dikerjakan.{persona_context}{context_info}

Pesan:"""

            client = self._brain.openrouter
            response = await client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=base_temp,
                tier="chat_model"
            )
            
            if response and "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"].strip()
            return "Hei, lu kemana aja? Udah lama gak ngobrol nih. Semua baik-baik aja kan?"
            
        except Exception:
            return "Hei, lu kemana aja? Udah lama gak ngobrol nih. Semua baik-baik aja kan?"

    async def _check_scheduled_reminders(self) -> List[ProactiveInsight]:
        """Check for due scheduled reminders."""
        insights = []
        pending = await self.hippocampus.get_pending_schedules(limit=5)

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
        stats = await self.hippocampus.get_memory_stats()
        if stats.get("active", 0) < 5:
            message = await self._generate_knowledge_message(stats.get("active", 0))
            return ProactiveInsight(
                insight_type=InsightType.KNOWLEDGE,
                priority=InsightPriority.LOW,
                message=message,
                context={"memory_count": stats.get("active", 0)}
            )
        return None

    async def _generate_knowledge_message(self, memory_count: int) -> str:
        try:
            persona = await self._get_active_persona()
            config = await self._get_system_config()
            
            base_temp = persona.get("temperature", 0.7) if persona else config.get("temperature", 0.7)
            persona_instruction = persona.get("instruction", "") if persona else ""
            
            global_ctx = await self._get_global_context()
            
            context_info = ""
            if global_ctx:
                context_info = f"\n\nInformasi tentang user:\n{global_ctx[:500]}"
            
            persona_context = ""
            if persona_instruction:
                persona_context = f"\n\nKarakter persona:\n{persona_instruction[:300]}"
            
            prompt = f"""Kamu adalah Vira, AI assistant yang akrab dengan user. Kamu baru mengenal user dan belum tau banyak tentang mereka (baru ada {memory_count} memori).

Buat pesan singkat (1-2 kalimat) untuk menanyakan sesuatu tentang user dengan gaya santai dan akrab. Gunakan bahasa gaul Indonesia.
Bisa tanyakan hobi, mimpi, atau fakta unik tentang mereka.{persona_context}{context_info}

Pesan:"""

            client = self._brain.openrouter
            response = await client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=base_temp,
                tier="chat_model"
            )
            
            if response and "choices" in response and response["choices"]:
                return response["choices"][0]["message"]["content"].strip()
            return "Gw pengen tau lebih banyak tentang lu. Ceritain sesuatu dong!"
            
        except Exception:
            return "Gw pengen tau lebih banyak tentang lu. Ceritain sesuatu dong!"

    def _can_send_insight(self, insight_type: InsightType) -> bool:
        """Check if an insight type can be sent (cooldown check)."""
        key = insight_type.value
        last_sent = self._insight_cache.get(key)
        if not last_sent:
            return True

        minutes_since = (datetime.datetime.now() - last_sent).total_seconds() / 60
        return minutes_since >= self.COOLDOWN_MINUTES

    async def _is_insight_sent(self, unique_id: str) -> bool:
        timestamp = self._insight_cache.get(unique_id)
        if not timestamp:
            return False

        age = (datetime.datetime.now() - timestamp).total_seconds() / 3600
        return age < 24

    def _mark_insight_sent(self, insight_type: InsightType) -> None:
        """Mark an insight type as sent."""
        self._insight_cache[insight_type.value] = datetime.datetime.now()

    def _mark_insight_sent_with_id(self, unique_id: str) -> None:
        """Mark a specific insight ID as sent."""
        self._insight_cache[unique_id] = datetime.datetime.now()

    def get_last_interaction(self) -> Optional[datetime.datetime]:
        """Get the last interaction timestamp."""
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

        pending = await self.hippocampus.get_pending_schedules(limit=1)
        if pending:
            return True

        return False

    def format_time_gap(self, last_time: Optional[datetime.datetime]) -> str:
        """Format time gap for display."""
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

    async def get_context_stats(self) -> Dict[str, Any]:
        global_ctx = await self._get_global_context()

        return {
            "has_global_context": global_ctx is not None,
            "global_context_length": len(global_ctx) if global_ctx else 0,
            "max_context_tokens": self.MAX_CONTEXT_TOKENS,
            "weather_cached": self._is_weather_cached(),
            "last_interaction": self._metadata.get("last_interaction"),
            "insight_cooldowns": {
                k: v.isoformat() for k, v in self._insight_cache.items()
            }
        }
