import os
import json
import asyncio
import re
import hashlib
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from functools import wraps
import time

import httpx
import PIL.Image
from dateutil import parser as date_parser

from src.brain.brainstem import (
    MemoryType,
    AllAPIExhaustedError,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    NeuralEventBus,
    OpenRouterClient
)
from src.brain.constants import (
    DEFAULT_PERSONA_INSTRUCTION,
    EXTRACTION_INSTRUCTION
)
from src.brain.hippocampus import Hippocampus, Memory
from src.brain.amygdala import Amygdala, PlanProgressState
from src.brain.thalamus import Thalamus
from src.brain.parietal_lobe import ParietalLobe


class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    id: Optional[int]
    plan_id: int
    order_index: int
    description: str
    action_type: str
    status: StepStatus = StepStatus.PENDING
    output_result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    retry_count: int = 0


@dataclass
class TaskPlan:
    id: Optional[int]
    goal: str
    original_request: str
    status: PlanStatus = PlanStatus.PENDING
    steps: List[TaskStep] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    priority: int = 5
    deadline: Optional[datetime] = None


class IntentType(str, Enum):
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    GREETING = "greeting"
    COMMAND = "command"
    SMALL_TALK = "small_talk"
    CONFIRMATION = "confirmation"
    CORRECTION = "correction"


class RequestType(str, Enum):
    INFORMATION = "information"
    RECOMMENDATION = "recommendation"
    MEMORY_RECALL = "memory_recall"
    OPINION = "opinion"
    ACTION = "action"
    SCHEDULE = "schedule"
    GENERAL_CHAT = "general_chat"


@dataclass
class ProcessingMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    tool_executions: int = 0
    memory_retrievals: int = 0
    
    def record_request(self, success: bool, duration: float):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + duration) / self.total_requests
        )


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: deque = deque()
    
    async def acquire(self) -> bool:
        now = time.time()
        while self.requests and self.requests[0] < now - self.window_seconds:
            self.requests.popleft()
        
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
    
    def get_wait_time(self) -> float:
        if not self.requests or len(self.requests) < self.max_requests:
            return 0.0
        oldest = self.requests[0]
        return max(0.0, self.window_seconds - (time.time() - oldest))


def async_retry(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
            raise last_exception
        return wrapper
    return decorator


class PrefrontalCortex:
    MAX_OUTPUT_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    INTENT_CACHE_TTL: int = 300
    RESPONSE_CACHE_TTL: int = 60
    MAX_CACHE_SIZE: int = 100
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: int = 60

    def __init__(self):
        self._brain = None
        self._openrouter: Optional[OpenRouterClient] = None
        self._active_plan: Optional[TaskPlan] = None
        self._intent_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._response_cache: Dict[str, Tuple[str, datetime]] = {}
        self._processing = False
        self._current_persona: Optional[Dict] = None
        self._metrics = ProcessingMetrics()
        self._rate_limiter = RateLimiter(self.RATE_LIMIT_REQUESTS, self.RATE_LIMIT_WINDOW)
        self._plan_history: List[TaskPlan] = []
        
    def bind_brain(self, brain) -> None:
        self._brain = brain

    @property
    def hippocampus(self):
        return self._brain.hippocampus if self._brain else None

    @property
    def amygdala(self):
        return self._brain.amygdala if self._brain else None

    @property
    def thalamus(self):
        return self._brain.thalamus if self._brain else None

    @property
    def parietal_lobe(self):
        return self._brain.parietal_lobe if self._brain else None

    async def initialize(self) -> None:
        self._openrouter = self._brain.openrouter
        await self._load_active_persona()
        await self._load_active_plan()
        await self._cleanup_old_cache()

    async def _load_active_persona(self) -> None:
        self._current_persona = await self.hippocampus.get_active_persona()
        if self._current_persona:
            print(f"  * Active Persona: {self._current_persona.get('name', 'Unknown')}")
        else:
            print("  ! No active persona found, using default")

    async def _load_active_plan(self) -> None:
        pass

    async def _cleanup_old_cache(self) -> None:
        now = datetime.now()
        self._intent_cache = {
            k: v for k, v in self._intent_cache.items()
            if (now - v[1]).total_seconds() < self.INTENT_CACHE_TTL
        }
        self._response_cache = {
            k: v for k, v in self._response_cache.items()
            if (now - v[1]).total_seconds() < self.RESPONSE_CACHE_TTL
        }
        
        if len(self._intent_cache) > self.MAX_CACHE_SIZE:
            sorted_items = sorted(self._intent_cache.items(), key=lambda x: x[1][1])
            self._intent_cache = dict(sorted_items[-self.MAX_CACHE_SIZE:])
        
        if len(self._response_cache) > self.MAX_CACHE_SIZE:
            sorted_items = sorted(self._response_cache.items(), key=lambda x: x[1][1])
            self._response_cache = dict(sorted_items[-self.MAX_CACHE_SIZE:])

    def _calculate_dynamic_params(
        self,
        intent_type: str,
        request_type: str,
        emotion: str,
        base_temp: float
    ) -> Dict[str, Any]:
        params = {
            "temperature": base_temp,
            "max_tokens": self.MAX_OUTPUT_TOKENS,
            "top_p": self.TOP_P
        }

        if request_type in [RequestType.SCHEDULE.value, RequestType.ACTION.value, RequestType.MEMORY_RECALL.value]:
            params["temperature"] = max(0.1, base_temp - 0.4)
            params["top_p"] = 0.85
        elif request_type in [RequestType.GENERAL_CHAT.value, RequestType.RECOMMENDATION.value]:
            params["temperature"] = min(1.0, base_temp + 0.1)
            params["top_p"] = 0.95

        if emotion in ["happy", "excited"]:
            params["temperature"] = min(1.0, params["temperature"] + 0.1)
            params["max_tokens"] = int(self.MAX_OUTPUT_TOKENS * 1.2)
        elif emotion in ["sad", "concerned", "anxious"]:
            params["temperature"] = max(0.3, params["temperature"] - 0.2)
            params["max_tokens"] = int(self.MAX_OUTPUT_TOKENS * 0.6)
        elif emotion in ["angry"]:
            params["temperature"] = 0.2
            params["max_tokens"] = 512

        if intent_type == IntentType.QUESTION.value:
            params["temperature"] = max(0.3, params["temperature"] - 0.1)

        params["temperature"] = round(max(0.0, min(1.0, params["temperature"])), 2)
        
        return params

    def _get_persona_instruction(self) -> str:
        if self._current_persona and self._current_persona.get("instruction"):
            return self._current_persona["instruction"]
        return DEFAULT_PERSONA_INSTRUCTION
    
    def _get_persona_temperature(self) -> float:
        if self._current_persona and self._current_persona.get("temperature"):
            return self._current_persona["temperature"]
        return self.TEMPERATURE

    async def process(
        self,
        message: str,
        image_path: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> str:
        if self._processing:
            return "⏳ Sedang memproses pesan sebelumnya..."

        if not await self._rate_limiter.acquire():
            wait_time = self._rate_limiter.get_wait_time()
            return f"⏳ Rate limit tercapai. Tunggu {int(wait_time)} detik."

        start_time = time.time()
        self._processing = True
        success = False
        
        try:
            cache_key = self._get_cache_key(message, image_path)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self._metrics.cache_hits += 1
                return cached_response
            self._metrics.cache_misses += 1

            if user_name:
                await self.hippocampus.update_admin_profile(telegram_name=user_name)

            user_embedding = await self._generate_embedding(message)

            await NeuralEventBus.set_activity("prefrontal_cortex", "Analyzing Intent")
            intent_data = await self._extract_intent(message)
            await NeuralEventBus.emit("prefrontal_cortex", "prefrontal_cortex", "intent_extracted", payload=intent_data)
            
            await NeuralEventBus.emit("prefrontal_cortex", "amygdala", "emotion_check")
            if self.amygdala:
                detected_emotion, intensity = await self.amygdala.detect_emotion_from_text(message)
                await NeuralEventBus.emit("amygdala", "prefrontal_cortex", "emotion_detected", payload={
                    "emotion": detected_emotion,
                    "intensity": intensity
                })
                self.amygdala.adjust_for_emotion(detected_emotion, intensity)

            reflex_context = ""
            if self.parietal_lobe:
                reflex_check = await self._detect_reflex_need(message)
                if reflex_check:
                    tool_name, tool_args = reflex_check
                    await NeuralEventBus.set_activity("prefrontal_cortex", f"Using Tool: {tool_name}")
                    
                    tool_result = await self._execute_tool_with_retry(tool_name, tool_args)
                    if tool_result:
                        reflex_context = f"[TOOL RESULT ({tool_name})]\n{tool_result}\n\n"
                        self._metrics.tool_executions += 1

            await NeuralEventBus.set_activity("prefrontal_cortex", "Retrieving Memories")
            await NeuralEventBus.emit("prefrontal_cortex", "hippocampus", "memory_retrieval", payload={
                "query": intent_data.get("search_query", message)[:50]
            })
            memories = await self._retrieve_memories(message, intent_data)
            self._metrics.memory_retrievals += 1
            await NeuralEventBus.emit("hippocampus", "prefrontal_cortex", "memories_found", payload={
                "count": len(memories),
                "types": [m.memory_type for m in memories]
            })

            if self._active_plan:
                await NeuralEventBus.set_activity("prefrontal_cortex", "Executing Plan")
                plan_response = await self._handle_active_plan(message, intent_data)
                if plan_response:
                    await NeuralEventBus.clear_activity("prefrontal_cortex")
                    success = True
                    return plan_response

            if self._should_create_plan(message, intent_data):
                await NeuralEventBus.set_activity("prefrontal_cortex", "Creating New Plan")
                plan = await self.create_plan(message)
                if plan:
                    self.amygdala.update_satisfaction(PlanProgressState.ON_TRACK)

            schedules = await self.hippocampus.get_upcoming_schedules(hours_ahead=24)
            schedule_context = self._format_schedules(schedules) if schedules else None

            await NeuralEventBus.set_activity("prefrontal_cortex", "Building Context")
            await NeuralEventBus.emit("prefrontal_cortex", "thalamus", "context_building")
            
            relevant_history = await self.thalamus.get_relevant_history_async(user_embedding, top_k=5)
            relevant_history_context = self._format_relevant_history(relevant_history)

            context = await self.thalamus.build_context(
                relevant_memories=[asdict(m) for m in memories] if memories else [],
                schedule_context=schedule_context,
                intent_data=intent_data,
                query_embedding=user_embedding
            )
            
            if relevant_history_context and "[RELEVANT PAST CONVERSATIONS]" not in context:
                context = f"[RELEVANT PAST CONVERSATIONS]\n{relevant_history_context}\n\n{context}"

            if reflex_context:
                context = f"{reflex_context}{context}"

            await NeuralEventBus.set_activity("prefrontal_cortex", "Generating Response")
            response = await self._generate_response(message, context, image_path, intent_info=intent_data)

            self._cache_response(cache_key, response)

            response_embedding = await self._generate_embedding(response)
            await self.thalamus.update_session(
                message, response, image_path,
                user_embedding=user_embedding,
                ai_embedding=response_embedding
            )

            asyncio.create_task(self._post_process(message, response, intent_data))
            
            await NeuralEventBus.emit("prefrontal_cortex", "motor_cortex", "output_sent", payload={
                "response_len": len(response),
                "has_image": image_path is not None
            })
            await NeuralEventBus.clear_activity("prefrontal_cortex")

            success = True
            return response

        except AllAPIExhaustedError:
            return "⚠️ Semua API dan model sedang kehabisan kuota. Coba lagi nanti."
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"ERROR: {str(e)}"
        finally:
            duration = time.time() - start_time
            self._metrics.record_request(success, duration)
            self._processing = False

    def _get_cache_key(self, message: str, image_path: Optional[str]) -> str:
        key_data = f"{message}:{image_path or ''}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cached_response(self, cache_key: str) -> Optional[str]:
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.RESPONSE_CACHE_TTL:
                return response
        return None

    def _cache_response(self, cache_key: str, response: str) -> None:
        self._response_cache[cache_key] = (response, datetime.now())

    @async_retry(max_retries=2, delay=0.5)
    async def _execute_tool_with_retry(self, tool_name: str, tool_args: Dict) -> Optional[str]:
        tool_result = await self.parietal_lobe.execute(tool_name, tool_args)
        if tool_result and "error" not in tool_result.lower()[:50]:
            return tool_result
        raise Exception(f"Tool execution failed: {tool_result}")

    def _format_relevant_history(self, messages: List) -> str:
        if not messages:
            return ""
        lines = []
        for msg in messages[:5]:
            time_str = msg.timestamp.strftime("%d/%m %H:%M") if hasattr(msg, 'timestamp') else ""
            role = "User" if msg.role == "user" else "AI"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"[{time_str}] {role}: {content}")
        return "\n".join(lines)

    async def _extract_intent(self, text: str) -> Dict:
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        if cache_key in self._intent_cache:
            cached, timestamp = self._intent_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.INTENT_CACHE_TTL:
                return cached

        try:
            prompt = f"{EXTRACTION_INSTRUCTION}\n\nInput: \"{text}\""
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.1,
                tier="analysis_model",
                json_mode=True
            )
            result = self._extract_json(response) or self._fallback_intent(text)
            self._intent_cache[cache_key] = (result, datetime.now())
            return result
        except Exception:
            return self._fallback_intent(text)

    async def _detect_reflex_need(self, text: str) -> Optional[Tuple[str, Dict]]:
        if not self.parietal_lobe:
            return None
            
        text_lower = text.lower()
        triggers = [
            "jam", "waktu", "pukul", "time", "date", "tanggal", "hitung", "calc", "cuaca", "weather",
            "ingat", "remember", "simpan", "save", "catat", "note", "memo",
            "jadwal", "schedule", "reminder", "ingatkan", "remind", "alarm",
            "lupa", "forget", "hapus", "delete", "update", "ubah"
        ]
        if not any(t in text_lower for t in triggers):
            return None

        tools_desc = self.parietal_lobe.get_tool_descriptions()
        if not tools_desc:
            return None

        prompt = f"""# TOOL SELECTION
Determine if the input requires a tool. Available tools:
{tools_desc}

Input: "{text}"

If a tool is needed, return JSON:
{{ "tool": "tool_name", "args": {{ "arg_name": "value" }} }}

If no tool is needed (just chat), return {{ "tool": null }}
"""
        
        try:
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=128,
                temperature=0.0,
                tier="utility_model",
                json_mode=True
            )
            data = self._extract_json(response)
            if data and data.get("tool"):
                return (data["tool"], data.get("args", {}))
        except Exception:
            pass
            
        return None

    def _fallback_intent(self, text: str) -> Dict:
        text_lower = text.lower()

        intent_type = IntentType.STATEMENT.value
        if "?" in text or any(w in text_lower for w in ["apa", "siapa", "kapan", "dimana", "gimana", "kenapa", "what", "who", "when", "where", "how", "why"]):
            intent_type = IntentType.QUESTION.value
        elif any(w in text_lower for w in ["ingatkan", "remind", "jadwal", "schedule", "tolong", "please"]):
            intent_type = IntentType.REQUEST.value
        elif any(w in text_lower for w in ["hai", "halo", "hi", "hello", "pagi", "siang", "sore", "malam"]):
            intent_type = IntentType.GREETING.value

        request_type = RequestType.GENERAL_CHAT.value
        if any(w in text_lower for w in ["jadwal", "ingatkan", "remind", "schedule"]):
            request_type = RequestType.SCHEDULE.value
        elif any(w in text_lower for w in ["ingat", "tau", "remember", "know"]):
            request_type = RequestType.MEMORY_RECALL.value

        entities = []
        words = text.split()
        for w in words:
            if w and w[0].isupper() and len(w) > 2:
                entities.append(w.lower())

        return {
            "intent_type": intent_type,
            "request_type": request_type,
            "entities": entities[:5],
            "key_concepts": [],
            "search_query": text[:100],
            "temporal_context": "present",
            "sentiment": "neutral",
            "language": "id",
            "needs_memory": request_type == RequestType.MEMORY_RECALL.value,
            "memory_scope": "personal" if request_type == RequestType.MEMORY_RECALL.value else None,
            "confidence": 0.6
        }

    async def _retrieve_memories(
        self,
        query: str,
        intent_data: Dict
    ) -> List[Memory]:
        if not intent_data.get("needs_memory", True):
            if intent_data.get("intent_type") == IntentType.GREETING.value:
                return []

        search_query = intent_data.get("search_query", query)
        memories = await self.hippocampus.recall(search_query, limit=3)

        entities = intent_data.get("entities", [])
        for entity in entities[:2]:
            entity_data = await self.hippocampus.query_entity(entity)
            if entity_data.get("memories"):
                for m in entity_data["memories"][:2]:
                    if not any(existing.id == m["id"] for existing in memories):
                        memories.append(Memory(
                            id=m["id"],
                            summary=m["summary"],
                            memory_type=m["type"],
                            confidence=m.get("confidence", 0.5)
                        ))

        return memories[:5]

    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if not text or len(text) < 5:
            return None

        try:
            ollama_url = OLLAMA_BASE_URL.replace("/v1", "")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": EMBEDDING_MODEL, "prompt": text[:2000]}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("embedding")
        except Exception:
            pass
        return None

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        return await self._generate_embedding(text)

    async def _generate_response(
        self,
        user_text: str,
        context: str,
        image_path: Optional[str] = None,
        intent_info: Optional[Dict] = None
    ) -> str:
        persona_instruction = self._get_persona_instruction()
        persona_modifier = self.amygdala.get_response_modifier()
        full_instruction = persona_instruction + persona_modifier
        
        base_temp = self._get_persona_temperature()
        
        i_type = intent_info.get("intent_type", "statement") if intent_info else "statement"
        r_type = intent_info.get("request_type", "general_chat") if intent_info else "general_chat"
        current_mood = self.amygdala.mood.value
        
        cognitive_params = self._calculate_dynamic_params(i_type, r_type, current_mood, base_temp)
        
        if cognitive_params["temperature"] != base_temp:
            print(f"  * Neuro-Modulation: Temp {base_temp} -> {cognitive_params['temperature']} | Mood: {current_mood}")
        
        user_content = user_text
        if context:
            user_content = f"[CONTEXT]\n{context}\n\n[USER INPUT]\n{user_text}"
        
        if image_path and os.path.exists(image_path):
            user_content += "\n\n[Note: User sent an image]"
        
        history = await self.thalamus.get_history_for_model_async()
        messages = [{"role": "system", "content": full_instruction}]
        
        for msg in history[-20:]:
            role = "user" if msg.role == "user" else "assistant"
            content = ""
            for part in msg.parts:
                if hasattr(part, 'text'):
                    content += part.text
            if content:
                messages.append({"role": role, "content": content})
        
        messages.append({"role": "user", "content": user_content})
        
        response = await self._openrouter.chat_completion(
            messages=messages,
            temperature=cognitive_params["temperature"],
            max_tokens=cognitive_params["max_tokens"],
            top_p=cognitive_params["top_p"],
            tier="chat_model"
        )
        
        prefix = self.amygdala.get_response_prefix()
        if response and "choices" in response and response["choices"]:
            text = response["choices"][0]["message"]["content"].strip()
        else:
            text = "Maaf, ada kendala dalam memproses permintaan."
        if prefix:
            text = f"{prefix} {text}"
        
        return text

    async def _post_process(
        self,
        user_text: str,
        ai_response: str,
        intent_data: Dict
    ) -> None:
        pass

    def _should_create_plan(self, text: str, intent_data: Dict) -> bool:
        if self._active_plan:
            return False

        text_lower = text.lower()
        plan_triggers = [
            "buatkan rencana", "buat jadwal", "susun rencana",
            "make a plan", "create schedule", "help me plan",
            "step by step", "langkah-langkah"
        ]

        for trigger in plan_triggers:
            if trigger in text_lower:
                return True

        return False

    async def create_plan(self, goal: str, priority: int = 5, deadline: Optional[datetime] = None) -> Optional[TaskPlan]:
        try:
            prompt = f"""Decompose this goal into 3-5 actionable steps:

Goal: {goal}

Output format (JSON):
{{
  "goal_summary": "brief summary",
  "steps": [
    {{"order": 1, "description": "step description", "action_type": "action|info|decision"}}
  ]
}}"""

            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.1
            )

            data = self._extract_json(response)
            if not data:
                return None

            plan = TaskPlan(
                id=len(self._plan_history) + 1,
                goal=data.get("goal_summary", goal),
                original_request=goal,
                status=PlanStatus.IN_PROGRESS,
                priority=priority,
                deadline=deadline
            )

            for step_data in data.get("steps", []):
                step = TaskStep(
                    id=step_data.get("order"),
                    plan_id=plan.id,
                    order_index=step_data.get("order", 0),
                    description=step_data.get("description", ""),
                    action_type=step_data.get("action_type", "action")
                )
                plan.steps.append(step)

            self._active_plan = plan
            return plan

        except Exception:
            return None

    async def _handle_active_plan(
        self,
        user_input: str,
        intent_data: Dict
    ) -> Optional[str]:
        if not self._active_plan:
            return None

        text_lower = user_input.lower()

        if any(w in text_lower for w in ["batal", "cancel", "stop", "hentikan"]):
            self._active_plan.status = PlanStatus.CANCELLED
            self.amygdala.update_satisfaction(PlanProgressState.ABANDONED)
            self._plan_history.append(self._active_plan)
            old_plan = self._active_plan
            self._active_plan = None
            return f"Oke, rencana '{old_plan.goal}' dibatalkan."

        if any(w in text_lower for w in ["pause", "jeda", "tunda"]):
            self._active_plan.status = PlanStatus.PAUSED
            return f"Rencana '{self._active_plan.goal}' dijeda sementara."

        if any(w in text_lower for w in ["lanjut", "continue", "resume"]):
            if self._active_plan.status == PlanStatus.PAUSED:
                self._active_plan.status = PlanStatus.IN_PROGRESS
                return f"Rencana '{self._active_plan.goal}' dilanjutkan."

        if any(w in text_lower for w in ["selesai", "done", "sudah", "completed"]):
            current_step = self._get_current_step()
            if current_step:
                current_step.status = StepStatus.COMPLETED
                current_step.completed_at = datetime.now()
                self.amygdala.update_satisfaction(PlanProgressState.ON_TRACK)

                next_step = self._get_current_step()
                if next_step:
                    return f"Mantap! Langkah selanjutnya: {next_step.description}"
                else:
                    self._active_plan.status = PlanStatus.COMPLETED
                    self.amygdala.update_satisfaction(PlanProgressState.COMPLETED)
                    self._plan_history.append(self._active_plan)
                    goal = self._active_plan.goal
                    self._active_plan = None
                    return f"Selamat! Rencana '{goal}' selesai semua. Bangga sama lu!"

        if any(w in text_lower for w in ["skip", "lewati"]):
            current_step = self._get_current_step()
            if current_step:
                current_step.status = StepStatus.SKIPPED
                next_step = self._get_current_step()
                if next_step:
                    return f"Langkah dilewati. Selanjutnya: {next_step.description}"

        return None

    def _get_current_step(self) -> Optional[TaskStep]:
        if not self._active_plan:
            return None

        for step in self._active_plan.steps:
            if step.status in [StepStatus.PENDING, StepStatus.IN_PROGRESS]:
                return step
        return None

    def get_active_plan(self) -> Optional[TaskPlan]:
        return self._active_plan

    def get_plan_history(self) -> List[TaskPlan]:
        return self._plan_history

    def get_plan_progress(self) -> Optional[Dict[str, Any]]:
        if not self._active_plan:
            return None
        
        total_steps = len(self._active_plan.steps)
        completed_steps = sum(1 for s in self._active_plan.steps if s.status == StepStatus.COMPLETED)
        
        return {
            "goal": self._active_plan.goal,
            "progress": f"{completed_steps}/{total_steps}",
            "percentage": int((completed_steps / total_steps) * 100) if total_steps > 0 else 0,
            "current_step": self._get_current_step().description if self._get_current_step() else None,
            "status": self._active_plan.status.value
        }

    async def check_pending_schedules(self, bot) -> None:
        pending = await self.hippocampus.get_pending_schedules(limit=5)

        for schedule in pending:
            schedule_id = schedule.get("id")
            context = schedule.get("context", "")

            from src.brain.brainstem import ADMIN_ID
            if ADMIN_ID:
                try:
                    await bot.send_message(
                        chat_id=int(ADMIN_ID),
                        text=f"⏰ *Pengingat*: {context}"
                    )
                    await self.hippocampus.mark_schedule_executed(schedule_id, "delivered")
                except Exception:
                    pass

    def _format_schedules(self, schedules: List[Dict]) -> str:
        if not schedules:
            return ""

        lines = []
        now = datetime.now()
        for s in schedules[:5]:
            scheduled_at = s.get("scheduled_at")
            if isinstance(scheduled_at, str):
                try:
                    scheduled_at = datetime.fromisoformat(scheduled_at)
                except Exception:
                    continue

            delta = scheduled_at - now
            hours = delta.total_seconds() / 3600

            if hours < 0:
                time_str = "OVERDUE"
            elif hours < 1:
                time_str = f"{int(delta.total_seconds() / 60)}m"
            elif hours < 24:
                time_str = f"{int(hours)}h"
            else:
                time_str = scheduled_at.strftime("%d/%m %H:%M")

            context = s.get("context", "")[:50]
            lines.append(f"• [{time_str}] {context}")

        return "\n".join(lines)

    def _extract_json(self, text: str) -> Optional[Dict]:
        if not text or not text.strip():
            return None
        
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx >= 0:
                    end_idx = i + 1
                    break
        
        if start_idx >= 0 and end_idx > start_idx:
            json_str = text[start_idx:end_idx]
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        
        simple_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if simple_match:
            try:
                return json.loads(simple_match.group())
            except json.JSONDecodeError:
                pass
        
        return None

    async def switch_persona(self, persona_id: str) -> bool:
        success = await self.hippocampus.set_active_persona(persona_id)
        if success:
            await self._load_active_persona()
            await NeuralEventBus.emit(
                "prefrontal_cortex", "dashboard", "persona_changed",
                payload={
                    "persona_id": persona_id,
                    "persona_name": self._current_persona.get("name") if self._current_persona else "Unknown"
                }
            )
        return success

    def get_system_stats(self) -> Dict[str, Any]:
        api_status = self._openrouter.get_status()
        return {
            "api_health": "OK" if api_status.get("api_configured") else "No API Key",
            "active_plan": self._active_plan.goal if self._active_plan else None,
            "sessions": 1,
            "active_processing": 1 if self._processing else 0,
            "total_users_tracked": 1,
            "current_persona": self._current_persona.get("name") if self._current_persona else "Default",
            "model_health": api_status.get("model_health", {}),
            "metrics": {
                "total_requests": self._metrics.total_requests,
                "success_rate": f"{(self._metrics.successful_requests / max(self._metrics.total_requests, 1)) * 100:.1f}%",
                "avg_response_time": f"{self._metrics.avg_response_time:.2f}s",
                "cache_hit_rate": f"{(self._metrics.cache_hits / max(self._metrics.cache_hits + self._metrics.cache_misses, 1)) * 100:.1f}%",
                "tool_executions": self._metrics.tool_executions,
                "memory_retrievals": self._metrics.memory_retrievals
            }
        }

    def get_metrics(self) -> ProcessingMetrics:
        return self._metrics

    def reset_metrics(self) -> None:
        self._metrics = ProcessingMetrics()

    async def clear_cache(self) -> Dict[str, int]:
        intent_cache_size = len(self._intent_cache)
        response_cache_size = len(self._response_cache)
        
        self._intent_cache.clear()
        self._response_cache.clear()
        
        return {
            "intent_cache_cleared": intent_cache_size,
            "response_cache_cleared": response_cache_size
        }

    async def check_pending_schedules(self, bot) -> None:
        if not self.hippocampus:
            return
        
        try:
            from src.brain.brainstem import SYSTEM_CONFIG
            
            pending = await self.hippocampus.get_pending_schedules(limit=10)
            
            for schedule in pending:
                schedule_id = schedule.get("id")
                context = schedule.get("context", "Pengingat")
                
                try:
                    message = f"⏰ Pengingat: {context}"
                    await bot.send_message(
                        chat_id=int(SYSTEM_CONFIG.admin_id),
                        text=message
                    )
                    
                    await self.hippocampus._mongo.schedules.update_one(
                        {"_id": schedule_id},
                        {"$set": {"status": "executed", "executed_at": datetime.now()}}
                    )
                except Exception:
                    pass
        except Exception:
            pass