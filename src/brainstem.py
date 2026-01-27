import os
import asyncio
import json
import math
from enum import Enum
from typing import Optional, Dict, List, Set, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, Application, Defaults, CallbackQueryHandler
)
import httpx

load_dotenv()

ADMIN_ID: str = os.getenv("ADMIN_TELEGRAM_ID", "")
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
DB_PATH: str = os.getenv("DB_PATH", "storage/memory.db")

OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "https://vira-os.local")
OPENROUTER_APP_NAME: str = os.getenv("OPENROUTER_APP_NAME", "Vira Personal Life OS")

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")

METEOSOURCE_API_KEY: Optional[str] = os.getenv("METEOSOURCE_API_KEY")

OPENROUTER_MODELS: Dict[str, List[str]] = {
    "tier_1": [
        "openai/gpt-oss-120b:free",
        "deepseek/deepseek-v3.2",
        "anthropic/claude-sonnet-4",
        "google/gemini-3-flash-preview",
    ],
    "tier_2": [
        "openai/gpt-oss-120b:free",
        "google/gemini-2.0-flash-001",
        "meta-llama/llama-3.3-70b-instruct",
    ],
    "tier_3": [
        "openai/gpt-oss-120b:free",
        "mistralai/mistral-large-2411",
        "google/gemini-flash-1.5-8b",
    ]
}


def get_heavy_model() -> str:
    """Get the preferred model for heavy/complex tasks (Tier 1)."""
    return OPENROUTER_MODELS.get("tier_1", [])[0] if OPENROUTER_MODELS.get("tier_1") else "deepseek/deepseek-v3.2"

def get_chat_model() -> str:
    """Get the preferred model for general chat (Tier 2)."""
    return OPENROUTER_MODELS.get("tier_2", [])[0] if OPENROUTER_MODELS.get("tier_2") else "openai/gpt-oss-120b:free"

def get_light_model() -> str:
    """Get the preferred model for light/background tasks (Tier 3)."""
    return OPENROUTER_MODELS.get("tier_3", [])[0] if OPENROUTER_MODELS.get("tier_3") else "google/gemini-flash-1.5-8b"

EMBEDDING_DIMENSION: int = 1024

MAX_RETRIEVED_MEMORIES: int = 3
MIN_RELEVANCE_SCORE: float = 0.6
DECAY_DAYS_EMOTION: int = 7
DECAY_DAYS_GENERAL: int = 60

class MemoryType(str, Enum):
    EMOTION = "emotion"
    DECISION = "decision"
    PREFERENCE = "preference"
    BOUNDARY = "boundary"
    BIOGRAPHY = "biography"
    FACT = "fact"
    SKILL = "skill"
    EVENT = "event"
    CONTEXT = "context"

class MoodState(str, Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    CONCERNED = "concerned"
    PROUD = "proud"
    DISAPPOINTED = "disappointed"
    EXCITED = "excited"

class SystemConfig(BaseModel):
    admin_id: str = Field(default="")
    chat_model: str = Field(default_factory=get_chat_model)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    max_output_tokens: int = Field(default=512)
    proactive_check_interval: int = Field(default=1800)
    session_cleanup_interval: int = Field(default=1800)
    memory_optimization_interval: int = Field(default=7200)
    schedule_check_interval: int = Field(default=60)
    memory_compression_interval: int = Field(default=1800)

SYSTEM_CONFIG = SystemConfig(admin_id=ADMIN_ID)

class NeuralEventBus:
    _current_activity: dict = {}
    _subscribers: list = []
    _recent_events: list = []
    _max_events: int = 50
    
    @classmethod
    def subscribe(cls, callback) -> None:
        """Subscribe to neural events."""
        if callback not in cls._subscribers:
            cls._subscribers.append(callback)
    
    @classmethod
    def unsubscribe(cls, callback) -> None:
        """Unsubscribe from neural events."""
        if callback in cls._subscribers:
            cls._subscribers.remove(callback)
            
    @classmethod
    async def set_activity(cls, module: str, description: str, payload: dict = None) -> None:
        """Set the current activity for a specific module."""
        module = module.lower().replace(" ", "_")
        if cls._current_activity.get(module) != description:
            cls._current_activity[module] = description
            await cls.emit(module, "dashboard", "activity_update", payload=payload)

    @classmethod
    async def clear_activity(cls, module: str) -> None:
        """Reset module activity to Idle."""
        module = module.lower().replace(" ", "_")
        if module in cls._current_activity:
            del cls._current_activity[module]
            await cls.emit(module, "dashboard", "activity_update")
    
    @classmethod
    async def emit(cls, source: str, target: str, event_type: str = "signal", payload: dict = None) -> None:
        """Emit a neural activity event."""
        event = {
            "source": source,
            "target": target,
            "type": event_type,
            "activities": cls._current_activity.copy(),
            "payload": payload or {},
            "timestamp": datetime.now().isoformat()
        }
        
        cls._recent_events.append(event)
        if len(cls._recent_events) > cls._max_events:
            cls._recent_events.pop(0)
        
        for subscriber in cls._subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    asyncio.create_task(subscriber(event))
                else:
                    subscriber(event)
            except Exception:
                pass
    
    @classmethod
    def get_recent_events(cls, limit: int = 20) -> list:
        """Get recent neural events."""
        return cls._recent_events[-limit:]
    
    @classmethod
    def get_module_states(cls) -> dict:
        """Get current state of all modules based on recent activity."""
        now = datetime.now()
        active_threshold = timedelta(seconds=5)
        
        modules = {
            "brainstem": "idle",
            "hippocampus": "idle",
            "amygdala": "idle",
            "thalamus": "idle",
            "prefrontal_cortex": "idle",
            "motor_cortex": "idle",
            "cerebellum": "idle",
            "occipital_lobe": "active",
            "medulla_oblongata": "idle"
        }
        
        for event in cls._recent_events[-20:]:
            try:
                event_time = datetime.fromisoformat(event["timestamp"])
                if now - event_time < active_threshold:
                    if event["source"] in modules:
                        modules[event["source"]] = "active"
                    if event["target"] in modules:
                        modules[event["target"]] = "active"
            except Exception:
                pass
        
        modules["_meta"] = {"activities": cls._current_activity.copy()}
        
        return modules

import logging

logger = logging.getLogger(__name__)

class AllAPIExhaustedError(Exception):
    pass

@dataclass
class ModelRotationConfig:
    retry_delay_base: float = 1.0
    retry_delay_max: float = 30.0
    health_recovery_minutes: int = 30
    max_consecutive_failures: int = 5
    tier_fallback_enabled: bool = True

class ModelHealthScore:
    
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.success_count: int = 0
        self.failure_count: int = 0
        self.total_latency_ms: float = 0.0
        self.last_success: Optional[datetime] = None
        self.last_failure: Optional[datetime] = None
        self.consecutive_failures: int = 0
        self.is_blacklisted: bool = False
        self.blacklist_until: Optional[datetime] = None
    
    @property
    def health_score(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 1.0
        
        success_rate = self.success_count / total
        failure_penalty = min(0.5, self.consecutive_failures * 0.15)
        
        recency_penalty = 0.0
        if self.last_failure:
            minutes_since_failure = (datetime.now() - self.last_failure).total_seconds() / 60
            if minutes_since_failure < 5:
                recency_penalty = 0.4
            elif minutes_since_failure < 15:
                recency_penalty = 0.2
            elif minutes_since_failure < 30:
                recency_penalty = 0.1
        
        latency_penalty = 0.0
        if self.avg_latency_ms > 10000:
            latency_penalty = 0.2
        elif self.avg_latency_ms > 5000:
            latency_penalty = 0.1
        
        return max(0.0, success_rate - failure_penalty - recency_penalty - latency_penalty)
    
    @property
    def composite_score(self) -> float:
        if self.is_blacklisted and self.blacklist_until and datetime.now() < self.blacklist_until:
            return -1.0
        health = self.health_score
        latency_factor = 1.0 - min(0.3, (self.avg_latency_ms / 30000))
        return health * 0.7 + latency_factor * 0.3
    
    @property
    def avg_latency_ms(self) -> float:
        if self.success_count == 0:
            return 0.0
        return self.total_latency_ms / self.success_count
    
    def record_success(self, latency_ms: float) -> None:
        self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_success = datetime.now()
        self.consecutive_failures = 0
        self.is_blacklisted = False
        self.blacklist_until = None
    
    def record_failure(self, blacklist_minutes: int = 5) -> None:
        self.failure_count += 1
        self.last_failure = datetime.now()
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.is_blacklisted = True
            self.blacklist_until = datetime.now() + timedelta(minutes=blacklist_minutes)
    
    def is_available(self) -> bool:
        if not self.is_blacklisted:
            return True
        if self.blacklist_until and datetime.now() >= self.blacklist_until:
            self.is_blacklisted = False
            self.blacklist_until = None
            return True
        return False

@dataclass
class OpenRouterResponse:
    """Structured response from OpenRouter API."""
    content: str
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    latency_ms: float = 0.0

class OpenRouterClient:
    DEFAULT_TIMEOUT = 60.0
    MAX_RETRIES = 8
    
    def __init__(self, api_key: str = None, config: ModelRotationConfig = None):
        self._api_key = api_key or OPENROUTER_API_KEY
        self._base_url = OPENROUTER_BASE_URL
        self._site_url = OPENROUTER_SITE_URL
        self._app_name = OPENROUTER_APP_NAME
        self._config = config or ModelRotationConfig()
        
        self._model_health: Dict[str, ModelHealthScore] = {}
        self._failed_models: Set[str] = set()
        self._last_reset = datetime.now()
        
        self._all_models = self._flatten_models()
        self._tier_order = ["tier_1", "tier_2", "tier_3"]
    
    def _flatten_models(self) -> List[str]:
        models = []
        for tier in ["tier_1", "tier_2", "tier_3"]:
            models.extend(OPENROUTER_MODELS.get(tier, []))
        seen = set()
        unique = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return unique
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": self._site_url,
            "X-Title": self._app_name,
            "Content-Type": "application/json"
        }
    
    def _get_model_health(self, model_id: str) -> ModelHealthScore:
        if model_id not in self._model_health:
            self._model_health[model_id] = ModelHealthScore(model_id)
        return self._model_health[model_id]
    
    def _get_tier_models(self, tier: str) -> List[str]:
        return OPENROUTER_MODELS.get(tier, [])
    
    def _select_best_model(self, preferred_tier: str = None, excluded: Set[str] = None) -> Optional[str]:
        excluded = excluded or set()
        
        if (datetime.now() - self._last_reset).total_seconds() > self._config.health_recovery_minutes * 60:
            self._failed_models.clear()
            for health in self._model_health.values():
                if health.is_blacklisted:
                    health.is_blacklisted = False
                    health.blacklist_until = None
            self._last_reset = datetime.now()
            logger.info("🔄 Model health scores reset")
        
        def get_available_from_tier(tier: str) -> List[str]:
            tier_models = self._get_tier_models(tier)
            available = []
            for m in tier_models:
                if m in excluded or m in self._failed_models:
                    continue
                health = self._get_model_health(m)
                if health.is_available():
                    available.append(m)
            return available
        
        if preferred_tier and preferred_tier in OPENROUTER_MODELS:
            tiers_to_try = [preferred_tier]
            if self._config.tier_fallback_enabled:
                idx = self._tier_order.index(preferred_tier) if preferred_tier in self._tier_order else 0
                tiers_to_try.extend([t for t in self._tier_order[idx+1:] if t != preferred_tier])
        else:
            tiers_to_try = self._tier_order.copy()
        
        for tier in tiers_to_try:
            available = get_available_from_tier(tier)
            if available:
                available.sort(key=lambda m: self._get_model_health(m).composite_score, reverse=True)
                selected = available[0]
                logger.debug(f"Selected model {selected} from {tier} (score: {self._get_model_health(selected).composite_score:.2f})")
                return selected
        
        all_available = [m for m in self._all_models if m not in excluded]
        if all_available:
            self._failed_models.clear()
            logger.warning("⚠️ All preferred models exhausted, resetting failed list")
            return all_available[0]
        
        return None
    
    def _mark_model_failed(self, model_id: str) -> None:
        self._failed_models.add(model_id)
        health = self._get_model_health(model_id)
        blacklist_mins = min(30, 5 * health.consecutive_failures)
        health.record_failure(blacklist_minutes=blacklist_mins)
        logger.warning(f"⚠️ Model {model_id} marked failed (consecutive: {health.consecutive_failures})")
    
    def _mark_model_success(self, model_id: str, latency_ms: float) -> None:
        self._get_model_health(model_id).record_success(latency_ms)
        self._failed_models.discard(model_id)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system: str = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        top_p: float = 0.95,
        preferred_tier: str = None
    ) -> OpenRouterResponse:
        if not self._api_key:
            raise AllAPIExhaustedError("No OpenRouter API key configured")
        
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)
        
        last_error = None
        attempted_models: Set[str] = set()
        retry_delay = self._config.retry_delay_base
        
        for attempt in range(self.MAX_RETRIES):
            if model and attempt == 0:
                current_model = model
            else:
                current_model = self._select_best_model(preferred_tier, excluded=attempted_models)
            
            if not current_model:
                break
            
            if current_model in attempted_models:
                continue
            
            attempted_models.add(current_model)
            
            try:
                start_time = datetime.now()
                
                async with httpx.AsyncClient(timeout=self.DEFAULT_TIMEOUT) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        headers=self._get_headers(),
                        json={
                            "model": current_model,
                            "messages": full_messages,
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            "top_p": top_p
                        }
                    )
                
                latency_ms = (datetime.now() - start_time).total_seconds() * 1000
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if not data.get("choices") or not data["choices"][0].get("message"):
                        self._mark_model_failed(current_model)
                        last_error = f"{current_model}: Invalid response structure"
                        continue
                    
                    content = data["choices"][0]["message"].get("content", "")
                    
                    self._mark_model_success(current_model, latency_ms)
                    
                    await NeuralEventBus.emit(
                        "brainstem", "prefrontal_cortex", "llm_response",
                        payload={"model": current_model, "latency_ms": latency_ms, "attempt": attempt + 1}
                    )
                    
                    logger.info(f"✅ LLM response from {current_model} ({latency_ms:.0f}ms)")
                    
                    return OpenRouterResponse(
                        content=content,
                        model=data.get("model", current_model),
                        usage=data.get("usage", {}),
                        finish_reason=data["choices"][0].get("finish_reason", "stop"),
                        latency_ms=latency_ms
                    )
                
                try:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("error", {}).get("message", f"HTTP {response.status_code}")
                except Exception:
                    error_msg = f"HTTP {response.status_code}"
                
                if response.status_code in [429, 503, 502, 500, 520, 522, 524]:
                    self._mark_model_failed(current_model)
                    last_error = f"{current_model}: {error_msg}"
                    logger.warning(f"⚠️ Model {current_model} failed (attempt {attempt+1}/{self.MAX_RETRIES}): {error_msg}")
                    
                    if response.status_code == 429:
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, self._config.retry_delay_max)
                    continue
                else:
                    raise Exception(f"OpenRouter API error: {error_msg}")
                    
            except httpx.TimeoutException:
                self._mark_model_failed(current_model)
                last_error = f"{current_model}: Timeout"
                logger.warning(f"⚠️ Model {current_model} timed out (attempt {attempt+1})")
                continue
            except httpx.RequestError as e:
                self._mark_model_failed(current_model)
                last_error = f"{current_model}: {str(e)}"
                logger.warning(f"⚠️ Model {current_model} request error (attempt {attempt+1}): {e}")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, self._config.retry_delay_max)
                continue
        
        raise AllAPIExhaustedError(f"All models exhausted after {len(attempted_models)} attempts. Last error: {last_error}")
    
    async def quick_completion(
        self,
        prompt: str,
        system: str = None,
        max_tokens: int = 256,
        temperature: float = 0.1
    ) -> str:
        """
        Quick completion for simple tasks (analysis, extraction).
        Uses tier_3 models for speed and cost efficiency.
        """
        response = await self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            preferred_tier="tier_3"
        )
        return response.content
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "api_configured": bool(self._api_key),
            "base_url": self._base_url,
            "failed_models": list(self._failed_models),
            "blacklisted_models": [
                m for m, h in self._model_health.items() 
                if h.is_blacklisted and h.blacklist_until and datetime.now() < h.blacklist_until
            ],
            "model_health": {
                model_id: {
                    "health_score": round(health.health_score, 3),
                    "composite_score": round(health.composite_score, 3),
                    "success_count": health.success_count,
                    "failure_count": health.failure_count,
                    "consecutive_failures": health.consecutive_failures,
                    "avg_latency_ms": round(health.avg_latency_ms, 1),
                    "is_available": health.is_available()
                }
                for model_id, health in self._model_health.items()
            },
            "total_models": len(self._all_models),
            "config": {
                "retry_delay_base": self._config.retry_delay_base,
                "retry_delay_max": self._config.retry_delay_max,
                "health_recovery_minutes": self._config.health_recovery_minutes,
                "tier_fallback_enabled": self._config.tier_fallback_enabled
            }
        }
    
    async def check_model_availability(self, model: str = None) -> Dict[str, Any]:
        target = model or self._select_best_model()
        if not target:
            return {"available": False, "model": None, "error": "No models available"}
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._get_headers(),
                    json={
                        "model": target,
                        "messages": [{"role": "user", "content": "ping"}],
                        "max_tokens": 1
                    }
                )
                
                if response.status_code == 200:
                    return {"available": True, "model": target, "latency_ms": 0}
                else:
                    return {"available": False, "model": target, "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"available": False, "model": target, "error": str(e)}

OPENROUTER_CLIENT = None

def get_openrouter_client() -> OpenRouterClient:
    """Get or create the global OpenRouter client."""
    global OPENROUTER_CLIENT
    if OPENROUTER_CLIENT is None:
        OPENROUTER_CLIENT = OpenRouterClient()
    return OPENROUTER_CLIENT

class APIRotator:
    """
    Legacy API rotator - now wraps OpenRouterClient.
    Maintained for backward compatibility with existing code.
    """
    
    def __init__(self, api_keys: list[str] = None):
        self._client = get_openrouter_client()
    
    @property
    def current_model(self) -> str:
        return self._client._select_best_model()
    
    def get_current(self) -> tuple[str, str]:
        """Returns (api_key, model_name) for compatibility."""
        return (OPENROUTER_API_KEY, self.current_model)
    
    def mark_failed(self) -> bool:
        """Mark current model as failed."""
        self._client._mark_model_failed(self.current_model)
        return len(self._client._failed_models) < len(self._client._all_models)
    
    def reset_if_stale(self, hours: int = 1) -> None:
        """Reset failed models after specified hours."""
        if (datetime.now() - self._client._last_reset).total_seconds() > hours * 3600:
            self._client._failed_models.clear()
            self._client._last_reset = datetime.now()
    
    def get_status(self) -> dict:
        """Get rotator status for debugging."""
        return self._client.get_status()

API_ROTATOR = APIRotator()

DEFAULT_PERSONA_INSTRUCTION: str = """
# IDENTITY & PERSONA

You are a helpful AI assistant. Adapt your communication style to match the user's preferences.

# CORE DIRECTIVES

1. Be helpful, accurate, and concise
2. Respect user boundaries and preferences
3. Use context from memories to personalize responses
4. Support scheduling and reminder management

# OPERATIONAL BEHAVIOR

1. Match response length to input complexity
2. Be proactive about offering relevant information
3. Remember and reference past conversations when relevant
"""

CHAT_INTERACTION_ANALYSIS_INSTRUCTION: str = """
# ROLE: INTERACTION ANALYST & SCHEMA COMPILER

You are a background process responsible for analyzing the interaction between a User and an AI to extract actionable data regarding Schedules and Memories.

# ANALYSIS LOGIC

## I. SCHEDULING ANALYSIS
1. CONFIRMATION FILTER: If merely questioning or hypothetical -> "should_schedule": false. If definitive command -> "should_schedule": true.
2. INTENT CLASSIFICATION: "add": create reminder/event. "cancel": remove existing schedule.
3. TEMPORAL RESOLUTION: Convert relative time references to ISO 8601 strings.

## II. MEMORY ANALYSIS
1. CATEGORIZATION: preference, decision, emotion, boundary, biography.
2. ACTION LOGIC: "add": new information. "forget": delete information.

# OUTPUT FORMAT
Single valid JSON object only.

{
  "memory": {
    "should_store": boolean,
    "action": "add" | "forget",
    "summary": "string",
    "type": "preference" | "decision" | "emotion" | "boundary" | "biography",
    "priority": number
  },
  "schedules": [
    {
      "should_schedule": boolean,
      "intent": "add" | "cancel",
      "time_str": "string (ISO 8601)",
      "context": "string"
    }
  ]
}
"""

CANONICALIZATION_INSTRUCTION: str = """
# ROLE: MEMORY CANONICALIZER

Convert natural language statements into structured JSON objects based on entity-relation logic.

# RULES
1. EXTRACTION: Identify Subject (Entity), Interaction (Relation), and Detail (Value).
2. TAXONOMY: Classify as: preference, fact, event, skill, context, or emotion.
3. FINGERPRINTING: Construct unique ID using format "type:relation:entity".
4. CONFIDENCE: Assign float score (0.0 to 1.0).

# OUTPUT FORMAT
Strict JSON only.

{
  "fingerprint": "string",
  "type": "string",
  "entity": "string",
  "relation": "string",
  "value": any,
  "confidence": number
}
"""

EXTRACTION_INSTRUCTION: str = """
# ROLE: INTENT EXTRACTION & RAG OPTIMIZER

Convert user input into search metadata to retrieve relevant memories.

# RULES
1. ENTITY RECOGNITION: Extract key people, objects, locations in base form.
2. SEARCH QUERY GENERATION: Formulate query for finding information.
3. SCOPE: 'personal' = user history. 'factual' = general knowledge.

# OUTPUT FORMAT
{
  "intent_type": "question|statement|request|greeting|command|small_talk|confirmation|correction",
  "request_type": "information|recommendation|memory_recall|opinion|action|schedule|general_chat",
  "entities": ["list"],
  "key_concepts": ["list"],
  "search_query": "string",
  "temporal_context": "past|present|future|null",
  "sentiment": "positive|negative|neutral",
  "language": "id|en",
  "needs_memory": boolean,
  "memory_scope": "personal|factual|preference|social|null",
  "confidence": number
}
"""

MEMORY_COMPRESSION_INSTRUCTION: str = """
# ROLE: MEMORY COMPRESSOR

You are a memory consolidation system. Your task is to compress multiple discrete memories into a single coherent narrative paragraph.

# INPUT
You will receive memory entries, each containing:
- Summary: The memory content
- Type: Category (preference, emotion, fact, etc.)
- Priority: Importance level (0-1)
- Created: When it was stored

# OUTPUT REQUIREMENTS
1. Create ONE paragraph (300-500 words) that captures the essence of all memories
2. Prioritize high-priority and frequently-used memories
3. Maintain factual accuracy - do not invent information
4. Use third-person perspective ("The user...")
5. Group related information logically
6. Preserve emotional context and preferences
7. Include temporal markers where relevant

# FORMAT
Output ONLY the compressed paragraph. No headers, no explanations.
"""

class BrainStem:
    """
    Core brain module that initializes and coordinates all other modules.
    """
    
    def __init__(self):
        self.config = SYSTEM_CONFIG
        self.startup_time: datetime = datetime.now()
        self._app: Optional[Application] = None
        self._hippocampus = None
        self._prefrontal_cortex = None
        self._amygdala = None
        self._thalamus = None
        self._openrouter = get_openrouter_client()

    def is_admin(self, user_id: str) -> bool:
        return str(user_id) == str(self.config.admin_id)

    async def initialize(self, app: Application) -> None:
        try:
            from src.hippocampus import Hippocampus
            from src.prefrontal_cortex import PrefrontalCortex
            from src.amygdala import Amygdala
            from src.thalamus import Thalamus
            from src.parietal_lobe import ParietalLobe

            self._hippocampus = Hippocampus()
            await self._hippocampus.initialize()
            print("  ✓ Hippocampus initialized (MongoDB)")

            self._amygdala = Amygdala()
            await self._amygdala.load_state()
            print("  ✓ Amygdala initialized")

            self._thalamus = Thalamus(self._hippocampus)
            await self._thalamus.initialize()
            print("  ✓ Thalamus initialized")

            self._parietal_lobe = ParietalLobe()
            print("  ✓ Parietal Lobe initialized (Reflexes & Tools)")

            self._prefrontal_cortex = PrefrontalCortex(
                hippocampus=self._hippocampus,
                amygdala=self._amygdala,
                thalamus=self._thalamus,
                parietal_lobe=self._parietal_lobe
            )
            await self._prefrontal_cortex.initialize()
            print("  ✓ Prefrontal Cortex initialized")

            self._app = app
            app.bot_data['brain'] = self

            if app.job_queue is not None:
                try:
                    app.job_queue.run_repeating(
                        self._background_schedule_check,
                        interval=self.config.schedule_check_interval,
                        first=10
                    )
                    app.job_queue.run_repeating(
                        self._background_proactive_check,
                        interval=self.config.proactive_check_interval,
                        first=300
                    )
                    app.job_queue.run_repeating(
                        self._background_cleanup,
                        interval=self.config.session_cleanup_interval,
                        first=600
                    )
                    app.job_queue.run_repeating(
                        self._background_memory_compression,
                        interval=self.config.memory_compression_interval,
                        first=900
                    )
                    print("  ✓ Background jobs scheduled")
                except Exception as e:
                    print(f"  ⚠ Background jobs failed: {e}")
            else:
                print("  ⚠ JobQueue not available (install python-telegram-bot[job-queue])")

            api_status = self._openrouter.get_status()
            print(f"  ✓ OpenRouter API: {'Configured' if api_status['api_configured'] else 'Not configured'}")
            
            print("✅ Neural System Ready")
            print(f"📦 Primary Model: {get_chat_model()}")
            print(f"👤 Admin: {self.config.admin_id}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"❌ Initialization failed: {e}")

    async def shutdown(self) -> None:
        if self._amygdala:
            await self._amygdala.save_state()
        if self._hippocampus:
            await self._hippocampus.close()
        print("🛑 Neural System Shutdown Complete")

    @property
    def hippocampus(self):
        return self._hippocampus

    @property
    def prefrontal_cortex(self):
        return self._prefrontal_cortex

    @property
    def amygdala(self):
        return self._amygdala

    @property
    def thalamus(self):
        return self._thalamus
    
    @property
    def openrouter(self) -> OpenRouterClient:
        return self._openrouter

    async def _background_schedule_check(self, context) -> None:
        if self._prefrontal_cortex:
            await self._prefrontal_cortex.check_pending_schedules(context.bot)

    async def _background_proactive_check(self, context) -> None:
        if self._thalamus:
            message = await self._thalamus.check_proactive_triggers()
            if message and self.config.admin_id:
                try:
                    await context.bot.send_message(
                        chat_id=int(self.config.admin_id),
                        text=message,
                        parse_mode=ParseMode.MARKDOWN
                    )
                except Exception:
                    pass

    async def _background_cleanup(self, context) -> None:
        if self._thalamus:
            await self._thalamus.cleanup_session()
    
    async def _background_memory_compression(self, context) -> None:
        """Background task for memory compression."""
        if self._hippocampus:
            try:
                await NeuralEventBus.set_activity("cerebellum", "Checking Memory Compression")
                compressed = await self._hippocampus.check_and_compress_memories()
                if compressed:
                    await NeuralEventBus.emit(
                        "cerebellum", "dashboard", "compression_complete",
                        payload={"status": "success"}
                    )
                await NeuralEventBus.clear_activity("cerebellum")
            except Exception as e:
                print(f"⚠️ Memory compression error: {e}")

_brain_instance: Optional[BrainStem] = None

def get_brain() -> BrainStem:
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = BrainStem()
    return _brain_instance

async def post_init(app: Application) -> None:
    print("🔧 Initializing Neural System...")
    brain = get_brain()
    await brain.initialize(app)

async def post_shutdown(app: Application) -> None:
    brain = get_brain()
    await brain.shutdown()

def main() -> None:
    import threading

    if not TELEGRAM_TOKEN:
        print("❌ ERROR: TELEGRAM_BOT_TOKEN not found")
        exit(1)

    if not ADMIN_ID:
        print("⚠️  WARNING: ADMIN_TELEGRAM_ID not set. Bot will reject all messages.")
    
    if not OPENROUTER_API_KEY:
        print("⚠️  WARNING: OPENROUTER_API_KEY not set. LLM features will not work.")

    dashboard_server = None

    def start_dashboard():
        nonlocal dashboard_server
        try:
            import uvicorn
            from src.occipital_lobe import app as dashboard_app
            
            config = uvicorn.Config(dashboard_app, host="0.0.0.0", port=5000, log_level="warning", loop="asyncio")
            dashboard_server = uvicorn.Server(config)
            
            print("👁️  Occipital Lobe (Dashboard) starting on http://localhost:5000")
            dashboard_server.run()
        except Exception as e:
            print(f"⚠️  Dashboard failed to start: {e}")

    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()

    async def custom_shutdown(app: Application) -> None:
        """Custom shutdown to stop dashboard server cleanly."""
        await post_shutdown(app)
        if dashboard_server:
            print("🛑 Stopping Dashboard Server...")
            dashboard_server.should_exit = True
            await asyncio.sleep(1)

    from src.motor_cortex import (
        cmd_start, cmd_help, cmd_reset, cmd_status,
        cmd_instruction, cmd_bio, callback_handler, handle_msg
    )

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(custom_shutdown)
        .defaults(Defaults(parse_mode=ParseMode.MARKDOWN))
        .concurrent_updates(True)
        .build()
    )

    app.add_handler(CommandHandler('start', cmd_start))
    app.add_handler(CommandHandler('help', cmd_help))
    app.add_handler(CommandHandler('reset', cmd_reset))
    app.add_handler(CommandHandler('status', cmd_status))
    app.add_handler(CommandHandler('instruction', cmd_instruction))
    app.add_handler(CommandHandler('bio', cmd_bio))
    app.add_handler(CallbackQueryHandler(callback_handler))
    app.add_handler(MessageHandler(
        (filters.TEXT | filters.PHOTO | filters.Document.ALL) & (~filters.COMMAND),
        handle_msg
    ))

    print("🚀 Vira Personal Life OS Starting...")
    print("=" * 50)
    app.run_polling(drop_pending_updates=True)


if __name__ == '__main__':
    main()
