import os
import asyncio
import json
import math
from enum import Enum
from typing import Optional, Dict, List, Set, Tuple, Any, Callable
from datetime import datetime, timedelta, time as dt_time
from dataclasses import dataclass, field
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, Application, Defaults, CallbackQueryHandler
)
from telegram.request import HTTPXRequest
import httpx
import logging
from functools import lru_cache
from collections import deque

from src.brain.infrastructure.neural_event_bus import (
    NeuralEventBus,
    NeuralEventBusRedis,
    init_event_bus,
    get_event_bus
)
from src.brain.infrastructure.redis_client import init_redis, close_redis

load_dotenv()

ADMIN_ID: str = os.getenv("ADMIN_TELEGRAM_ID", "")
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
DB_PATH: str = os.getenv("DB_PATH", "storage/memory.db")
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_SITE_URL: str = os.getenv("OPENROUTER_SITE_URL", "https://vira-os.local")
OPENROUTER_APP_NAME: str = os.getenv("OPENROUTER_APP_NAME", "Vira Personal Life OS")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "embeddinggemma")
METEOSOURCE_API_KEY: Optional[str] = os.getenv("METEOSOURCE_API_KEY")

OPENROUTER_MODELS: Dict[str, List[str]] = {
    "chat_model": [
        "deepseek/deepseek-v3.2",
        "tngtech/deepseek-r1t2-chimera:free",
        "openai/gpt-oss-120b:free",
    ],
    "analysis_model": [
        "tngtech/deepseek-r1t2-chimera:free",
        "openai/gpt-oss-120b:free",
    ],
    "utility_model": [
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "openai/gpt-oss-120b:free",
    ]
}

@lru_cache(maxsize=1)
def get_main_chat_model() -> str:
    return OPENROUTER_MODELS.get("chat_model", ["nvidia/nemotron-3-nano-30b-a3b:free"])[0]

@lru_cache(maxsize=1)
def get_analysis_model() -> str:
    return OPENROUTER_MODELS.get("analysis_model", ["nvidia/nemotron-3-nano-30b-a3b:free"])[0]

@lru_cache(maxsize=1)
def get_utility_model() -> str:
    return OPENROUTER_MODELS.get("utility_model", ["nvidia/nemotron-3-nano-30b-a3b:free"])[0]

EMBEDDING_DIMENSION: int = 1024
MAX_RETRIEVED_MEMORIES: int = 3
MIN_RELEVANCE_SCORE: float = 0.82
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
    chat_model: str = Field(default_factory=get_analysis_model)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_output_tokens: int = Field(default=512, ge=1)
    proactive_check_interval: int = Field(default=1800, ge=60)
    session_cleanup_interval: int = Field(default=1800, ge=300)
    memory_optimization_interval: int = Field(default=7200, ge=600)
    schedule_check_interval: int = Field(default=60, ge=10)
    memory_compression_interval: int = Field(default=1800, ge=300)

SYSTEM_CONFIG = SystemConfig(admin_id=ADMIN_ID)


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
        
        recency_score = 1.0
        if self.last_failure:
            hours_since_failure = (datetime.now() - self.last_failure).total_seconds() / 3600
            recency_score = min(1.0, hours_since_failure / 24)
        
        latency_score = 1.0
        if self.success_count > 0:
            avg_latency = self.total_latency_ms / self.success_count
            if avg_latency < 1000:
                latency_score = 1.0
            elif avg_latency < 3000:
                latency_score = 0.8
            else:
                latency_score = 0.6
        
        return (success_rate * 0.6) + (recency_score * 0.2) + (latency_score * 0.2)
    
    def should_skip(self, config: ModelRotationConfig) -> bool:
        if self.is_blacklisted and self.blacklist_until:
            if datetime.now() < self.blacklist_until:
                return True
            else:
                self.is_blacklisted = False
                self.blacklist_until = None
                self.consecutive_failures = 0
        
        return self.consecutive_failures >= config.max_consecutive_failures
    
    def record_success(self, latency_ms: float) -> None:
        self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_success = datetime.now()
        self.consecutive_failures = 0
    
    def record_failure(self, config: ModelRotationConfig) -> None:
        self.failure_count += 1
        self.last_failure = datetime.now()
        self.consecutive_failures += 1
        
        if self.consecutive_failures >= config.max_consecutive_failures:
            self.is_blacklisted = True
            self.blacklist_until = datetime.now() + timedelta(minutes=config.health_recovery_minutes)

class OpenRouterClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        config: Optional[ModelRotationConfig] = None
    ):
        self.api_key = api_key or OPENROUTER_API_KEY
        self.base_url = base_url
        self.config = config or ModelRotationConfig()
        self.health_scores: Dict[str, ModelHealthScore] = {}
        self.tier_preference_order = ["chat_model", "analysis_model", "utility_model"]
        self._client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self) -> None:
        await self._client.aclose()
    
    def get_status(self) -> Dict[str, Any]:
        return {
            "api_configured": bool(self.api_key),
            "base_url": self.base_url,
            "health_scores": {
                model_id: {
                    "score": health.health_score,
                    "success": health.success_count,
                    "failure": health.failure_count,
                    "blacklisted": health.is_blacklisted
                }
                for model_id, health in self.health_scores.items()
            }
        }
    
    def _get_health_score(self, model_id: str) -> ModelHealthScore:
        if model_id not in self.health_scores:
            self.health_scores[model_id] = ModelHealthScore(model_id)
        return self.health_scores[model_id]
    
    def _get_tier_models(self, tier: str) -> List[str]:
        return OPENROUTER_MODELS.get(tier, [])
    
    def _get_all_models_in_tier_order(self) -> List[Tuple[str, str]]:
        models = []
        for tier in self.tier_preference_order:
            for model in self._get_tier_models(tier):
                models.append((tier, model))
        return models
    
    def _select_best_model(self, tier: Optional[str] = None) -> Optional[str]:
        if tier:
            candidates = self._get_tier_models(tier)
        else:
            candidates = [model for _, model in self._get_all_models_in_tier_order()]
        
        available = [
            model for model in candidates
            if not self._get_health_score(model).should_skip(self.config)
        ]
        
        if not available:
            return None
        
        best_model = max(
            available,
            key=lambda m: self._get_health_score(m).health_score
        )
        
        return best_model
    
    async def _make_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stream: bool = False,
        extra_body: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": OPENROUTER_SITE_URL,
            "X-Title": OPENROUTER_APP_NAME,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        if extra_body:
            payload.update(extra_body)
        
        start_time = datetime.now()
        
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            )
            
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if response.status_code == 200:
                self._get_health_score(model).record_success(latency_ms)
                return response.json()
            else:
                self._get_health_score(model).record_failure(self.config)
                logger.error(f"API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            self._get_health_score(model).record_failure(self.config)
            logger.error(f"Request failed for {model}: {e}")
            return None
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        tier: Optional[str] = None,
        stream: bool = False,
        extra_body: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        
        if tier:
            model = self._select_best_model(tier)
            if model:
                result = await self._make_request(
                    model, messages, temperature, max_tokens, top_p, stream, extra_body
                )
                if result:
                    return result
            
            if not self.config.tier_fallback_enabled:
                raise AllAPIExhaustedError(f"All models in {tier} exhausted")
        
        for attempt_tier, model in self._get_all_models_in_tier_order():
            health = self._get_health_score(model)
            if health.should_skip(self.config):
                continue
            
            result = await self._make_request(
                model, messages, temperature, max_tokens, top_p, stream, extra_body
            )
            
            if result:
                return result
            
            delay = min(
                self.config.retry_delay_base * (2 ** health.consecutive_failures),
                self.config.retry_delay_max
            )
            await asyncio.sleep(delay)
        
        raise AllAPIExhaustedError("All models exhausted across all tiers")

    async def quick_completion(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 256,
        system: Optional[str] = None,
        tier: Optional[str] = "utility_model",
        json_mode: bool = False
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        extra_body = None
        if json_mode:
            extra_body = {"response_format": {"type": "json_object"}}
        
        result = await self.chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
            extra_body=extra_body
        )
        
        if result and "choices" in result and result["choices"]:
            return result["choices"][0]["message"]["content"]
        return ""

@dataclass
class BrainStem:
    config: SystemConfig = field(default_factory=lambda: SYSTEM_CONFIG)
    
    def __post_init__(self):
        self._hippocampus = None
        self._prefrontal_cortex = None
        self._amygdala = None
        self._thalamus = None
        self._parietal_lobe = None
        self._app = None
        self._openrouter = OpenRouterClient()
        self._shutdown_event = asyncio.Event()
        self._event_bus: Optional[NeuralEventBusRedis] = None
        self._redis_initialized = False
        self._consolidator = None
    
    @property
    def hippocampus(self):
        return self._hippocampus

    @property
    def amygdala(self):
        return self._amygdala

    @property
    def thalamus(self):
        return self._thalamus

    @property
    def parietal_lobe(self):
        return self._parietal_lobe

    @property
    def prefrontal_cortex(self):
        return self._prefrontal_cortex

    async def initialize(self, app: Application) -> None:
        try:
            from src.brain.hippocampus import Hippocampus
            from src.brain.prefrontal_cortex import PrefrontalCortex
            from src.brain.amygdala import Amygdala
            from src.brain.thalamus import Thalamus
            from src.brain.parietal_lobe import ParietalLobe
            from src.brain.db.mongo_client import get_mongo_client

            try:
                await init_redis()
                mongo = get_mongo_client()
                self._event_bus = await init_event_bus(mongo)
                self._redis_initialized = True
                logger.info("Redis Event Bus initialized")
            except Exception as e:
                logger.warning(f"Redis unavailable, using in-memory fallback: {e}")
                self._event_bus = get_event_bus()
                self._redis_initialized = False

            self._hippocampus = Hippocampus()
            self._amygdala = Amygdala()
            self._thalamus = Thalamus()
            self._parietal_lobe = ParietalLobe()
            self._prefrontal_cortex = PrefrontalCortex()

            self._hippocampus.bind_brain(self)
            self._amygdala.bind_brain(self)
            self._thalamus.bind_brain(self)
            self._parietal_lobe.bind_brain(self)
            self._prefrontal_cortex.bind_brain(self)

            await self._hippocampus.initialize()
            logger.info("Hippocampus initialized (MongoDB)")

            await self._amygdala.load_state()
            logger.info("Amygdala initialized")

            await self._thalamus.initialize()
            logger.info("Thalamus initialized")

            logger.info("Parietal Lobe initialized (Reflexes & Tools + CRUD)")

            await self._prefrontal_cortex.initialize()
            logger.info("Prefrontal Cortex initialized")

            self._app = app
            app.bot_data['brain'] = self

            if app.job_queue is not None:
                self._schedule_background_jobs(app)
            else:
                logger.warning("JobQueue not available (install python-telegram-bot[job-queue])")

            api_status = self._openrouter.get_status()
            logger.info(f"OpenRouter API: {'Configured' if api_status['api_configured'] else 'Not configured'}")
            
            logger.info("Neural System Ready")
            logger.info(f"Primary Model: {get_main_chat_model()}")
            logger.info(f"Admin: {self.config.admin_id}")
            
            event_bus_status = self._event_bus.get_status() if self._event_bus else {}
            logger.info(f"Event Bus: Redis={'connected' if event_bus_status.get('redis_connected') else 'fallback'}")
            
            try:
                from src.brain.consolidation.nocturnal_consolidator import NocturnalConsolidator
                self._consolidator = NocturnalConsolidator(
                    hippocampus=self._hippocampus,
                    openrouter_client=self._openrouter,
                    mongo_client=mongo
                )
                logger.info("Nocturnal Consolidator initialized")
            except Exception as e:
                logger.warning(f"Consolidator initialization failed: {e}")
        except Exception as e:
            logger.exception(f"Initialization failed: {e}")
            raise

    def _schedule_background_jobs(self, app: Application) -> None:
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
            
            app.job_queue.run_daily(
                self._background_consolidation,
                time=dt_time(hour=3, minute=0, second=0)
            )
            
            logger.info("Background jobs scheduled (including 3 AM consolidation)")
        except Exception as e:
            logger.error(f"Background jobs failed: {e}")

    async def shutdown(self) -> None:
        self._shutdown_event.set()
        
        if self._amygdala:
            await self._amygdala.save_state()
        if self._hippocampus:
            await self._hippocampus.close()
        if self._openrouter:
            await self._openrouter.close()
        
        if self._event_bus:
            await self._event_bus.shutdown()
        await close_redis()
        
        logger.info("Neural System Shutdown Complete")

    @property
    def openrouter(self) -> OpenRouterClient:
        return self._openrouter

    async def _background_schedule_check(self, context) -> None:
        if self._prefrontal_cortex and not self._shutdown_event.is_set():
            try:
                await self._prefrontal_cortex.check_pending_schedules(context.bot)
            except Exception as e:
                logger.error(f"Schedule check error: {e}")

    async def _background_proactive_check(self, context) -> None:
        if self._thalamus and not self._shutdown_event.is_set():
            try:
                message = await self._thalamus.check_proactive_triggers()
                if message and self.config.admin_id:
                    await context.bot.send_message(
                        chat_id=int(self.config.admin_id),
                        text=message,
                        parse_mode=ParseMode.MARKDOWN
                    )
            except Exception as e:
                logger.error(f"Proactive check error: {e}")

    async def _background_cleanup(self, context) -> None:
        if self._thalamus and not self._shutdown_event.is_set():
            try:
                await self._thalamus.cleanup_session()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    async def _background_memory_compression(self, context) -> None:
        if self._hippocampus and not self._shutdown_event.is_set():
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
                logger.error(f"Memory compression error: {e}")
    
    async def _background_consolidation(self, context) -> None:
        if self._consolidator and not self._shutdown_event.is_set():
            try:
                await NeuralEventBus.set_activity("cerebellum", "Running Nocturnal Consolidation")
                result = await self._consolidator.run_consolidation(force=True)
                
                if result.success:
                    await NeuralEventBus.emit(
                        "cerebellum", "dashboard", "consolidation_complete",
                        payload={
                            "memories_created": result.memories_created,
                            "kg_triples": result.kg_triples_created,
                            "duration": result.duration_seconds
                        }
                    )
                    logger.info(
                        f"Consolidation complete: {result.memories_created} memories, "
                        f"{result.kg_triples_created} KG triples"
                    )
                else:
                    logger.warning(f"Consolidation failed: {result.error}")
                    
                await NeuralEventBus.clear_activity("cerebellum")
            except Exception as e:
                logger.error(f"Consolidation error: {e}")

_global_brain: Optional[BrainStem] = None
_brain_init_lock = asyncio.Lock()

async def get_brain() -> BrainStem:
    global _global_brain
    if _global_brain is None:
        async with _brain_init_lock:
            if _global_brain is None:
                _global_brain = BrainStem()
    return _global_brain

async def post_init(app: Application) -> None:
    logger.info("Initializing Neural System...")
    brain = await get_brain()
    await brain.initialize(app)

async def post_shutdown(app: Application) -> None:
    brain = await get_brain()
    await brain.shutdown()

def main() -> None:
    import threading

    from src.brain.utils.logging_config import setup_logging
    setup_logging(level=logging.ERROR)

    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_BOT_TOKEN not found")
        exit(1)

    if not ADMIN_ID:
        logger.warning("ADMIN_TELEGRAM_ID not set. Bot will reject all messages.")
    
    if not OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. LLM features will not work.")

    dashboard_server = None

    def start_dashboard():
        nonlocal dashboard_server
        try:
            import uvicorn
            from src.brain.occipital_lobe import app as dashboard_app
            
            config = uvicorn.Config(
                dashboard_app, 
                host="0.0.0.0", 
                port=5000, 
                log_level="warning", 
                loop="asyncio"
            )
            dashboard_server = uvicorn.Server(config)
            
            logger.info("Occipital Lobe (Dashboard) starting on http://localhost:5000")
            dashboard_server.run()
        except Exception as e:
            logger.error(f"Dashboard failed to start: {e}")

    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()

    async def custom_shutdown(app: Application) -> None:
        await post_shutdown(app)
        if dashboard_server:
            logger.info("Stopping Dashboard Server...")
            dashboard_server.should_exit = True
            await asyncio.sleep(1)

    from src.brain.motor_cortex import (
        cmd_start, cmd_help, cmd_reset, cmd_status,
        cmd_instruction, cmd_bio, callback_handler, handle_msg
    )

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(custom_shutdown)
        .request(HTTPXRequest(connect_timeout=20.0, read_timeout=20.0))
        .get_updates_request(HTTPXRequest(connect_timeout=20.0, read_timeout=20.0))
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

    print("Vira is running. Dashboard: http://localhost:5000  (Ctrl+C to stop)")
    app.run_polling(drop_pending_updates=True)

if __name__ == '__main__':
    main()