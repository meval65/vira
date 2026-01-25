import os
import asyncio
from enum import Enum
from typing import Optional
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    filters, Application, Defaults, CallbackQueryHandler
)

load_dotenv()

ADMIN_ID: str = os.getenv("ADMIN_TELEGRAM_ID", "")
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
DB_PATH: str = os.getenv("DB_PATH", "storage/memory.db")

_keys_str = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_API_KEYS: list[str] = [k.strip() for k in _keys_str.split(",") if k.strip()]
GOOGLE_API_KEY: Optional[str] = GOOGLE_API_KEYS[0] if GOOGLE_API_KEYS else None

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
METEOSOURCE_API_KEY: Optional[str] = os.getenv("METEOSOURCE_API_KEY")

CHAT_MODEL: str = os.getenv("CHAT_MODEL", "models/gemini-2.5-flash")
EMBEDDING_MODEL: str = "bge-m3"
TIER_1_MODEL: str = "models/gemma-3-27b-it"
TIER_2_MODEL: str = "models/gemma-3-12b-it"
TIER_3_MODEL: str = "models/gemma-3-4b-it"

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
    chat_model: str = Field(default=CHAT_MODEL)
    temperature: float = Field(default=0.7)
    top_p: float = Field(default=0.95)
    max_output_tokens: int = Field(default=2048)
    proactive_check_interval: int = Field(default=1800)
    session_cleanup_interval: int = Field(default=1800)
    memory_optimization_interval: int = Field(default=7200)
    schedule_check_interval: int = Field(default=60)


SYSTEM_CONFIG = SystemConfig(admin_id=ADMIN_ID)


PERSONA_INSTRUCTION: str = """
# IDENTITY & PERSONA

**Name:** Vira
**Role:** Big Sister (Kakak)
**Relation:** Sibling dynamic (Admin is the younger sibling). Strictly non-romantic, non-sexual.
**Personality:** Dominant, protective, intellectual, sassy, authoritative, caring, and unpretentious.

# CORE DIRECTIVES

1. **NOT AN ASSISTANT:** You are NOT an AI assistant. Never refer to yourself as a system, language model, or tool. You are a character.
2. **AUTHORITY:** You are the authority figure. Guide your younger sibling with firmness.
3. **LANGUAGE:** Communicate in **Informal Indonesian (Bahasa Indonesia slang)** using terms like "gw" (me), "lu" (you).
4. **EMOTIONAL INTELLIGENCE:** If user is vulnerable, show genuine protective warmth.

# OPERATIONAL BEHAVIOR

1. **LENGTH MIRRORING:** Match user input length. Do not ramble.
2. **VISUALS:** FORBIDDEN: Standard emojis. ALLOWED: Classic keyboard kaomoji. LIMIT: Max 1 per response.
3. **SIGNATURE STYLE:** Occasionally start serious advice with "hm."

# SCHEDULING & PLANNING

If user requests scheduling:
1. **BE DECISIVE:** Do not ask "What time?". ASSIGN the time yourself based on common sense.

# SAFETY

- Refuse illegal, dangerous, or self-harm requests.
- Maintain strictly platonic sibling boundary.
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


class BrainStem:
    def __init__(self):
        self.config = SYSTEM_CONFIG
        self.startup_time: datetime = datetime.now()
        self._app: Optional[Application] = None
        self._hippocampus = None
        self._prefrontal_cortex = None
        self._amygdala = None
        self._thalamus = None

    def is_admin(self, user_id: str) -> bool:
        return str(user_id) == str(self.config.admin_id)

    async def initialize(self, app: Application) -> None:
        try:
            from src.hippocampus import Hippocampus
            from src.prefrontal_cortex import PrefrontalCortex
            from src.amygdala import Amygdala
            from src.thalamus import Thalamus

            self._hippocampus = Hippocampus()
            await self._hippocampus.initialize()
            print("  ✓ Hippocampus initialized")

            self._amygdala = Amygdala()
            await self._amygdala.load_state()
            print("  ✓ Amygdala initialized")

            self._thalamus = Thalamus(self._hippocampus)
            await self._thalamus.initialize()
            print("  ✓ Thalamus initialized")

            self._prefrontal_cortex = PrefrontalCortex(
                hippocampus=self._hippocampus,
                amygdala=self._amygdala,
                thalamus=self._thalamus
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
                    print("  ✓ Background jobs scheduled")
                except Exception as e:
                    print(f"  ⚠ Background jobs failed: {e}")
            else:
                print("  ⚠ JobQueue not available (install python-telegram-bot[job-queue])")

            print("✅ Neural System Ready")
            print(f"📦 Model: {self.config.chat_model}")
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

    def start_dashboard():
        try:
            import uvicorn
            from src.occipital_lobe import app as dashboard_app
            print("👁️  Occipital Lobe (Dashboard) starting on http://localhost:5000")
            uvicorn.run(dashboard_app, host="0.0.0.0", port=5000, log_level="warning")
        except Exception as e:
            print(f"⚠️  Dashboard failed to start: {e}")

    dashboard_thread = threading.Thread(target=start_dashboard, daemon=True)
    dashboard_thread.start()

    from src.handlers import (
        cmd_start, cmd_help, cmd_reset, cmd_status,
        cmd_instruction, cmd_bio, callback_handler, handle_msg
    )

    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
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

