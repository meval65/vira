import os
import json
import asyncio
import re
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

import httpx
import PIL.Image
from google import genai
from google.genai import types
from dateutil import parser as date_parser

from src.brainstem import (
    GOOGLE_API_KEY, CHAT_MODEL, TIER_1_MODEL, TIER_2_MODEL, TIER_3_MODEL,
    PERSONA_INSTRUCTION, CHAT_INTERACTION_ANALYSIS_INSTRUCTION,
    EXTRACTION_INSTRUCTION, MemoryType, API_ROTATOR, AllAPIExhaustedError,
    OLLAMA_BASE_URL, EMBEDDING_MODEL, NeuralEventBus
)
from src.hippocampus import Hippocampus, Memory
from src.amygdala import Amygdala, PlanProgressState
from src.thalamus import Thalamus

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


class PrefrontalCortex:
    MAX_OUTPUT_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    INTENT_CACHE_TTL: int = 300

    def __init__(
        self,
        hippocampus: Hippocampus,
        amygdala: Amygdala,
        thalamus: Thalamus
    ):
        self._hippocampus = hippocampus
        self._amygdala = amygdala
        self._thalamus = thalamus
        self._client: Optional[genai.Client] = None
        self._active_plan: Optional[TaskPlan] = None
        self._intent_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._processing = False

        amygdala.set_hippocampus(hippocampus)

    async def initialize(self) -> None:
        if GOOGLE_API_KEY:
            self._client = genai.Client(api_key=GOOGLE_API_KEY)
        await self._load_active_plan()

    async def _load_active_plan(self) -> None:
        pass

    async def process(
        self,
        message: str,
        image_path: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> str:
        if self._processing:
            return "⏳ Sedang memproses pesan sebelumnya..."

        self._processing = True
        try:
            if user_name:
                await self._hippocampus.update_admin_profile(telegram_name=user_name)

            # Generate embedding for user message
            user_embedding = await self._generate_embedding(message)

            await NeuralEventBus.set_activity("prefrontal_cortex", "Analyzing Intent")
            intent_data = await self._extract_intent(message)
            
            await NeuralEventBus.emit("prefrontal_cortex", "amygdala", "emotion_check")
            detected_emotion = self._amygdala.detect_emotion_from_text(message)
            self._amygdala.adjust_for_emotion(detected_emotion)

            await NeuralEventBus.set_activity("prefrontal_cortex", "Retrieving Memories")
            await NeuralEventBus.emit("prefrontal_cortex", "hippocampus", "memory_retrieval")
            memories = await self._retrieve_memories(message, intent_data)

            if self._active_plan:
                await NeuralEventBus.set_activity("prefrontal_cortex", "Executing Plan")
                plan_response = await self._handle_active_plan(message, intent_data)
                if plan_response:
                    await NeuralEventBus.clear_activity("prefrontal_cortex")
                    return plan_response

            if self._should_create_plan(message, intent_data):
                await NeuralEventBus.set_activity("prefrontal_cortex", "Creating New Plan")
                plan = await self.create_plan(message)
                if plan:
                    self._amygdala.update_satisfaction(PlanProgressState.ON_TRACK)

            schedules = await self._hippocampus.get_upcoming_schedules(hours_ahead=24)
            schedule_context = self._format_schedules(schedules) if schedules else None

            # Retrieve semantically relevant history via hybrid context
            await NeuralEventBus.set_activity("prefrontal_cortex", "Building Context")
            await NeuralEventBus.emit("prefrontal_cortex", "thalamus", "context_building")
            
            # Get long-term relevant history via async method
            relevant_history = await self._thalamus.get_relevant_history_async(user_embedding, top_k=5)
            relevant_history_context = self._format_relevant_history(relevant_history)

            context = await self._thalamus.build_context(
                relevant_memories=[asdict(m) for m in memories] if memories else [],
                schedule_context=schedule_context,
                intent_data=intent_data,
                query_embedding=user_embedding  # Pass embedding for hybrid retrieval
            )
            
            # Add relevant history to context if not already included
            if relevant_history_context and "[RELEVANT PAST CONVERSATIONS]" not in context:
                context = f"[RELEVANT PAST CONVERSATIONS]\n{relevant_history_context}\n\n{context}"

            await NeuralEventBus.set_activity("prefrontal_cortex", "Generating Response")
            response = await self._generate_response_with_rotation(message, context, image_path)

            # Store both messages with embeddings
            response_embedding = await self._generate_embedding(response)
            await self._thalamus.update_session(
                message, response, image_path,
                user_embedding=user_embedding,
                ai_embedding=response_embedding
            )

            asyncio.create_task(self._post_process(message, response, intent_data))
            
            await NeuralEventBus.emit("prefrontal_cortex", "motor_cortex", "output_sent")
            await NeuralEventBus.clear_activity("prefrontal_cortex")

            return response

        except AllAPIExhaustedError:
            return "⚠️ Semua API dan model sedang kehabisan kuota. Coba lagi nanti."
        except Exception as e:
            return f"ERROR: {str(e)}"
        finally:
            self._processing = False

    def _format_relevant_history(self, messages: List) -> str:
        """Format relevant history messages for context."""
        if not messages:
            return ""
        lines = []
        for msg in messages[:5]:
            time_str = msg.timestamp.strftime("%d/%m %H:%M") if hasattr(msg, 'timestamp') else ""
            role = "User" if msg.role == "user" else "Vira"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"[{time_str}] {role}: {content}")
        return "\n".join(lines)

    async def _extract_intent(self, text: str) -> Dict:
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        if cache_key in self._intent_cache:
            cached, timestamp = self._intent_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.INTENT_CACHE_TTL:
                return cached

        if not self._client:
            return self._fallback_intent(text)

        try:
            prompt = f"{EXTRACTION_INSTRUCTION}\n\nInput: \"{text}\""
            response = self._client.models.generate_content(
                model=TIER_3_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256
                )
            )
            result = self._extract_json(response.text) or self._fallback_intent(text)
            self._intent_cache[cache_key] = (result, datetime.now())
            return result
        except Exception:
            return self._fallback_intent(text)

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
            if w[0].isupper() and len(w) > 2:
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
        memories = await self._hippocampus.recall(search_query, limit=3)

        entities = intent_data.get("entities", [])
        for entity in entities[:2]:
            entity_data = await self._hippocampus.query_entity(entity)
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
        """Generate embedding via Ollama (local) or return None."""
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
        """Public alias for _generate_embedding."""
        return await self._generate_embedding(text)

    async def _generate_response_with_rotation(
        self,
        user_text: str,
        context: str,
        image_path: Optional[str] = None
    ) -> str:
        """Generate response with automatic API/model rotation on failure."""
        API_ROTATOR.reset_if_stale(hours=1)

        history = self._thalamus.get_history_for_model()
        persona_modifier = self._amygdala.get_response_modifier()
        
        # Fetch dynamic persona
        active_persona = await self._hippocampus.get_active_persona()
        base_instruction = active_persona["instruction"] if active_persona else PERSONA_INSTRUCTION
        current_temperature = active_persona["temperature"] if active_persona else self.TEMPERATURE
        
        full_instruction = base_instruction + persona_modifier

        parts = []
        if context:
            parts.append(types.Part.from_text(text=f"[CONTEXT]\n{context}\n\n[USER INPUT]"))
        parts.append(types.Part.from_text(text=user_text))

        if image_path and os.path.exists(image_path):
            try:
                img = PIL.Image.open(image_path)
                parts.append(types.Part.from_image(image=img))
            except Exception:
                pass

        history.append(types.Content(role="user", parts=parts))

        max_retries = API_ROTATOR.total_combinations
        last_error = None

        for attempt in range(max_retries):
            try:
                api_key, model = API_ROTATOR.get_current()
                if not api_key:
                    raise AllAPIExhaustedError("No API key available")

                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=history,
                    config=types.GenerateContentConfig(
                        temperature=current_temperature,
                        top_p=self.TOP_P,
                        max_output_tokens=self.MAX_OUTPUT_TOKENS,
                        system_instruction=full_instruction
                    )
                )

                prefix = self._amygdala.get_response_prefix()
                text = response.text.strip()
                if prefix:
                    text = f"{prefix} {text}"
                return text

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                retry_triggers = ["rate", "quota", "429", "exhausted", "503", "unavailable", "overloaded"]
                if any(trigger in error_str for trigger in retry_triggers):
                    if not API_ROTATOR.mark_failed():
                        raise AllAPIExhaustedError(f"All API combinations exhausted. Last error: {e}")
                    print(f"⚠️ Rotating API/Model. Attempt {attempt+1}/{max_retries}. Error: {e}")
                else:
                    raise e

        raise AllAPIExhaustedError(f"Failed after {max_retries} attempts. Last error: {last_error}")

    async def _generate_response(
        self,
        user_text: str,
        context: str,
        image_path: Optional[str] = None
    ) -> str:
        """Legacy method - now calls rotation-aware version."""
        return await self._generate_response_with_rotation(user_text, context, image_path)

    async def _post_process(
        self,
        user_text: str,
        ai_response: str,
        intent_data: Dict
    ) -> None:
        try:
            analysis = await self._analyze_interaction(user_text, ai_response)
            if analysis:
                await self._process_analysis(analysis)
        except Exception:
            pass

    async def _analyze_interaction(
        self,
        user_text: str,
        ai_response: str
    ) -> Optional[Dict]:
        if not self._client:
            return None

        try:
            now = datetime.now().isoformat()
            prompt = f"""{CHAT_INTERACTION_ANALYSIS_INSTRUCTION}

[CURRENT SYSTEM TIME]: {now}
[USER INPUT]: {user_text}
[AI RESPONSE]: {ai_response}"""

            response = self._client.models.generate_content(
                model=TIER_2_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=512
                )
            )
            return self._extract_json(response.text)
        except Exception:
            return None

    async def _process_analysis(self, analysis: Dict) -> None:
        memory_data = analysis.get("memory", {})
        if memory_data.get("should_store"):
            summary = memory_data.get("summary", "")
            mem_type = memory_data.get("type", "general")
            priority = memory_data.get("priority", 0.5)

            if memory_data.get("action") == "add":
                await self._hippocampus.store(summary, mem_type, priority)
            elif memory_data.get("action") == "forget":
                pass

        schedules = analysis.get("schedules", [])
        for sched in schedules:
            if sched.get("should_schedule"):
                time_str = sched.get("time_str")
                context = sched.get("context", "")
                intent = sched.get("intent", "add")

                if intent == "add" and time_str:
                    try:
                        trigger_time = date_parser.parse(time_str)
                        await self._hippocampus.add_schedule(trigger_time, context)
                    except Exception:
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

    async def create_plan(self, goal: str) -> Optional[TaskPlan]:
        if not self._client:
            return None

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

            response = self._client.models.generate_content(
                model=TIER_2_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=512
                )
            )

            data = self._extract_json(response.text)
            if not data:
                return None

            plan = TaskPlan(
                id=1,
                goal=data.get("goal_summary", goal),
                original_request=goal,
                status=PlanStatus.IN_PROGRESS
            )

            for step_data in data.get("steps", []):
                step = TaskStep(
                    id=step_data.get("order"),
                    plan_id=1,
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
            self._amygdala.update_satisfaction(PlanProgressState.ABANDONED)
            old_plan = self._active_plan
            self._active_plan = None
            return f"Oke, rencana '{old_plan.goal}' dibatalkan."

        if any(w in text_lower for w in ["selesai", "done", "sudah", "completed"]):
            current_step = self._get_current_step()
            if current_step:
                current_step.status = StepStatus.COMPLETED
                current_step.completed_at = datetime.now()
                self._amygdala.update_satisfaction(PlanProgressState.ON_TRACK)

                next_step = self._get_current_step()
                if next_step:
                    return f"Mantap! Langkah selanjutnya: {next_step.description}"
                else:
                    self._active_plan.status = PlanStatus.COMPLETED
                    self._amygdala.update_satisfaction(PlanProgressState.COMPLETED)
                    goal = self._active_plan.goal
                    self._active_plan = None
                    return f"Selamat! Rencana '{goal}' selesai semua. Bangga sama lu!"

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

    async def check_pending_schedules(self, bot) -> None:
        pending = await self._hippocampus.get_pending_schedules(limit=5)

        for schedule in pending:
            schedule_id = schedule.get("id")
            context = schedule.get("context", "")

            from src.brainstem import ADMIN_ID
            if ADMIN_ID:
                try:
                    await bot.send_message(
                        chat_id=int(ADMIN_ID),
                        text=f"⏰ *Pengingat*: {context}"
                    )
                    await self._hippocampus.mark_schedule_executed(schedule_id, "delivered")
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
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
        text = re.sub(r'\s*```$', '', text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    def get_system_stats(self) -> Dict[str, Any]:
        return {
            "api_health": "OK" if self._client else "No API Key",
            "active_plan": self._active_plan.goal if self._active_plan else None,
            "sessions": 1,
            "active_processing": 1 if self._processing else 0,
            "total_users_tracked": 1
        }
