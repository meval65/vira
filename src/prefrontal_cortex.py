import os
import json
import asyncio
import re
import hashlib
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict

import httpx
import PIL.Image

from dateutil import parser as date_parser

from src.brainstem import (
    DEFAULT_PERSONA_INSTRUCTION,
    CHAT_INTERACTION_ANALYSIS_INSTRUCTION, EXTRACTION_INSTRUCTION,
    MemoryType, AllAPIExhaustedError, OLLAMA_BASE_URL, EMBEDDING_MODEL,
    NeuralEventBus, get_openrouter_client, OpenRouterClient
)
from src.hippocampus import Hippocampus, Memory
from src.amygdala import Amygdala, PlanProgressState
from src.thalamus import Thalamus
from src.parietal_lobe import ParietalLobe

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
    """
    Main processing module for Vira.
    
    Handles message processing, response generation, and coordination
    between memory, emotion, and context modules.
    
    Now uses OpenRouter API for all LLM operations.
    """
    
    MAX_OUTPUT_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    INTENT_CACHE_TTL: int = 300

    def __init__(
        self,
        hippocampus: Hippocampus,
        amygdala: Amygdala,
        thalamus: Thalamus,
        parietal_lobe: Optional[ParietalLobe] = None
    ):
        self._hippocampus = hippocampus
        self._amygdala = amygdala
        self._thalamus = thalamus
        self._parietal_lobe = parietal_lobe
        self._openrouter: OpenRouterClient = get_openrouter_client()
        self._active_plan: Optional[TaskPlan] = None
        self._intent_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._processing = False
        self._current_persona: Optional[Dict] = None

        amygdala.set_hippocampus(hippocampus)

    async def initialize(self) -> None:
        """Initialize the prefrontal cortex."""
        await self._load_active_persona()
        await self._load_active_plan()

    async def _load_active_persona(self) -> None:
        """Load the currently active persona from MongoDB."""
        self._current_persona = await self._hippocampus.get_active_persona()
        if self._current_persona:
            print(f"  ✓ Active Persona: {self._current_persona.get('name', 'Unknown')}")
        else:
            print("  ⚠ No active persona found, using default")

    async def _load_active_plan(self) -> None:
        """Load any active plan from storage."""
        pass

    def _calculate_dynamic_params(
        self,
        intent_type: str,
        request_type: str,
        emotion: str,
        base_temp: float
    ) -> Dict[str, Any]:
        """
        Dynamically adjust LLM parameters based on context (Neuro-Modulation).
        Simulates changing cognitive states (Focus vs Creativity).
        """
        params = {
            "temperature": base_temp,
            "max_tokens": self.MAX_OUTPUT_TOKENS,
            "top_p": self.TOP_P
        }

        # 1. Modulation based on INTENT (The Task)
        # Low temp for precision/logic
        if request_type in [RequestType.SCHEDULE.value, RequestType.ACTION.value, RequestType.MEMORY_RECALL.value]:
            params["temperature"] = max(0.1, base_temp - 0.4)
            params["top_p"] = 0.85
        # High temp for chat/ideas
        elif request_type in [RequestType.GENERAL_CHAT.value, RequestType.RECOMMENDATION.value]:
            params["temperature"] = min(1.0, base_temp + 0.1)
            params["top_p"] = 0.95
        
        # 2. Modulation based on EMOTION (The Mood)
        # Excited/Happy = Higher entropy (creative/energetic)
        if emotion in ["happy", "excited"]:
            params["temperature"] = min(1.0, params["temperature"] + 0.1)
            params["max_tokens"] = int(self.MAX_OUTPUT_TOKENS * 1.2) # Talk more
        # Sad/Concerned = Lower entropy (reserved/careful)
        elif emotion in ["sad", "concerned", "anxious"]:
            params["temperature"] = max(0.3, params["temperature"] - 0.2)
            params["max_tokens"] = int(self.MAX_OUTPUT_TOKENS * 0.6) # Talk less
        # Angry/Serious = Strict focus
        elif emotion in ["angry"]:
            params["temperature"] = 0.2
            params["max_tokens"] = 512 # Concise

        # 3. Modulation based on INPUT TYPE
        if intent_type == IntentType.QUESTION.value:
            # Questions need slightly more focus than statements
            params["temperature"] = max(0.3, params["temperature"] - 0.1)

        # Safety clamps
        params["temperature"] = round(max(0.0, min(1.0, params["temperature"])), 2)
        
        return params

    def _get_persona_instruction(self) -> str:
        """Get the current persona instruction."""
        if self._current_persona and self._current_persona.get("instruction"):
            return self._current_persona["instruction"]
        return DEFAULT_PERSONA_INSTRUCTION
    
    def _get_persona_temperature(self) -> float:
        """Get the current persona temperature setting."""
        if self._current_persona and self._current_persona.get("temperature"):
            return self._current_persona["temperature"]
        return self.TEMPERATURE

    async def process(
        self,
        message: str,
        image_path: Optional[str] = None,
        user_name: Optional[str] = None
    ) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            message: User's text message
            image_path: Optional path to an image file
            user_name: Optional user's display name
        
        Returns:
            AI response string
        """
        if self._processing:
            return "⏳ Sedang memproses pesan sebelumnya..."

        self._processing = True
        try:
            if user_name:
                await self._hippocampus.update_admin_profile(telegram_name=user_name)

            user_embedding = await self._generate_embedding(message)

            await NeuralEventBus.set_activity("prefrontal_cortex", "Analyzing Intent")
            intent_data = await self._extract_intent(message)
            await NeuralEventBus.emit("prefrontal_cortex", "prefrontal_cortex", "intent_extracted", payload=intent_data)
            
            await NeuralEventBus.emit("prefrontal_cortex", "amygdala", "emotion_check")
            detected_emotion = self._amygdala.detect_emotion_from_text(message)
            await NeuralEventBus.emit("amygdala", "prefrontal_cortex", "emotion_detected", payload={
                "emotion": detected_emotion
            })
            self._amygdala.adjust_for_emotion(detected_emotion)

            # --- REFLEX LAYER ---
            reflex_context = ""
            if self._parietal_lobe:
                reflex_check = await self._detect_reflex_need(message)
                if reflex_check:
                    tool_name, tool_args = reflex_check
                    await NeuralEventBus.set_activity("prefrontal_cortex", f"Using Tool: {tool_name}")
                    tool_result = await self._parietal_lobe.execute(tool_name, tool_args)
                    reflex_context = f"[TOOL RESULT ({tool_name})]\n{tool_result}\n\n"
                    await NeuralEventBus.emit("prefrontal_cortex", "parietal_lobe", "tool_executed", payload={
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result[:100]
                    })
            # --------------------

            await NeuralEventBus.set_activity("prefrontal_cortex", "Retrieving Memories")
            await NeuralEventBus.emit("prefrontal_cortex", "hippocampus", "memory_retrieval", payload={
                "query": intent_data.get("search_query", message)[:50]
            })
            memories = await self._retrieve_memories(message, intent_data)
            await NeuralEventBus.emit("hippocampus", "prefrontal_cortex", "memories_found", payload={
                "count": len(memories),
                "types": [m.memory_type for m in memories]
            })

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

            await NeuralEventBus.set_activity("prefrontal_cortex", "Building Context")
            await NeuralEventBus.emit("prefrontal_cortex", "thalamus", "context_building")
            
            relevant_history = await self._thalamus.get_relevant_history_async(user_embedding, top_k=5)
            relevant_history_context = self._format_relevant_history(relevant_history)

            context = await self._thalamus.build_context(
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

            response_embedding = await self._generate_embedding(response)
            await self._thalamus.update_session(
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

            return response

        except AllAPIExhaustedError:
            return "⚠️ Semua API dan model sedang kehabisan kuota. Coba lagi nanti."
        except Exception as e:
            import traceback
            traceback.print_exc()
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
            role = "User" if msg.role == "user" else "AI"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"[{time_str}] {role}: {content}")
        return "\n".join(lines)

    async def _extract_intent(self, text: str) -> Dict:
        """Extract intent and metadata from user text."""
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
                temperature=0.1
            )
            result = self._extract_json(response) or self._fallback_intent(text)
            self._intent_cache[cache_key] = (result, datetime.now())
            return result
        except Exception:
            return self._fallback_intent(text)

    async def _detect_reflex_need(self, text: str) -> Optional[Tuple[str, Dict]]:
        """
        Check if the user input requires a reflex tool execution.
        Returns: (tool_name, tool_args) or None
        """
        if not self._parietal_lobe:
            return None
            
        # Fast heuristic checks to avoid LLM overhead for obvious non-tools
        text_lower = text.lower()
        triggers = ["jam", "waktu", "pukul", "time", "date", "tanggal", "hitung", "calc", "cuaca", "weather"]
        if not any(t in text_lower for t in triggers):
            return None

        tools_desc = self._parietal_lobe.get_tool_descriptions()
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
                temperature=0.0
            )
            data = self._extract_json(response)
            if data and data.get("tool"):
                return (data["tool"], data.get("args", {}))
        except Exception:
            pass
            
        return None

    def _fallback_intent(self, text: str) -> Dict:
        """Fallback intent extraction without LLM."""
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
        """Retrieve relevant memories based on query and intent."""
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

    async def _generate_response(
        self,
        user_text: str,
        context: str,
        image_path: Optional[str] = None,
        intent_info: Optional[Dict] = None
    ) -> str:
        """
        Generate response using OpenRouter API.
        
        Args:
            user_text: User's message
            context: Built context string
            image_path: Optional image path (vision support)
            intent_info: Information about user intent for parameter shaping
        
        Returns:
            Generated response string
        """
        persona_instruction = self._get_persona_instruction()
        persona_modifier = self._amygdala.get_response_modifier()
        full_instruction = persona_instruction + persona_modifier
        
        # --- DYNAMIC COGNITIVE PARAMETERS ---
        base_temp = self._get_persona_temperature()
        
        # Defaults if intent info is missing
        i_type = intent_info.get("intent_type", "statement") if intent_info else "statement"
        r_type = intent_info.get("request_type", "general_chat") if intent_info else "general_chat"
        current_mood = self._amygdala.mood.value
        
        cognitive_params = self._calculate_dynamic_params(i_type, r_type, current_mood, base_temp)
        
        # Log parameter change for debugging (or "thought" awareness)
        if cognitive_params["temperature"] != base_temp:
            print(f"  🧠 Neuro-Modulation: Temp {base_temp} -> {cognitive_params['temperature']} | Mood: {current_mood}")
        # ------------------------------------
        
        user_content = user_text
        if context:
            user_content = f"[CONTEXT]\n{context}\n\n[USER INPUT]\n{user_text}"
        
        if image_path and os.path.exists(image_path):
            user_content += "\n\n[Note: User sent an image]"
        
        history = await self._thalamus.get_history_for_model_async()
        messages = []
        
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
            system=full_instruction,
            temperature=cognitive_params["temperature"],
            max_tokens=cognitive_params["max_tokens"],
            top_p=cognitive_params["top_p"],
            preferred_tier="tier_1"
        )
        
        prefix = self._amygdala.get_response_prefix()
        text = response.content.strip()
        if prefix:
            text = f"{prefix} {text}"
        
        return text

    async def _post_process(
        self,
        user_text: str,
        ai_response: str,
        intent_data: Dict
    ) -> None:
        """Post-process interaction for memory and schedule extraction."""
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
        """Analyze interaction for memory/schedule extraction."""
        try:
            now = datetime.now().isoformat()
            prompt = f"""{CHAT_INTERACTION_ANALYSIS_INSTRUCTION}

[CURRENT SYSTEM TIME]: {now}
[USER INPUT]: {user_text}
[AI RESPONSE]: {ai_response}"""

            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.2
            )
            return self._extract_json(response)
        except Exception:
            return None

    async def _process_analysis(self, analysis: Dict) -> None:
        """Process analysis results for memory and schedule storage."""
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
        """Check if a task plan should be created."""
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
        """Create a task plan for a complex goal."""
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
        """Handle user input in context of active plan."""
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
        """Get the current active step in the plan."""
        if not self._active_plan:
            return None

        for step in self._active_plan.steps:
            if step.status in [StepStatus.PENDING, StepStatus.IN_PROGRESS]:
                return step
        return None

    def get_active_plan(self) -> Optional[TaskPlan]:
        """Get the currently active plan."""
        return self._active_plan

    async def check_pending_schedules(self, bot) -> None:
        """Check and send pending schedule reminders."""
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
        """Format schedules for context display."""
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
        """Switch to a different persona."""
        success = await self._hippocampus.set_active_persona(persona_id)
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
        """Get system statistics."""
        api_status = self._openrouter.get_status()
        return {
            "api_health": "OK" if api_status.get("api_configured") else "No API Key",
            "active_plan": self._active_plan.goal if self._active_plan else None,
            "sessions": 1,
            "active_processing": 1 if self._processing else 0,
            "total_users_tracked": 1,
            "current_persona": self._current_persona.get("name") if self._current_persona else "Default",
            "model_health": api_status.get("model_health", {})
        }
