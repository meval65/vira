import os
import json
import asyncio
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple

import httpx

from src.brain.brainstem import (
    AllAPIExhaustedError,
    OLLAMA_BASE_URL,
    EMBEDDING_MODEL,
    NeuralEventBus,
    OpenRouterClient
)
from src.brain.constants import (
    DEFAULT_PERSONA_INSTRUCTION,
    FILLER_WORDS
)
from src.brain.hippocampus import Memory
from src.brain.amygdala import Amygdala, PlanProgressState
from src.brain.self_correction import SelfCorrectionLoop

from .types import (
    TaskPlan, TaskStep, PlanStatus, StepStatus,
    IntentType, RequestType, ProcessingMetrics
)
from .utils import RateLimiter, async_retry
from .vision import VisualProcessor
from .intent import IntentAnalyzer
from .planning import PlanManager

class PrefrontalCortex:
    MAX_OUTPUT_TOKENS: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95
    RESPONSE_CACHE_TTL: int = 60
    MAX_CACHE_SIZE: int = 100
    RATE_LIMIT_REQUESTS: int = 30
    RATE_LIMIT_WINDOW: int = 60

    def __init__(self):
        self._brain = None
        self._openrouter: Optional[OpenRouterClient] = None
        self._response_cache: Dict[str, Tuple[str, datetime]] = {}
        self._processing = False
        self._current_persona: Optional[Dict] = None
        self._metrics = ProcessingMetrics()
        self._rate_limiter = RateLimiter(self.RATE_LIMIT_REQUESTS, self.RATE_LIMIT_WINDOW)
        self._self_correction: Optional[SelfCorrectionLoop] = None
        
        # Sub-modules
        self.vision = None
        self.intent = None
        self.planner = None
        
    def bind_brain(self, brain) -> None:
        self._brain = brain
        self.vision = VisualProcessor(self.hippocampus)
        # We will initialize intent and planner in initialize() as they might need openrouter etc
        
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
        
        self.intent = IntentAnalyzer(self._openrouter, self.parietal_lobe)
        self.planner = PlanManager(self._openrouter, self.amygdala)
        
        if self.parietal_lobe:
            from src.brain.db.mongo_client import get_mongo_client
            mongo = get_mongo_client()
            self._self_correction = SelfCorrectionLoop(
                openrouter_client=self._openrouter,
                parietal_lobe=self.parietal_lobe,
                mongo_client=mongo
            )
        
        await self._load_active_persona()
        await self._cleanup_old_cache()

    async def _load_active_persona(self) -> None:
        self._current_persona = await self.hippocampus.get_active_persona()
        if self._current_persona:
            print(f"  * Active Persona: {self._current_persona.get('name', 'Unknown')}")
            
            calibration = self._current_persona.get("calibration", {})
            if not calibration.get("calibration_status", False):
                print(f"  ! Persona '{self._current_persona.get('name')}' not calibrated, running calibration...")
                await self._run_persona_calibration(self._current_persona)
                self._current_persona = await self.hippocampus.get_active_persona()
            
            if self.amygdala:
                await self.amygdala.sync_with_persona(self._current_persona)
        else:
            print("  ! No active persona found, using default")

    async def _cleanup_old_cache(self) -> None:
        now = datetime.now()
        # Intent cache is managed by IntentAnalyzer, maybe we should expose cleanup there?
        if self.intent:
            self.intent.clear_cache()
            
        self._response_cache = {
            k: v for k, v in self._response_cache.items()
            if (now - v[1]).total_seconds() < self.RESPONSE_CACHE_TTL
        }
        
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
    
    async def describe_and_store_image(
        self,
        image_path: str,
        user_context: Optional[str] = None
    ) -> Optional[str]:
        if self.vision:
            return await self.vision.describe_and_store_image(
                image_path, user_context, self._generate_embedding
            )
        return None

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
            
            # 1. Vision Processing
            image_description = None
            if image_path:
                image_description = await self.describe_and_store_image(
                    image_path=image_path,
                    user_context=message if message.strip() else None
                )

            # 2. Embedding
            should_embed = self._should_embed_text(message)
            user_embedding = await self._generate_embedding(message) if should_embed else None

            # 3. Intent Analysis
            await NeuralEventBus.set_activity("prefrontal_cortex", "Unified Analysis")
            analysis_result = await self.intent.analyze(message)
            
            intent_data = {
                "intent_type": analysis_result.get("intent_type", "statement"),
                "request_type": analysis_result.get("request_type", "general_chat"),
                "entities": analysis_result.get("entities", []),
                "search_query": analysis_result.get("search_query", message[:100]),
                "sentiment": analysis_result.get("sentiment", "neutral"),
                "needs_memory": analysis_result.get("needs_memory", True),
                "confidence": analysis_result.get("confidence", 0.6)
            }
            
            detected_emotion = analysis_result.get("emotion", "neutral")
            emotion_intensity = analysis_result.get("emotion_intensity", 0.5)
            
            if self.amygdala:
                self.amygdala.adjust_for_emotion(detected_emotion, emotion_intensity)
            
            asyncio.create_task(NeuralEventBus.emit("prefrontal_cortex", "prefrontal_cortex", "analysis_complete", payload={
                "intent": intent_data,
                "emotion": detected_emotion,
                "intensity": emotion_intensity
            }))

            # 4. Tool Execution / Reflex
            reflex_context = ""
            tool_needed = analysis_result.get("tool_needed")
            if tool_needed and self.parietal_lobe:
                tool_name = tool_needed.get("tool")
                tool_args = tool_needed.get("args", {})
                if tool_name:
                    await NeuralEventBus.set_activity("prefrontal_cortex", f"Using Tool: {tool_name}")
                    tool_result = await self._execute_tool_with_retry(tool_name, tool_args)
                    if tool_result:
                        reflex_context = f"[TOOL RESULT ({tool_name})]\n{tool_result}\n\n"
                        self._metrics.tool_executions += 1

            # 5. Retrieval
            await NeuralEventBus.set_activity("prefrontal_cortex", "Retrieving Memories")
            if intent_data.get("request_type") == RequestType.MEMORY_RECALL.value:
                # Specific event for memory recall
                pass
            
            memories = await self._retrieve_memories(message, intent_data)
            self._metrics.memory_retrievals += 1
            await NeuralEventBus.emit("hippocampus", "prefrontal_cortex", "memories_found", payload={
                "count": len(memories),
                "types": [m.memory_type for m in memories]
            })

            # 6. Planning
            if self.planner.get_active_plan():
                await NeuralEventBus.set_activity("prefrontal_cortex", "Executing Plan")
                plan_response = await self.planner.handle_active_plan(message)
                if plan_response:
                    await NeuralEventBus.clear_activity("prefrontal_cortex")
                    success = True
                    return plan_response

            if self.planner.should_create_plan(message):
                await NeuralEventBus.set_activity("prefrontal_cortex", "Creating New Plan")
                plan = await self.planner.create_plan(message)
                if plan and self.amygdala:
                    self.amygdala.update_satisfaction(PlanProgressState.ON_TRACK)

            # 7. Context Building
            schedules = await self.hippocampus.get_upcoming_schedules(hours_ahead=24)
            schedule_context = self._format_schedules(schedules) if schedules else None

            await NeuralEventBus.set_activity("prefrontal_cortex", "Building Context")
            await NeuralEventBus.emit("prefrontal_cortex", "thalamus", "context_building")
            
            relevant_history = await self.thalamus.get_relevant_history_async(user_embedding, top_k=5)
            relevant_history_context = self._format_relevant_history(relevant_history)

            from dataclasses import asdict
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

            insight_context = await self._get_relevant_insights(message)
            if insight_context:
                context = f"{context}\n\n{insight_context}"

            # 8. Generation
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
        if self._self_correction:
            result, success, attempts = await self._self_correction.execute_with_correction(
                tool_name, tool_args
            )
            if success:
                return result
            # Log warning if needed
            return None
        
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

    def _should_embed_text(self, text: str) -> bool:
        if not text:
            return False
        
        words = text.strip().split()
        if len(words) < 3:
            return False
        
        meaningful_words = [w for w in words if w.lower().strip(".,!?;:") not in FILLER_WORDS]
        return len(meaningful_words) >= 2

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

    async def _get_relevant_insights(self, user_message: str) -> Optional[str]:
        if not self.hippocampus:
            return None
        
        try:
            insights = await self.hippocampus.get_relevant_insights(user_message, limit=2)
            if not insights:
                return None
            
            insight_texts = []
            for insight in insights:
                if insight.get("insight_text"):
                    insight_texts.append(insight["insight_text"])
                    insight_id = insight.get("id")
                    if insight_id:
                        asyncio.create_task(self.hippocampus.mark_insight_used(insight_id))
            
            if not insight_texts:
                return None
            
            formatted = "[INTERNAL INSIGHT - Gunakan secara natural jika relevan, jangan sebutkan bahwa ini dari 'insight' atau 'lamunan']\n"
            formatted += "\n".join(f"- {text}" for text in insight_texts)
            
            return formatted
        except Exception:
            return None

    def get_system_stats(self) -> Dict[str, Any]:
        api_status = self._openrouter.get_status()
        active_plan = self.planner.get_active_plan()
        
        return {
            "api_health": "OK" if api_status.get("api_configured") else "No API Key",
            "active_plan": active_plan.goal if active_plan else None,
            "sessions": 1,
            "active_processing": 1 if self._processing else 0,
            "total_users_tracked": 1,
            "current_persona": self._current_persona.get("name") if self._current_persona else "Default",
            "model_health": api_status.get("model_health", {}),
            "metrics": {
                "total_requests": self._metrics.total_requests,
                "success_rate": f"{self._metrics.get_success_rate():.1f}%",
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
        intent_cache_size = 0
        if self.intent:
            intent_cache_size = self.intent.clear_cache()
            
        response_cache_size = len(self._response_cache)
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
    
    async def switch_persona(self, persona_id: str) -> bool:
        success = await self.hippocampus.set_active_persona(persona_id)
        if success:
            await self._load_active_persona()
            
            if self.amygdala and self._current_persona:
                await self.amygdala.sync_with_persona(self._current_persona)
            
            await NeuralEventBus.emit(
                "prefrontal_cortex", "dashboard", "persona_changed",
                payload={
                    "persona_id": persona_id,
                    "persona_name": self._current_persona.get("name") if self._current_persona else "Unknown",
                    "calibration_status": self._current_persona.get("calibration", {}).get("calibration_status", False) if self._current_persona else False
                }
            )
        return success

    async def _run_persona_calibration(self, persona: Dict) -> None:
        if not persona or not self._openrouter:
            return
        
        persona_id = persona.get("id")
        description = persona.get("description", "")
        traits = persona.get("traits", {})
        voice_tone = persona.get("voice_tone", "friendly")
        
        calibration_data = await self._generate_calibration_from_instruction(
            description, traits, voice_tone
        )
        
        if calibration_data and persona_id:
            await self.hippocampus.update_persona_calibration(
                persona_id=persona_id,
                emotional_inertia=calibration_data.get("emotional_inertia", 0.7),
                base_arousal=calibration_data.get("base_arousal", 0.0),
                base_valence=calibration_data.get("base_valence", 0.0),
                calibration_status=True,
                identity_anchor=calibration_data.get("identity_anchor", description)
            )
            print(f"  * Calibration complete for '{persona.get('name')}'")

    async def _generate_calibration_from_instruction(
        self,
        description: str,
        traits: Dict,
        voice_tone: str
    ) -> Dict:
        try:
            prompt = f"""Analyze this persona and generate emotional calibration parameters.

Persona Description: {description}
Traits: {traits}
Voice Tone: {voice_tone}

Generate calibration values in JSON format:
{{
  "emotional_inertia": 0.0-1.0 (how stable/slow emotions change, higher = more stable),
  "base_arousal": -1.0 to 1.0 (baseline energy level, negative = calm, positive = energetic),
  "base_valence": -1.0 to 1.0 (baseline mood, negative = serious, positive = cheerful),
  "identity_anchor": "A concise identity statement for this persona"
}}

Consider the traits and tone to determine appropriate emotional baseline."""

            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.2,
                tier="analysis_model",
                json_mode=True
            )
            
            from .utils import extract_json
            result = extract_json(response)
            if result:
                return {
                    "emotional_inertia": max(0.0, min(1.0, result.get("emotional_inertia", 0.7))),
                    "base_arousal": max(-1.0, min(1.0, result.get("base_arousal", 0.0))),
                    "base_valence": max(-1.0, min(1.0, result.get("base_valence", 0.0))),
                    "identity_anchor": result.get("identity_anchor", description)
                }
        except Exception:
            pass
        
        return self._fallback_calibration_from_traits(traits, voice_tone, description)

    def _fallback_calibration_from_traits(self, traits: Dict, voice_tone: str, description: str) -> Dict:
        emotional_inertia = 0.7
        base_arousal = 0.0
        base_valence = 0.0
        
        if "enthusiasm" in traits:
            enthusiasm = traits["enthusiasm"]
            base_arousal = (enthusiasm - 0.5) * 0.6
            base_valence = (enthusiasm - 0.5) * 0.4
        
        if "formality" in traits:
            formality = traits["formality"]
            emotional_inertia = 0.5 + (formality * 0.4)
        
        tone_modifiers = {
            "friendly": (0.0, 0.2),
            "professional": (0.0, -0.1),
            "playful": (0.3, 0.3),
            "calm": (-0.2, 0.0),
            "energetic": (0.4, 0.2),
            "empathetic": (0.0, 0.1)
        }
        
        if voice_tone in tone_modifiers:
            arousal_mod, valence_mod = tone_modifiers[voice_tone]
            base_arousal = max(-1.0, min(1.0, base_arousal + arousal_mod))
            base_valence = max(-1.0, min(1.0, base_valence + valence_mod))
        
        return {
            "emotional_inertia": emotional_inertia,
            "base_arousal": base_arousal,
            "base_valence": base_valence,
            "identity_anchor": description
        }

    # Creating proxy methods for PlanManager to maintain backward compatibility if needed,
    # or just updating usage to access self.planner
    # The original file had these methods on self.
    # To keep simple API, we can expose them.
    
    async def create_plan(self, goal: str, priority: int = 5, deadline: Optional[datetime] = None) -> Optional[TaskPlan]:
        return await self.planner.create_plan(goal, priority, deadline)
        
    def get_active_plan(self) -> Optional[TaskPlan]:
        return self.planner.get_active_plan()

    def get_plan_history(self) -> List[TaskPlan]:
        return self.planner.get_plan_history()

    def get_plan_progress(self) -> Optional[Dict[str, Any]]:
        return self.planner.get_plan_progress()
