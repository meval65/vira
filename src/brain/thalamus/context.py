import datetime
from typing import List, Dict, Optional, Any
import tiktoken

from src.brain.brainstem import NeuralEventBus
from .types import ContextPriority


class ContextBuilderMixin:
    MAX_CONTEXT_TOKENS: int = 1500

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

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
        
        context_sections: List[Dict[str, Any]] = []
        
        global_ctx = await self._get_global_context()
        if global_ctx:
            context_sections.append({
                "priority": ContextPriority.GLOBAL_CONTEXT.value,
                "header": "[GLOBAL CONTEXT - USER PROFILE]",
                "content": global_ctx
            })
        
        persona = await self._get_active_persona()
        if persona and persona.get("instruction"):
            context_sections.append({
                "priority": ContextPriority.PERSONA_CONTEXT.value,
                "header": f"[ACTIVE PERSONA: {persona.get('name', 'Default')}]",
                "content": persona.get("instruction", "")
            })
        
        now = datetime.datetime.now()
        context_sections.append({
            "priority": ContextPriority.TIME_CONTEXT.value,
            "header": "[SYSTEM TIME]",
            "content": f"{now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A')})"
        })
        
        weather = await self._get_weather()
        if weather:
            context_sections.append({
                "priority": ContextPriority.WEATHER_CONTEXT.value,
                "header": "[WEATHER]",
                "content": weather
            })
        
        profile = self.hippocampus.admin_profile
        if profile.telegram_name or profile.additional_info:
            profile_parts = []
            if profile.telegram_name:
                profile_parts.append(f"Name: {profile.telegram_name}")
            if profile.additional_info:
                profile_parts.append(f"Info: {profile.additional_info}")
            context_sections.append({
                "priority": ContextPriority.PERSONA_CONTEXT.value - 5,
                "header": "[ADMIN PROFILE]",
                "content": "\n".join(profile_parts)
            })
        
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
        
        if schedule_context:
            context_sections.append({
                "priority": ContextPriority.SCHEDULE_CONTEXT.value,
                "header": "[UPCOMING SCHEDULES]",
                "content": schedule_context
            })
        
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
