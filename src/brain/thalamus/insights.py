import datetime
from typing import List, Optional, Dict, Any

from .types import InsightType, InsightPriority, ProactiveInsight


class InsightsManagerMixin:
    INACTIVITY_THRESHOLD_HOURS: int = 72
    COOLDOWN_MINUTES: int = 120

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
        insights = []
        memory_events = await self._check_memory_events()
        insights.extend(memory_events)
        reminders = await self._check_scheduled_reminders()
        insights.extend(reminders)
        inactivity = await self._check_inactivity()
        if inactivity:
            insights.append(inactivity)
        knowledge = await self._check_knowledge_gaps()
        if knowledge:
            insights.append(knowledge)
        return insights

    async def _check_memory_events(self) -> List[ProactiveInsight]:
        insights = []
        try:
            today_start = datetime.datetime.now().replace(hour=0, minute=0, second=0)
            today_end = today_start + datetime.timedelta(days=1)
            
            recent_schedules = await self._mongo.schedules.find({
                "scheduled_at": {"$gte": today_start, "$lt": today_end},
                "status": "executed"
            }).to_list(10)
            
            for schedule in recent_schedules:
                context = schedule.get("context", "")
                if await self._is_insight_sent(f"followup_{schedule['_id']}"):
                    continue
                    
                if any(w in context.lower() for w in ["ujian", "tes", "meeting", "rapat", "dokter", "janji"]):
                    msg = f"Gimana {context}-nya tadi? Lancar kan?"
                    insights.append(ProactiveInsight(
                        insight_type=InsightType.FOLLOW_UP,
                        priority=InsightPriority.HIGH,
                        message=msg,
                        context={"schedule_id": str(schedule["_id"]), "original_context": context}
                    ))
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
                context_info = f"\n\nInformasi tentang user:\n{global_ctx}"
            
            persona_context = ""
            if persona_instruction:
                persona_context = f"\n\nKarakter persona:\n{persona_instruction}"
            
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
        self._insight_cache[insight_type.value] = datetime.datetime.now()

    def _mark_insight_sent_with_id(self, unique_id: str) -> None:
        self._insight_cache[unique_id] = datetime.datetime.now()

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

        pending = await self.hippocampus.get_pending_schedules(limit=1)
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


