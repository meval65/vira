"""
Proactive Insight Engine for Vira AI

Analyzes user context, memories, and patterns to generate proactive
insights and conversation starters. Enables Vira to reach out to users
with relevant, timely messages.

Features:
- Pattern detection in user behavior
- Insight generation from knowledge graph
- Proactive reminder triggering
- Inactivity detection and re-engagement
- Context-aware conversation starters
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

from src.database import DBConnection
from src.services.knowledge_graph import KnowledgeGraph
from src.config import GLOBAL_GEN_CONFIG

logger = logging.getLogger(__name__)


class InsightType(Enum):
    REMINDER = "reminder"           # Scheduled reminder is due
    FOLLOW_UP = "follow_up"         # Follow up on previous conversation
    PATTERN = "pattern"             # Detected behavioral pattern
    ANNIVERSARY = "anniversary"     # Date-based milestone
    INACTIVITY = "inactivity"       # User hasn't interacted recently
    KNOWLEDGE = "knowledge"         # Insight from knowledge graph
    WEATHER = "weather"             # Weather-based suggestion
    WELLNESS = "wellness"           # Wellness check-in


class InsightPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class ProactiveInsight:
    """Represents a proactive insight/message to send to user."""
    user_id: str
    insight_type: InsightType
    priority: InsightPriority
    message: str
    context: Dict = field(default_factory=dict)
    trigger_time: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    delivered: bool = False
    delivery_callback: Optional[str] = None  # Function name to call on delivery


@dataclass
class UserActivityProfile:
    """Tracks user activity patterns."""
    user_id: str
    last_interaction: Optional[datetime] = None
    total_interactions: int = 0
    avg_response_time: float = 0.0
    active_hours: List[int] = field(default_factory=list)
    preferred_topics: Dict[str, int] = field(default_factory=dict)
    sentiment_trend: float = 0.0
    engagement_score: float = 0.5



class ProactiveInsightEngine:
    """
    Engine for generating and managing proactive insights.
    Analyzes user behavior, memories, and context to provide
    timely, relevant outreach.
    """
    
    # Configuration
    INACTIVITY_THRESHOLD_HOURS = 72  # 3 days
    RE_ENGAGEMENT_COOLDOWN_HOURS = 24
    MAX_INSIGHTS_PER_DAY = 5
    INSIGHT_EXPIRY_HOURS = 24
    
    def __init__(self, db: DBConnection, 
                 knowledge_graph: Optional[KnowledgeGraph] = None,
                 session_manager = None,
                 context_builder = None):
        self.db = db
        self.knowledge_graph = knowledge_graph
        self.session_manager = session_manager
        self.context_builder = context_builder
        self._lock = asyncio.Lock()
        
        # In-memory caches
        self._user_profiles: Dict[str, UserActivityProfile] = {}
        self._pending_insights: Dict[str, List[ProactiveInsight]] = defaultdict(list)
        self._delivery_history: Dict[str, List[datetime]] = defaultdict(list)
        self._pattern_cache: Dict[str, Dict] = {}

    # ... (skipping methods until generate_proactive_message) ...
        
    async def generate_proactive_message(
        self, 
        user_id: str, 
        insight: ProactiveInsight,
        genai_client=None,
        model_name: str = None
    ) -> str:
        """
        Generate a natural, contextual proactive message.
        Uses LLM if available, otherwise returns template.
        """
        if not genai_client or not model_name:
            return insight.message
        
        try:
            # Build Rich Context
            history_context = ""
            if self.session_manager:
                recent_history = self.session_manager.get_session(user_id)
                # Take last 5 interactions to inform tone/context
                # Assuming get_session returns full history, take tail
                history_subset = list(recent_history)[-10:] 
                # Convert to string summary
                history_context = "\n".join([f"{role}: {text}" for role, text, _ in history_subset])
            
            # Use ContextBuilder if available
            system_context = ""
            if self.context_builder:
                # We can reuse cached context or build partial
                # For efficiency, we might just grab memory summary or manually build
                # But let's check if we can build full context?
                # It might be expensive. Let's build a lightweight prompt.
                pass

            context_prompt = f"""
Generate a friendly, natural Indonesian message for Vira AI to proactively reach out to the user.

[USER CONTEXT]
Last Interactions:
{history_context}

[INSIGHT TRIGGER]
Type: {insight.insight_type.value}
Core Message: {insight.message}
Details: {insight.context}

[INSTRUCTION]
- Use the User Context to avoid awkwardness (e.g. don't be cheerful if user was sad).
- Keep it under 2 sentences.
- Be warm, personal, and use 'Lu/Gw' (casual slang).
- Emulate Vira's persona (Big Sister).

Response (message only):
"""
            
            response = genai_client.models.generate_content(
                model=model_name,
                contents=context_prompt,
                config=GLOBAL_GEN_CONFIG # Use unified config
            )
            
            if response.text:
                return response.text.strip()
        
        except Exception as e:
            logger.error(f"[PROACTIVE] Message generation failed: {e}")
        
        return insight.message
        
    async def initialize(self):
        """Create tables for insight tracking."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS proactive_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                insight_type TEXT NOT NULL,
                priority INTEGER DEFAULT 2,
                message TEXT NOT NULL,
                context TEXT,
                trigger_time TIMESTAMP,
                expires_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                delivered BOOLEAN DEFAULT 0,
                delivered_at TIMESTAMP,
                user_response TEXT
            )
        """, ())
        
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS user_activity_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                activity_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        """, ())
        
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_insights_user_pending ON proactive_insights(user_id, delivered)",
            ()
        )
        
        logger.info("[PROACTIVE] Insight engine initialized")
    
    async def record_activity(self, user_id: str, activity_type: str, metadata: Dict = None):
        """Record user activity for pattern analysis."""
        import json
        
        await self.db.execute("""
            INSERT INTO user_activity_logs (user_id, activity_type, timestamp, metadata)
            VALUES (?, ?, ?, ?)
        """, (user_id, activity_type, datetime.now(), json.dumps(metadata) if metadata else None))
        
        # Update profile cache
        await self._update_user_profile(user_id)
    
    async def _update_user_profile(self, user_id: str):
        """Update cached user activity profile."""
        profile = self._user_profiles.get(user_id, UserActivityProfile(user_id=user_id))
        
        # Get recent activity
        rows = await self.db.fetchall("""
            SELECT activity_type, timestamp FROM user_activity_logs
            WHERE user_id=? ORDER BY timestamp DESC LIMIT 100
        """, (user_id,))
        
        if rows:
            profile.last_interaction = rows[0][1]
            profile.total_interactions = len(rows)
            
            # Extract active hours
            hours = [row[1].hour if isinstance(row[1], datetime) else 12 for row in rows]
            profile.active_hours = list(set(hours))
        
        self._user_profiles[user_id] = profile
    
    async def check_for_insights(self, user_id: str) -> List[ProactiveInsight]:
        """
        Check all insight sources and return pending insights for user.
        Called periodically or on specific triggers.
        """
        insights = []
        
        async with self._lock:
            # Check rate limits
            if not self._can_send_insight(user_id):
                return []
            
            # 1. Check scheduled reminders
            reminder_insights = await self._check_scheduled_reminders(user_id)
            insights.extend(reminder_insights)
            
            # 2. Check for inactivity
            inactivity_insight = await self._check_inactivity(user_id)
            if inactivity_insight:
                insights.append(inactivity_insight)
            
            # 3. Check knowledge graph for interesting connections
            if self.knowledge_graph:
                kg_insights = await self._check_knowledge_insights(user_id)
                insights.extend(kg_insights)
            
            # 4. Check for date-based insights (anniversaries, etc)
            date_insights = await self._check_date_insights(user_id)
            insights.extend(date_insights)
            
            # 5. Check for follow-up opportunities
            followup_insights = await self._check_follow_ups(user_id)
            insights.extend(followup_insights)
        
        # Sort by priority and filter expired
        now = datetime.now()
        valid_insights = [
            i for i in insights 
            if i.expires_at is None or i.expires_at > now
        ]
        valid_insights.sort(key=lambda x: x.priority.value, reverse=True)
        
        return valid_insights[:3]  # Return top 3 insights
    
    async def _check_scheduled_reminders(self, user_id: str) -> List[ProactiveInsight]:
        """Check for scheduled reminders that are due."""
        insights = []
        now = datetime.now()
        
        rows = await self.db.fetchall("""
            SELECT id, context, scheduled_at, priority
            FROM schedules
            WHERE user_id=? AND status='pending' AND scheduled_at <= ?
            ORDER BY scheduled_at ASC LIMIT 5
        """, (user_id, now))
        
        for row in rows:
            insight = ProactiveInsight(
                user_id=user_id,
                insight_type=InsightType.REMINDER,
                priority=InsightPriority.HIGH if row[3] >= 2 else InsightPriority.MEDIUM,
                message=f"🔔 Reminder: {row[1]}",
                context={"schedule_id": row[0], "trigger_time": str(row[2])},
                trigger_time=row[2],
                expires_at=now + timedelta(hours=2)
            )
            insights.append(insight)
        
        return insights
    
    async def _check_inactivity(self, user_id: str) -> Optional[ProactiveInsight]:
        """Check if user has been inactive and needs re-engagement."""
        profile = self._user_profiles.get(user_id)
        
        if not profile or not profile.last_interaction:
            return None
        
        hours_inactive = (datetime.now() - profile.last_interaction).total_seconds() / 3600
        
        if hours_inactive < self.INACTIVITY_THRESHOLD_HOURS:
            return None
        
        # Check cooldown
        last_reengagement = self._get_last_delivery(user_id, InsightType.INACTIVITY)
        if last_reengagement:
            cooldown_hours = (datetime.now() - last_reengagement).total_seconds() / 3600
            if cooldown_hours < self.RE_ENGAGEMENT_COOLDOWN_HOURS:
                return None
        
        # Generate re-engagement message
        messages = [
            "👋 Hey! Sudah beberapa hari kita tidak ngobrol. Ada yang bisa Vira bantu?",
            "🌟 Apa kabar? Vira rindu mengobrol denganmu!",
            "💭 Vira ada di sini kalau kamu butuh teman bicara atau bantuan.",
            "✨ Lama tidak jumpa! Ada rencana menarik yang mau diceritakan?"
        ]
        
        import random
        message = random.choice(messages)
        
        return ProactiveInsight(
            user_id=user_id,
            insight_type=InsightType.INACTIVITY,
            priority=InsightPriority.LOW,
            message=message,
            context={"hours_inactive": hours_inactive},
            expires_at=datetime.now() + timedelta(hours=12)
        )
    
    async def _check_knowledge_insights(self, user_id: str) -> List[ProactiveInsight]:
        """Generate insights from knowledge graph patterns."""
        if not self.knowledge_graph:
            return []
        
        insights = []
        
        try:
            # Get user's entity statistics
            stats = await self.knowledge_graph.get_stats(user_id)
            
            if stats['total_triples'] < 5:
                return []  # Not enough data
            
            # Check for entities with many connections (important people/things)
            top_entities = await self.db.fetchall("""
                SELECT subject, COUNT(*) as cnt FROM kg_triples
                WHERE user_id=? GROUP BY subject
                HAVING cnt >= 3 ORDER BY cnt DESC LIMIT 3
            """, (user_id,))
            
            for entity_row in top_entities:
                entity = entity_row[0]
                
                # Get recent relations
                relations = await self.knowledge_graph.query_by_subject(user_id, entity, min_confidence=0.6)
                
                if relations:
                    # Generate insight about connected entity
                    predicates = [r.predicate for r in relations[:3]]
                    
                    insight = ProactiveInsight(
                        user_id=user_id,
                        insight_type=InsightType.KNOWLEDGE,
                        priority=InsightPriority.LOW,
                        message=f"💡 Tentang {entity.title()}: Vira ingat beberapa hal menarik yang pernah kamu ceritakan.",
                        context={
                            "entity": entity,
                            "relations": predicates,
                            "connection_count": len(relations)
                        },
                        expires_at=datetime.now() + timedelta(days=1)
                    )
                    insights.append(insight)
                    break  # Only one KG insight at a time
        
        except Exception as e:
            logger.error(f"[PROACTIVE] Knowledge insight check failed: {e}")
        
        return insights
    
    async def _check_date_insights(self, user_id: str) -> List[ProactiveInsight]:
        """Check for date-based insights like anniversaries."""
        insights = []
        today = datetime.now()
        
        # Check memories for date mentions that match today
        try:
            # Look for memories with date patterns
            rows = await self.db.fetchall("""
                SELECT summary, metadata FROM memories
                WHERE user_id=? AND status='active' 
                AND (summary LIKE '%birthday%' OR summary LIKE '%anniversary%' 
                     OR summary LIKE '%ultah%' OR summary LIKE '%tanggal%')
                LIMIT 10
            """, (user_id,))
            
            for row in rows:
                summary = row[0].lower()
                
                # Simple date matching (can be enhanced with NLP)
                month_day = today.strftime("%d-%m")
                month_name = today.strftime("%B").lower()
                
                if month_day in summary or month_name in summary:
                    insight = ProactiveInsight(
                        user_id=user_id,
                        insight_type=InsightType.ANNIVERSARY,
                        priority=InsightPriority.HIGH,
                        message=f"🎉 Vira ingat ada sesuatu spesial hari ini: {row[0][:100]}",
                        context={"memory_summary": row[0]},
                        expires_at=today.replace(hour=23, minute=59)
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.debug(f"[PROACTIVE] Date insight check failed: {e}")
        
        return insights[:1]  # Max 1 date insight
    
    async def _check_follow_ups(self, user_id: str) -> List[ProactiveInsight]:
        """Check for opportunities to follow up on previous conversations."""
        insights = []
        
        try:
            # Find recent memories that mentioned future events/plans
            rows = await self.db.fetchall("""
                SELECT summary, created_at FROM memories
                WHERE user_id=? AND status='active'
                AND (summary LIKE '%akan%' OR summary LIKE '%mau%' 
                     OR summary LIKE '%rencana%' OR summary LIKE '%besok%')
                AND created_at > ?
                ORDER BY created_at DESC LIMIT 5
            """, (user_id, datetime.now() - timedelta(days=3)))
            
            for row in rows:
                memory_age_hours = (datetime.now() - row[1]).total_seconds() / 3600 if row[1] else 0
                
                # Follow up on plans mentioned 1-3 days ago
                if 24 <= memory_age_hours <= 72:
                    insight = ProactiveInsight(
                        user_id=user_id,
                        insight_type=InsightType.FOLLOW_UP,
                        priority=InsightPriority.MEDIUM,
                        message=f"💬 Vira penasaran, bagaimana kelanjutan tentang: {row[0][:80]}...?",
                        context={"original_memory": row[0], "days_ago": int(memory_age_hours / 24)},
                        expires_at=datetime.now() + timedelta(hours=12)
                    )
                    insights.append(insight)
                    break  # Only one follow-up at a time
        
        except Exception as e:
            logger.debug(f"[PROACTIVE] Follow-up check failed: {e}")
        
        return insights
    
    def _can_send_insight(self, user_id: str) -> bool:
        """Check if we can send an insight (rate limiting)."""
        today = datetime.now().date()
        history = self._delivery_history.get(user_id, [])
        
        today_count = sum(1 for dt in history if dt.date() == today)
        return today_count < self.MAX_INSIGHTS_PER_DAY
    
    def _get_last_delivery(self, user_id: str, insight_type: InsightType) -> Optional[datetime]:
        """Get last delivery time for a specific insight type."""
        # This would be enhanced with DB lookup
        history = self._delivery_history.get(user_id, [])
        return history[-1] if history else None
    
    async def mark_insight_delivered(self, user_id: str, insight: ProactiveInsight):
        """Mark an insight as delivered."""
        import json
        
        await self.db.execute("""
            INSERT INTO proactive_insights 
            (user_id, insight_type, priority, message, context, delivered, delivered_at)
            VALUES (?, ?, ?, ?, ?, 1, ?)
        """, (
            user_id,
            insight.insight_type.value,
            insight.priority.value,
            insight.message,
            json.dumps(insight.context),
            datetime.now()
        ))
        
        self._delivery_history[user_id].append(datetime.now())
        insight.delivered = True
    
    async def get_best_insight(self, user_id: str) -> Optional[ProactiveInsight]:
        """Get the single best insight to send right now."""
        insights = await self.check_for_insights(user_id)
        return insights[0] if insights else None
    

    
    async def get_pending_insights_count(self, user_id: str) -> int:
        """Get count of pending insights for a user."""
        insights = await self.check_for_insights(user_id)
        return len(insights)
    
    async def get_insight_stats(self, user_id: str) -> Dict:
        """Get insight delivery statistics."""
        rows = await self.db.fetchall("""
            SELECT insight_type, COUNT(*) as cnt
            FROM proactive_insights
            WHERE user_id=? AND delivered=1
            GROUP BY insight_type
        """, (user_id,))
        
        total = await self.db.fetchone(
            "SELECT COUNT(*) FROM proactive_insights WHERE user_id=? AND delivered=1",
            (user_id,)
        )
        
        return {
            "total_delivered": total[0] if total else 0,
            "by_type": {row[0]: row[1] for row in rows} if rows else {},
            "remaining_today": self.MAX_INSIGHTS_PER_DAY - len([
                dt for dt in self._delivery_history.get(user_id, [])
                if dt.date() == datetime.now().date()
            ])
        }
    
    async def cleanup_old_insights(self, days_old: int = 30) -> int:
        """Remove old insight records."""
        cutoff = datetime.now() - timedelta(days=days_old)
        
        await self.db.execute(
            "DELETE FROM proactive_insights WHERE created_at < ?",
            (cutoff,)
        )
        
        await self.db.execute(
            "DELETE FROM user_activity_logs WHERE timestamp < ?",
            (cutoff,)
        )
        
        logger.info(f"[PROACTIVE] Cleaned up insights older than {days_old} days")
        return 0
