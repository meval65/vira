import logging
import asyncio
import threading
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict
from src.database import DBConnection
from dateutil.relativedelta import relativedelta

logger = logging.getLogger(__name__)

class ScheduleAnalytics:
    def __init__(self):
        self._completion_history: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = threading.Lock()

    def record_completion(self, user_id: str, schedule_id: int, scheduled_time: datetime,
                         completed_time: datetime, was_on_time: bool):
        # Note: using asyncio lock would require async method; keeping sync for simplicity
        with self._lock:
            delay_minutes = (completed_time - scheduled_time).total_seconds() / 60
            entry = {
                'schedule_id': schedule_id,
                'scheduled': scheduled_time.isoformat(),
                'completed': completed_time.isoformat(),
                'delay_minutes': delay_minutes,
                'on_time': was_on_time
            }
            self._completion_history[user_id].append(entry)
            if len(self._completion_history[user_id]) > 100:
                self._completion_history[user_id].pop(0)

    def get_user_analytics(self, user_id: str) -> Dict:
        with self._lock:
            history = self._completion_history.get(user_id, [])
            if not history:
                return {
                    'total_completed': 0,
                    'on_time_rate': 0.0,
                    'avg_delay_minutes': 0.0,
                    'completion_trend': 'no_data'
                }
            total = len(history)
            on_time_count = sum(1 for h in history if h['on_time'])
            total_delay = sum(h['delay_minutes'] for h in history)
            recent_10 = history[-10:]
            recent_on_time = sum(1 for h in recent_10 if h['on_time'])
            return {
                'total_completed': total,
                'on_time_rate': round(on_time_count / total * 100, 2),
                'avg_delay_minutes': round(total_delay / total, 2),
                'recent_performance': round(recent_on_time / len(recent_10) * 100, 2) if recent_10 else 0.0,
                'completion_trend': 'improving' if recent_on_time / len(recent_10) > on_time_count / total else 'stable'
            }

class SchedulerService:
    CACHE_TTL = 300
    DUPLICATE_WINDOW = 5
    CANCEL_WINDOW = 30
    RECURRING_TYPES = ['daily', 'weekly', 'monthly']

    def __init__(self, db: DBConnection):
        self.db = db
        self._lock = asyncio.Lock()
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self.analytics = ScheduleAnalytics()

    def _get_cache_key(self, user_id: str, prefix: str) -> str:
        return f"{user_id}:{prefix}"

    def _get_from_cache(self, key: str) -> Optional[Any]:
        if key in self._cache:
            data, timestamp = self._cache[key]
            if (datetime.now() - timestamp).total_seconds() < self.CACHE_TTL:
                return data
            del self._cache[key]
        return None

    def _update_cache(self, key: str, data: Any):
        self._cache[key] = (data, datetime.now())

    def _invalidate_user_cache(self, user_id: str):
        keys = list(self._cache.keys())
        prefix = f"{user_id}:"
        for k in keys:
            if k.startswith(prefix):
                self._cache.pop(k, None)

    async def add_schedule(self, user_id: str, trigger_time: datetime, context: str, priority: int = 0,
                           metadata: Dict = None, recurring: str = None) -> Tuple[Optional[int], str]:
        if not context or len(context.strip()) < 3 or trigger_time < datetime.now():
            return None, "invalid_input"
        if recurring and recurring not in self.RECURRING_TYPES:
            return None, "invalid_recurring_type"
        async with self._lock:
            try:
                context = context.strip()
                if metadata is None:
                    metadata = {}
                if recurring:
                    metadata['recurring'] = recurring
                    metadata['created_at'] = datetime.now().isoformat()
                meta_json = json.dumps(metadata) if metadata else None
                start = trigger_time - timedelta(minutes=self.DUPLICATE_WINDOW)
                end = trigger_time + timedelta(minutes=self.DUPLICATE_WINDOW)
                existing = await self.db.fetchone(
                    "SELECT id, context, priority FROM schedules WHERE user_id = ? AND status = 'pending' AND scheduled_at BETWEEN ? AND ?",
                    (user_id, start, end)
                )
                if existing:
                    ex_id, ex_ctx, ex_prio = existing
                    if ex_ctx == context:
                        return ex_id, "duplicate"
                    new_ctx = f"{ex_ctx} & {context}"
                    new_prio = max(ex_prio, priority)
                    await self.db.execute(
                        "UPDATE schedules SET context = ?, priority = ?, metadata = ? WHERE id = ?",
                        (new_ctx, new_prio, meta_json, ex_id)
                    )
                    self._invalidate_user_cache(user_id)
                    return ex_id, "merged"
                await self.db.execute(
                    "INSERT INTO schedules (user_id, scheduled_at, context, status, priority, created_at, metadata) VALUES (?, ?, ?, 'pending', ?, ?, ?)",
                    (user_id, trigger_time, context, priority, datetime.now(), meta_json)
                )
                self._invalidate_user_cache(user_id)
                row = await self.db.fetchone("SELECT last_insert_rowid()", ())
                return row[0], "created"
            except Exception as e:
                logger.error(f"[SCHEDULER] Add failed: {e}")
                await self.db.rollback()
                return None, "error"

    async def snooze_schedule(self, schedule_id: int, snooze_minutes: int = 15) -> bool:
        async with self._lock:
            try:
                row = await self.db.fetchone(
                    "SELECT user_id, scheduled_at FROM schedules WHERE id = ?",
                    (schedule_id,)
                )
                if not row:
                    return False
                user_id, current_time = row
                if isinstance(current_time, str):
                    current_time = datetime.fromisoformat(current_time)
                new_time = max(datetime.now(), current_time) + timedelta(minutes=snooze_minutes)
                await self.db.execute(
                    "UPDATE schedules SET scheduled_at = ?, status = 'pending', priority = priority + 1 WHERE id = ?",
                    (new_time, schedule_id)
                )
                self._invalidate_user_cache(user_id)
                return True
            except Exception as e:
                logger.error(f"[SCHEDULER] Snooze failed: {e}")
                await self.db.rollback()
                return False

    async def get_upcoming_schedules(self, user_id: str, limit: int = 7, hours_ahead: int = 72) -> List[str]:
        cache_key = self._get_cache_key(user_id, f"upcoming_fmt_{limit}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        now = datetime.now()
        end = now + timedelta(hours=hours_ahead)
        try:
            rows = await self.db.fetchall(
                "SELECT scheduled_at, context, priority, metadata FROM schedules WHERE user_id = ? AND status = 'pending' AND scheduled_at > ? AND scheduled_at <= ? ORDER BY scheduled_at ASC LIMIT ?",
                (user_id, now, end, limit)
            )
            results = []
            for dt, ctx, prio, meta_str in rows:
                metadata = json.loads(meta_str) if meta_str else {}
                recurring = metadata.get('recurring')
                line = self._format_schedule_line(dt, ctx, prio, now, recurring)
                if line:
                    results.append(line)
            self._update_cache(cache_key, results)
            return results
        except Exception:
            return []

    async def get_upcoming_schedules_raw(self, user_id: str, limit: int = 5) -> List[Dict]:
        cache_key = self._get_cache_key(user_id, f"upcoming_raw_{limit}")
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            return cached
        try:
            rows = await self.db.fetchall(
                "SELECT id, scheduled_at, context, priority, metadata FROM schedules WHERE user_id = ? AND status = 'pending' AND scheduled_at > ? ORDER BY scheduled_at ASC LIMIT ?",
                (user_id, datetime.now(), limit)
            )
            data = []
            for r in rows:
                meta = json.loads(r[4]) if r[4] else {}
                data.append({
                    "id": r[0],
                    "scheduled_at": r[1],
                    "context": r[2],
                    "priority": r[3],
                    "metadata": meta,
                    "recurring": meta.get('recurring')
                })
            self._update_cache(cache_key, data)
            return data
        except Exception:
            return []

    async def cancel_schedule_by_context(self, user_id: str, time_hint: datetime, context_hint: str) -> bool:
        async with self._lock:
            try:
                start = time_hint - timedelta(minutes=self.CANCEL_WINDOW)
                end = time_hint + timedelta(minutes=self.CANCEL_WINDOW)
                rows = await self.db.fetchall(
                    "SELECT id FROM schedules WHERE user_id=? AND status='pending' AND scheduled_at BETWEEN ? AND ? AND context LIKE ?",
                    (user_id, start, end, f"%{context_hint}%")
                )
                if not rows:
                    return False
                ids = [r[0] for r in rows]
                placeholders = ','.join('?' * len(ids))
                await self.db.execute(f"UPDATE schedules SET status='cancelled' WHERE id IN ({placeholders})", ids)
                self._invalidate_user_cache(user_id)
                return True
            except Exception:
                await self.db.rollback()
                return False

    async def get_pending_trigger(self, user_id: str) -> Optional[Dict]:
        key = self._get_cache_key(user_id, "trigger")
        cached = self._get_from_cache(key)
        if cached:
            return cached
        row = await self.db.fetchone(
            "SELECT id, scheduled_at, context, priority, metadata FROM schedules WHERE user_id = ? AND status = 'pending' AND scheduled_at <= ? ORDER BY priority DESC, scheduled_at ASC LIMIT 1",
            (user_id, datetime.now())
        )
        if row:
            metadata = json.loads(row[4]) if row[4] else {}
            res = {
                "id": row[0],
                "context": row[2],
                "scheduled_at": row[1],
                "metadata": metadata,
                "recurring": metadata.get('recurring')
            }
            self._update_cache(key, res)
            return res
        return None

    async def mark_as_executed(self, schedule_id: int, note: Optional[str] = None) -> bool:
        async with self._lock:
            try:
                row = await self.db.fetchone(
                    "SELECT user_id, metadata, scheduled_at FROM schedules WHERE id = ?",
                    (schedule_id,)
                )
                if not row:
                    return False
                user_id, metadata_str, scheduled_at = row
                if isinstance(scheduled_at, str):
                    scheduled_at = datetime.fromisoformat(scheduled_at)
                completed_time = datetime.now()
                was_on_time = (completed_time - scheduled_at).total_seconds() / 60 <= 15
                self.analytics.record_completion(user_id, schedule_id, scheduled_at, completed_time, was_on_time)
                await self.db.execute(
                    "UPDATE schedules SET status='executed', executed_at=?, execution_note=? WHERE id=?",
                    (completed_time, note, schedule_id)
                )
                if metadata_str:
                    metadata = json.loads(metadata_str)
                    if 'recurring' in metadata:
                        sched_row = await self.db.fetchone(
                            "SELECT context, priority FROM schedules WHERE id = ?",
                            (schedule_id,)
                        )
                        if sched_row:
                            context, priority = sched_row
                            next_time = self._calculate_next_occurrence(scheduled_at, metadata['recurring'])
                            if next_time:
                                await self.add_schedule(user_id, next_time, context, priority, metadata, metadata['recurring'])
                if user_id:
                    self._invalidate_user_cache(user_id)
                return True
            except Exception as e:
                logger.error(f"[SCHEDULER] Mark executed failed: {e}")
                await self.db.rollback()
                return False

    async def get_due_schedules_batch(self, batch_size: int = 50) -> List[Dict]:
        try:
            rows = await self.db.fetchall(
                "SELECT id, user_id, context, priority, metadata FROM schedules WHERE status = 'pending' AND scheduled_at <= ? ORDER BY scheduled_at ASC LIMIT ?",
                (datetime.now(), batch_size)
            )
            results = []
            for r in rows:
                results.append({
                    "id": r[0],
                    "user_id": r[1],
                    "context": r[2],
                    "priority": r[3],
                    "metadata": json.loads(r[4]) if r[4] else {}
                })
            return results
        except Exception:
            return []

    async def cleanup_old_schedules(self, days_old: int = 30) -> int:
        async with self._lock:
            try:
                cutoff = datetime.now() - timedelta(days=days_old)
                await self.db.execute(
                    "DELETE FROM schedules WHERE status IN ('executed', 'cancelled') AND scheduled_at < ?",
                    (cutoff,)
                )
                self._cache.clear()
                # aiosqlite does not provide rowcount; assume success
                return 0
            except Exception:
                return 0

    async def get_schedule_stats(self, user_id: str) -> Dict:
        stats = {"pending": 0, "executed": 0, "cancelled": 0, "recurring": 0}
        try:
            rows = await self.db.fetchall(
                "SELECT status, COUNT(*) FROM schedules WHERE user_id=? GROUP BY status",
                (user_id,)
            )
            for status, count in rows:
                if status in stats:
                    stats[status] = count
            rec_row = await self.db.fetchone(
                "SELECT COUNT(*) FROM schedules WHERE user_id=? AND status='pending' AND metadata LIKE '%recurring%'",
                (user_id,)
            )
            if rec_row:
                stats['recurring'] = rec_row[0]
        except Exception:
            pass
        return stats

    async def update_schedule_priority(self, schedule_id: int, new_priority: int) -> bool:
        async with self._lock:
            try:
                row = await self.db.fetchone(
                    "SELECT user_id FROM schedules WHERE id = ?",
                    (schedule_id,)
                )
                if not row:
                    return False
                user_id = row[0]
                await self.db.execute(
                    "UPDATE schedules SET priority = ? WHERE id = ?",
                    (new_priority, schedule_id)
                )
                self._invalidate_user_cache(user_id)
                return True
            except Exception:
                await self.db.rollback()
                return False

    def _calculate_next_occurrence(self, base_time: datetime, recurrence: str) -> Optional[datetime]:
        """Calculate the next occurrence datetime based on recurrence type.
        Supports 'daily', 'weekly', 'monthly'. Returns None for unknown types."""
        if recurrence == 'daily':
            return base_time + timedelta(days=1)
        elif recurrence == 'weekly':
            return base_time + timedelta(weeks=1)
        elif recurrence == 'monthly':
            return base_time + relativedelta(months=1)
        return None

    def _format_schedule_line(self, dt: Any, context: str, priority: int, now: datetime, recurring: str = None) -> str:
        try:
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt)
            diff = dt.date() - now.date()
            delta_days = diff.days
            time_str = dt.strftime("%H:%M")
            if delta_days == 0:
                day_str = "Hari ini"
            elif delta_days == 1:
                day_str = "Besok"
            elif 0 <= delta_days < 7:
                day_str = dt.strftime("%A")
            else:
                day_str = dt.strftime("%d %b")
            icon = "🔴" if priority > 1 else "🟡" if priority == 1 else "🔵"
            recurring_label = ""
            if recurring:
                recurring_labels = {'daily': '🔄', 'weekly': '📅', 'monthly': '🗓️'}
                recurring_label = f" {recurring_labels.get(recurring, '🔄')}"
            return f"{icon} {day_str}, {time_str} - {context}{recurring_label}"
        except Exception:
            return f"• {context}"