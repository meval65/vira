import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from src.database import DBConnection

logger = logging.getLogger(__name__)

class SchedulerService:
    def __init__(self, db: DBConnection):
        self.db = db
        self._lock = threading.RLock()
        self._cache = {}
        self._cache_ttl = 60

    def _get_cached(self, key: str) -> Any:
        if key in self._cache:
            data, ts = self._cache[key]
            if (datetime.now() - ts).total_seconds() < self._cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (data, datetime.now())

    def _invalidate_cache(self, user_id: str):
        keys = [k for k in self._cache if user_id in k]
        for k in keys:
            self._cache.pop(k, None)

    def add_schedule(self, user_id: str, trigger_time: datetime, context: str, priority: int = 0):
        if not context or len(context.strip()) < 3 or trigger_time < datetime.now():
            return

        with self._lock:
            cursor = self.db.get_cursor()
            context = context.strip()
            
            check_start = trigger_time - timedelta(minutes=1)
            check_end = trigger_time + timedelta(minutes=1)
            
            cursor.execute("""
                SELECT 1 FROM schedules 
                WHERE user_id = ? AND context = ? AND status = 'pending'
                AND scheduled_at BETWEEN ? AND ?
            """, (user_id, context, check_start, check_end))
            
            if cursor.fetchone():
                return

            try:
                cursor.execute("""
                    INSERT INTO schedules (user_id, scheduled_at, context, status, priority, created_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                """, (user_id, trigger_time, context, priority, datetime.now()))
                self.db.commit()
                self._invalidate_cache(user_id)
            except Exception as e:
                logger.error(f"Add schedule failed: {e}")

    def get_due_schedules(self, lookback_minutes: int = 5) -> List[Dict]:
        cursor = self.db.get_cursor()
        now = datetime.now()
        start = now - timedelta(minutes=lookback_minutes)
        
        cursor.execute("""
            SELECT id, user_id, scheduled_at, context, priority 
            FROM schedules 
            WHERE status = 'pending' AND scheduled_at BETWEEN ? AND ?
            ORDER BY priority DESC, scheduled_at ASC
        """, (start, now))
        
        return [
            {
                "id": r[0], "user_id": r[1], "scheduled_at": r[2],
                "context": r[3], "priority": r[4]
            }
            for r in cursor.fetchall()
        ]

    def get_pending_schedule_for_user(self, user_id: str) -> Optional[Dict]:
        cache_key = f"pending_{user_id}"
        cached = self._get_cached(cache_key)
        if cached: return cached
        
        cursor = self.db.get_cursor()
        cursor.execute("""
            SELECT id, scheduled_at, context, priority 
            FROM schedules 
            WHERE user_id = ? AND status = 'pending' AND scheduled_at <= ?
            ORDER BY priority DESC, scheduled_at ASC LIMIT 1
        """, (user_id, datetime.now()))
        
        row = cursor.fetchone()
        result = None
        if row:
            result = {
                "id": row[0], "scheduled_at": row[1],
                "context": row[2], "priority": row[3]
            }
        
        self._set_cache(cache_key, result)
        return result

    def get_upcoming_schedules(self, user_id: str, limit: int = 7, hours_ahead: int = 72) -> List[str]:
        cache_key = f"upcoming_{user_id}_{limit}"
        cached = self._get_cached(cache_key)
        if cached: return cached
        
        cursor = self.db.get_cursor()
        now = datetime.now()
        end = now + timedelta(hours=hours_ahead)
        
        cursor.execute("""
            SELECT scheduled_at, context, priority 
            FROM schedules 
            WHERE user_id = ? AND status = 'pending' 
            AND scheduled_at > ? AND scheduled_at <= ?
            ORDER BY scheduled_at ASC LIMIT ?
        """, (user_id, now, end, limit))
        
        results = []
        days_map = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
        
        for dt_raw, ctx, prio in cursor.fetchall():
            try:
                dt = datetime.fromisoformat(str(dt_raw)) if isinstance(dt_raw, str) else dt_raw
                delta = (dt.date() - now.date()).days
                time_str = dt.strftime("%H:%M")
                
                if delta == 0:
                    day_str = "Hari ini"
                elif delta == 1:
                    day_str = "Besok"
                elif delta == 2:
                    day_str = "Lusa"
                elif 0 <= delta < 7:
                    day_str = days_map[dt.weekday()]
                else:
                    day_str = dt.strftime("%d/%m")
                
                prefix = "⚠️ " if prio > 0 else ""
                results.append(f"{prefix}- {day_str} {time_str}: {ctx}")
            except Exception:
                continue
        
        self._set_cache(cache_key, results)
        return results

    def mark_as_executed(self, schedule_id: int, note: str = None):
        with self._lock:
            cursor = self.db.get_cursor()
            try:
                cursor.execute("SELECT user_id FROM schedules WHERE id = ?", (schedule_id,))
                row = cursor.fetchone()
                if row:
                    cursor.execute("""
                        UPDATE schedules SET status='executed', executed_at=?, execution_note=?
                        WHERE id=?
                    """, (datetime.now(), note, schedule_id))
                    self.db.commit()
                    self._invalidate_cache(row[0])
            except Exception as e:
                logger.error(f"Mark executed failed: {e}")

    def cancel_schedule(self, schedule_id: int, user_id: str = None) -> bool:
        with self._lock:
            cursor = self.db.get_cursor()
            try:
                query = "UPDATE schedules SET status='cancelled' WHERE id=?"
                params = [schedule_id]
                
                if user_id:
                    query += " AND user_id=?"
                    params.append(user_id)
                
                cursor.execute(query, params)
                if cursor.rowcount > 0:
                    self.db.commit()
                    if user_id: self._invalidate_cache(user_id)
                    return True
            except Exception:
                pass
            return False

    def get_schedule_history(self, user_id: str, limit: int = 20, include_pending: bool = False) -> List[Dict]:
        cursor = self.db.get_cursor()
        statuses = "'executed', 'cancelled'"
        if include_pending:
            statuses += ", 'pending'"
            
        cursor.execute(f"""
            SELECT id, scheduled_at, context, status, executed_at 
            FROM schedules WHERE user_id=? AND status IN ({statuses})
            ORDER BY scheduled_at DESC LIMIT ?
        """, (user_id, limit))
        
        return [
            {
                "id": r[0], "scheduled_at": r[1], "context": r[2],
                "status": r[3], "executed_at": r[4]
            }
            for r in cursor.fetchall()
        ]

    def cleanup_old_schedules(self, days_old: int = 30) -> int:
        with self._lock:
            try:
                cutoff = datetime.now() - timedelta(days=days_old)
                self.db.get_cursor().execute("""
                    DELETE FROM schedules WHERE status IN ('executed', 'cancelled') 
                    AND scheduled_at < ?
                """, (cutoff,))
                self.db.commit()
                return self.db.get_cursor().rowcount
            except Exception:
                return 0

    def get_schedule_stats(self, user_id: str) -> Dict:
        stats = {"pending": 0, "executed": 0, "cancelled": 0, "total": 0, "next_schedule": None}
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute("SELECT status, COUNT(*) FROM schedules WHERE user_id=? GROUP BY status", (user_id,))
            for status, count in cursor.fetchall():
                if status in stats:
                    stats[status] = count
                    stats["total"] += count
            
            cursor.execute("""
                SELECT scheduled_at, context FROM schedules 
                WHERE user_id=? AND status='pending' AND scheduled_at > ?
                ORDER BY scheduled_at ASC LIMIT 1
            """, (user_id, datetime.now()))
            
            nxt = cursor.fetchone()
            if nxt:
                stats["next_schedule"] = {"time": nxt[0], "context": nxt[1]}
                
        except Exception:
            pass
        return stats

    def reschedule(self, schedule_id: int, new_time: datetime, user_id: str = None) -> bool:
        if new_time < datetime.now(): return False
        
        with self._lock:
            try:
                query = "UPDATE schedules SET scheduled_at=? WHERE id=? AND status='pending'"
                params = [new_time, schedule_id]
                if user_id:
                    query += " AND user_id=?"
                    params.append(user_id)
                
                cursor = self.db.get_cursor()
                cursor.execute(query, params)
                if cursor.rowcount > 0:
                    self.db.commit()
                    if user_id: self._invalidate_cache(user_id)
                    return True
            except Exception:
                pass
            return False

    def clear_cache(self):
        self._cache.clear()