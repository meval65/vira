import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from src.database import DBConnection

logger = logging.getLogger(__name__)

class SchedulerService:
    CACHE_TTL = 60
    DUPLICATE_WINDOW = 5
    CANCEL_WINDOW = 30
    
    def __init__(self, db: DBConnection):
        self.db = db
        self._lock = threading.RLock()
        self._cache: Dict[str, Tuple[Any, datetime]] = {}

    def _get_cached(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        data, timestamp = self._cache[key]
        if (datetime.now() - timestamp).total_seconds() >= self.CACHE_TTL:
            del self._cache[key]
            return None
        return data

    def _set_cache(self, key: str, data: Any) -> None:
        self._cache[key] = (data, datetime.now())

    def _invalidate_cache(self, user_id: str) -> None:
        keys_to_delete = [k for k in self._cache if user_id in k]
        for key in keys_to_delete:
            self._cache.pop(key, None)

    def _validate_schedule_input(self, context: str, trigger_time: datetime) -> bool:
        return (
            context and 
            len(context.strip()) >= 3 and 
            trigger_time >= datetime.now()
        )

    def _get_time_window(self, target_time: datetime, minutes: int) -> Tuple[datetime, datetime]:
        return (
            target_time - timedelta(minutes=minutes),
            target_time + timedelta(minutes=minutes)
        )

    def _check_duplicate(self, cursor, user_id: str, context: str, start: datetime, end: datetime) -> bool:
        cursor.execute("""
            SELECT id FROM schedules 
            WHERE user_id = ? AND context = ? AND status = 'pending'
            AND scheduled_at BETWEEN ? AND ?
        """, (user_id, context, start, end))
        return cursor.fetchone() is not None

    def _find_existing_schedule(self, cursor, user_id: str, start: datetime, end: datetime) -> Optional[Tuple[int, str]]:
        cursor.execute("""
            SELECT id, context FROM schedules 
            WHERE user_id = ? AND status = 'pending'
            AND scheduled_at BETWEEN ? AND ?
        """, (user_id, start, end))
        row = cursor.fetchone()
        return (row[0], row[1]) if row else None

    def _merge_schedule(self, cursor, schedule_id: int, old_context: str, new_context: str, priority: int) -> Optional[int]:
        combined_context = f"{old_context} & {new_context}"
        try:
            cursor.execute("""
                UPDATE schedules 
                SET context = ?, priority = MAX(priority, ?)
                WHERE id = ?
            """, (combined_context, priority, schedule_id))
            self.db.commit()
            logger.info(f"[SCHEDULER] Merged schedule ID {schedule_id}: {combined_context}")
            return schedule_id
        except Exception as e:
            logger.error(f"[SCHEDULER] Merge failed: {e}")
            return None

    def _insert_schedule(self, cursor, user_id: str, trigger_time: datetime, context: str, priority: int) -> Optional[int]:
        try:
            cursor.execute("""
                INSERT INTO schedules (user_id, scheduled_at, context, status, priority, created_at)
                VALUES (?, ?, ?, 'pending', ?, ?)
            """, (user_id, trigger_time, context, priority, datetime.now()))
            self.db.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"[SCHEDULER] Insert failed: {e}")
            return None

    def add_schedule(self, user_id: str, trigger_time: datetime, context: str, priority: int = 0) -> Optional[int]:
        if not self._validate_schedule_input(context, trigger_time):
            return None

        with self._lock:
            cursor = self.db.get_cursor()
            context = context.strip()
            
            start, end = self._get_time_window(trigger_time, self.DUPLICATE_WINDOW)
            
            if self._check_duplicate(cursor, user_id, context, start, end):
                logger.info(f"[SCHEDULER] Duplicate blocked: {context} at {trigger_time}")
                return None

            existing = self._find_existing_schedule(cursor, user_id, start, end)
            
            if existing:
                schedule_id, old_context = existing
                result = self._merge_schedule(cursor, schedule_id, old_context, context, priority)
                if result:
                    self._invalidate_cache(user_id)
                return result

            result = self._insert_schedule(cursor, user_id, trigger_time, context, priority)
            if result:
                self._invalidate_cache(user_id)
            return result

    def get_upcoming_schedules_raw(self, user_id: str, limit: int = 5) -> List[Dict]:
        cursor = self.db.get_cursor()
        cursor.execute("""
            SELECT scheduled_at, context, priority 
            FROM schedules 
            WHERE user_id = ? AND status = 'pending' AND scheduled_at > ?
            ORDER BY scheduled_at ASC LIMIT ?
        """, (user_id, datetime.now(), limit))
        
        return [
            {"scheduled_at": row[0], "context": row[1], "priority": row[2]}
            for row in cursor.fetchall()
        ]
    
    def _find_schedules_to_cancel(self, cursor, user_id: str, start: datetime, end: datetime, context_hint: str) -> List[int]:
        like_query = f"%{context_hint}%"
        cursor.execute("""
            SELECT id FROM schedules 
            WHERE user_id = ? AND status = 'pending' 
            AND scheduled_at BETWEEN ? AND ?
            AND context LIKE ?
        """, (user_id, start, end, like_query))
        
        rows = cursor.fetchall()
        if rows:
            return [row[0] for row in rows]
        
        cursor.execute("""
            SELECT id FROM schedules 
            WHERE user_id = ? AND status = 'pending' 
            AND scheduled_at BETWEEN ? AND ?
        """, (user_id, start, end))
        
        return [row[0] for row in cursor.fetchall()]

    def cancel_schedule_by_context(self, user_id: str, time_hint: datetime, context_hint: str) -> bool:
        with self._lock:
            cursor = self.db.get_cursor()
            start, end = self._get_time_window(time_hint, self.CANCEL_WINDOW)
            
            schedule_ids = self._find_schedules_to_cancel(cursor, user_id, start, end, context_hint)
            
            if not schedule_ids:
                return False
            
            placeholders = ','.join('?' * len(schedule_ids))
            cursor.execute(
                f"UPDATE schedules SET status='cancelled' WHERE id IN ({placeholders})", 
                schedule_ids
            )
            self.db.commit()
            self._invalidate_cache(user_id)
            logger.info(f"[SCHEDULER] Cancelled {len(schedule_ids)} schedules for {user_id}")
            return True
        
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
                "id": row[0], 
                "user_id": row[1], 
                "scheduled_at": row[2],
                "context": row[3], 
                "priority": row[4]
            }
            for row in cursor.fetchall()
        ]

    def get_pending_schedule_for_user(self, user_id: str) -> Optional[Dict]:
        cache_key = f"pending_{user_id}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
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
                "id": row[0], 
                "scheduled_at": row[1],
                "context": row[2], 
                "priority": row[3]
            }
        
        self._set_cache(cache_key, result)
        return result

    def _format_schedule_line(self, dt: datetime, context: str, priority: int, now: datetime) -> Optional[str]:
        try:
            if isinstance(dt, str):
                dt = datetime.fromisoformat(dt)
            
            delta = (dt.date() - now.date()).days
            time_str = dt.strftime("%H:%M")
            
            day_labels = {
                0: "Hari ini",
                1: "Besok",
                2: "Lusa"
            }
            
            if delta in day_labels:
                day_str = day_labels[delta]
            elif 0 <= delta < 7:
                days_map = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
                day_str = days_map[dt.weekday()]
            else:
                day_str = dt.strftime("%d/%m")
            
            prefix = "⚠️ " if priority > 0 else ""
            return f"{prefix}- {day_str} {time_str}: {context}"
        except Exception as e:
            logger.warning(f"Format schedule failed: {e}")
            return None

    def get_upcoming_schedules(self, user_id: str, limit: int = 7, hours_ahead: int = 72) -> List[str]:
        cache_key = f"upcoming_{user_id}_{limit}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
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
        for dt_raw, context, priority in cursor.fetchall():
            formatted = self._format_schedule_line(dt_raw, context, priority, now)
            if formatted:
                results.append(formatted)
        
        self._set_cache(cache_key, results)
        return results

    def mark_as_executed(self, schedule_id: int, note: Optional[str] = None) -> bool:
        with self._lock:
            cursor = self.db.get_cursor()
            try:
                cursor.execute("SELECT user_id FROM schedules WHERE id = ?", (schedule_id,))
                row = cursor.fetchone()
                if not row:
                    return False
                
                cursor.execute("""
                    UPDATE schedules 
                    SET status='executed', executed_at=?, execution_note=?
                    WHERE id=?
                """, (datetime.now(), note, schedule_id))
                self.db.commit()
                self._invalidate_cache(row[0])
                return True
            except Exception as e:
                logger.error(f"[SCHEDULER] Mark executed failed: {e}")
                return False

    def cancel_schedule(self, schedule_id: int, user_id: Optional[str] = None) -> bool:
        with self._lock:
            cursor = self.db.get_cursor()
            try:
                query = "UPDATE schedules SET status='cancelled' WHERE id=?"
                params = [schedule_id]
                
                if user_id:
                    query += " AND user_id=?"
                    params.append(user_id)
                
                cursor.execute(query, params)
                success = cursor.rowcount > 0
                
                if success:
                    self.db.commit()
                    if user_id:
                        self._invalidate_cache(user_id)
                
                return success
            except Exception as e:
                logger.error(f"[SCHEDULER] Cancel failed: {e}")
                return False

    def get_schedule_history(self, user_id: str, limit: int = 20, include_pending: bool = False) -> List[Dict]:
        cursor = self.db.get_cursor()
        
        statuses = ['executed', 'cancelled']
        if include_pending:
            statuses.append('pending')
        
        placeholders = ','.join('?' * len(statuses))
        query = f"""
            SELECT id, scheduled_at, context, status, executed_at 
            FROM schedules 
            WHERE user_id=? AND status IN ({placeholders})
            ORDER BY scheduled_at DESC LIMIT ?
        """
        
        cursor.execute(query, (user_id, *statuses, limit))
        
        return [
            {
                "id": row[0], 
                "scheduled_at": row[1], 
                "context": row[2],
                "status": row[3], 
                "executed_at": row[4]
            }
            for row in cursor.fetchall()
        ]

    def cleanup_old_schedules(self, days_old: int = 30) -> int:
        with self._lock:
            try:
                cutoff = datetime.now() - timedelta(days=days_old)
                cursor = self.db.get_cursor()
                cursor.execute("""
                    DELETE FROM schedules 
                    WHERE status IN ('executed', 'cancelled') AND scheduled_at < ?
                """, (cutoff,))
                self.db.commit()
                return cursor.rowcount
            except Exception as e:
                logger.error(f"[SCHEDULER] Cleanup failed: {e}")
                return 0

    def get_schedule_stats(self, user_id: str) -> Dict[str, Any]:
        stats = {
            "pending": 0, 
            "executed": 0, 
            "cancelled": 0, 
            "total": 0, 
            "next_schedule": None
        }
        
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute(
                "SELECT status, COUNT(*) FROM schedules WHERE user_id=? GROUP BY status", 
                (user_id,)
            )
            
            for status, count in cursor.fetchall():
                if status in stats:
                    stats[status] = count
                    stats["total"] += count
            
            cursor.execute("""
                SELECT scheduled_at, context FROM schedules 
                WHERE user_id=? AND status='pending' AND scheduled_at > ?
                ORDER BY scheduled_at ASC LIMIT 1
            """, (user_id, datetime.now()))
            
            next_schedule = cursor.fetchone()
            if next_schedule:
                stats["next_schedule"] = {
                    "time": next_schedule[0], 
                    "context": next_schedule[1]
                }
                
        except Exception as e:
            logger.error(f"[SCHEDULER] Stats failed: {e}")
        
        return stats

    def reschedule(self, schedule_id: int, new_time: datetime, user_id: Optional[str] = None) -> bool:
        if new_time < datetime.now():
            return False
        
        with self._lock:
            try:
                query = "UPDATE schedules SET scheduled_at=? WHERE id=? AND status='pending'"
                params = [new_time, schedule_id]
                
                if user_id:
                    query += " AND user_id=?"
                    params.append(user_id)
                
                cursor = self.db.get_cursor()
                cursor.execute(query, params)
                success = cursor.rowcount > 0
                
                if success:
                    self.db.commit()
                    if user_id:
                        self._invalidate_cache(user_id)
                
                return success
            except Exception as e:
                logger.error(f"[SCHEDULER] Reschedule failed: {e}")
                return False

    def clear_cache(self) -> None:
        self._cache.clear()