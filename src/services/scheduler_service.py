import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from src.database import DBConnection
import threading

logger = logging.getLogger(__name__)

class SchedulerService:
    def __init__(self, db: DBConnection):
        self.db = db
        self._lock = threading.RLock()
        self._cache = {}
        self._cache_timeout = 60

    def add_schedule(self, user_id: str, trigger_time: datetime, context: str, priority: int = 0):
        with self._lock:
            cursor = self.db.get_cursor()
            
            if not context or len(context.strip()) < 3:
                logger.warning("[SCHEDULER] Invalid context, skipping")
                return
            
            if trigger_time < datetime.now():
                logger.warning(f"[SCHEDULER] Cannot schedule in the past: {trigger_time}")
                return
            
            cursor.execute("""
                SELECT id FROM schedules 
                WHERE user_id = ? AND context = ? AND status = 'pending'
                AND ABS(julianday(scheduled_at) - julianday(?)) < 0.02
            """, (user_id, context.strip(), trigger_time))
            
            if cursor.fetchone():
                logger.info("[SCHEDULER] Duplicate schedule detected, skipping")
                return
            
            try:
                cursor.execute("""
                    INSERT INTO schedules (user_id, scheduled_at, context, status, priority, created_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                """, (user_id, trigger_time, context.strip(), priority, datetime.now()))
                self.db.commit()
                
                self._invalidate_cache(user_id)
                
                logger.info(f"[SCHEDULER] Added for {user_id} at {trigger_time.isoformat()}")
            except Exception as e:
                logger.error(f"[SCHEDULER] Add failed: {e}")

    def get_due_schedules(self, lookback_minutes: int = 5) -> List[Dict]:
        cursor = self.db.get_cursor()
        now = datetime.now()
        lookback_time = now - timedelta(minutes=lookback_minutes)
        
        cursor.execute("""
            SELECT id, user_id, scheduled_at, context, priority 
            FROM schedules 
            WHERE status = 'pending' 
            AND scheduled_at <= ? 
            AND scheduled_at >= ?
            ORDER BY priority DESC, scheduled_at ASC
        """, (now, lookback_time))
        
        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            results.append({
                "id": row["id"],
                "user_id": row["user_id"],
                "scheduled_at": row["scheduled_at"],
                "context": row["context"],
                "priority": row["priority"] if len(row) > 4 else 0
            })
        
        return results

    def get_pending_schedule_for_user(self, user_id: str) -> Optional[Dict]:
        cache_key = f"pending_{user_id}"
        
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_timeout:
                return cached_data
        
        cursor = self.db.get_cursor()
        now = datetime.now()
        
        cursor.execute("""
            SELECT id, scheduled_at, context, priority 
            FROM schedules 
            WHERE user_id = ? AND status = 'pending' AND scheduled_at <= ?
            ORDER BY priority DESC, scheduled_at ASC
            LIMIT 1
        """, (user_id, now))
        
        row = cursor.fetchone()
        result = None
        
        if row:
            result = {
                "id": row["id"],
                "context": row["context"],
                "scheduled_at": row["scheduled_at"],
                "priority": row["priority"] if len(row) > 3 else 0
            }
        
        self._cache[cache_key] = (result, datetime.now())
        return result

    def get_upcoming_schedules(self, user_id: str, limit: int = 7, hours_ahead: int = 72) -> List[str]:
        cache_key = f"upcoming_{user_id}_{limit}_{hours_ahead}"
        
        if cache_key in self._cache:
            cached_data, cached_time = self._cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < 300:
                return cached_data
        
        cursor = self.db.get_cursor()
        now = datetime.now()
        future_limit = now + timedelta(hours=hours_ahead)
        
        cursor.execute("""
            SELECT scheduled_at, context, priority 
            FROM schedules 
            WHERE user_id = ? AND status = 'pending' 
            AND scheduled_at > ? 
            AND scheduled_at <= ?
            ORDER BY scheduled_at ASC
            LIMIT ?
        """, (user_id, now, future_limit, limit))
        
        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            dt = row["scheduled_at"]
            if isinstance(dt, str):
                try:
                    dt = datetime.fromisoformat(dt)
                except:
                    continue
            
            time_str = dt.strftime("%H:%M")
            delta = dt - now
            
            if delta.days == 0:
                if delta.total_seconds() < 3600:
                    day_str = f"Dalam {int(delta.total_seconds() / 60)} menit"
                else:
                    day_str = "Hari ini"
            elif delta.days == 1:
                day_str = "Besok"
            elif delta.days == 2:
                day_str = "Lusa"
            elif delta.days < 7:
                day_names = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
                day_str = day_names[dt.weekday()]
            else:
                day_str = dt.strftime("%d/%m")
            
            priority_marker = "⚠️ " if (len(row) > 2 and row["priority"] > 0) else ""
            results.append(f"{priority_marker}- {day_str} pukul {time_str}: {row['context']}")
        
        self._cache[cache_key] = (results, datetime.now())
        return results

    def mark_as_executed(self, schedule_id: int, execution_note: str = None):
        with self._lock:
            cursor = self.db.get_cursor()
            
            try:
                cursor.execute(
                    "SELECT user_id FROM schedules WHERE id = ?", 
                    (schedule_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    user_id = row["user_id"]
                    
                    cursor.execute("""
                        UPDATE schedules 
                        SET status = 'executed', 
                            executed_at = ?,
                            execution_note = ?
                        WHERE id = ?
                    """, (datetime.now(), execution_note, schedule_id))
                    
                    self.db.commit()
                    self._invalidate_cache(user_id)
                    
                    logger.info(f"[SCHEDULER] Marked {schedule_id} as executed")
            except Exception as e:
                logger.error(f"[SCHEDULER] Update failed: {e}")

    def cancel_schedule(self, schedule_id: int, user_id: str = None) -> bool:
        with self._lock:
            cursor = self.db.get_cursor()
            
            try:
                if user_id:
                    cursor.execute(
                        "UPDATE schedules SET status = 'cancelled' WHERE id = ? AND user_id = ?",
                        (schedule_id, user_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE schedules SET status = 'cancelled' WHERE id = ?",
                        (schedule_id,)
                    )
                
                affected = cursor.rowcount
                self.db.commit()
                
                if user_id:
                    self._invalidate_cache(user_id)
                
                return affected > 0
            except Exception as e:
                logger.error(f"[SCHEDULER] Cancel failed: {e}")
                return False

    def get_schedule_history(self, user_id: str, limit: int = 20, 
                            include_pending: bool = False) -> List[Dict]:
        cursor = self.db.get_cursor()
        
        statuses = ['executed', 'cancelled']
        if include_pending:
            statuses.append('pending')
        
        placeholders = ','.join('?' * len(statuses))
        
        cursor.execute(f"""
            SELECT id, scheduled_at, context, status, executed_at, created_at
            FROM schedules 
            WHERE user_id = ? AND status IN ({placeholders})
            ORDER BY scheduled_at DESC
            LIMIT ?
        """, [user_id] + statuses + [limit])
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row["id"],
                "scheduled_at": row["scheduled_at"],
                "context": row["context"],
                "status": row["status"],
                "executed_at": row["executed_at"] if len(row) > 4 else None,
                "created_at": row["created_at"] if len(row) > 5 else None
            })
        
        return results

    def cleanup_old_schedules(self, days_old: int = 30) -> int:
        with self._lock:
            cursor = self.db.get_cursor()
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            try:
                cursor.execute("""
                    DELETE FROM schedules 
                    WHERE status IN ('executed', 'cancelled') 
                    AND scheduled_at < ?
                """, (cutoff_date,))
                
                deleted = cursor.rowcount
                self.db.commit()
                
                logger.info(f"[SCHEDULER] Cleaned up {deleted} old schedules")
                return deleted
            except Exception as e:
                logger.error(f"[SCHEDULER] Cleanup failed: {e}")
                return 0

    def get_schedule_stats(self, user_id: str) -> Dict:
        cursor = self.db.get_cursor()
        stats = {
            "pending": 0,
            "executed": 0,
            "cancelled": 0,
            "total": 0,
            "next_schedule": None
        }
        
        try:
            cursor.execute("""
                SELECT status, COUNT(*) 
                FROM schedules 
                WHERE user_id = ?
                GROUP BY status
            """, (user_id,))
            
            for row in cursor.fetchall():
                stats[row[0]] = row[1]
                stats["total"] += row[1]
            
            cursor.execute("""
                SELECT scheduled_at, context 
                FROM schedules 
                WHERE user_id = ? AND status = 'pending' 
                AND scheduled_at > ?
                ORDER BY scheduled_at ASC
                LIMIT 1
            """, (user_id, datetime.now()))
            
            next_row = cursor.fetchone()
            if next_row:
                stats["next_schedule"] = {
                    "time": next_row[0],
                    "context": next_row[1]
                }
        except Exception as e:
            logger.error(f"[SCHEDULER] Stats failed: {e}")
        
        return stats

    def reschedule(self, schedule_id: int, new_time: datetime, user_id: str = None) -> bool:
        with self._lock:
            cursor = self.db.get_cursor()
            
            if new_time < datetime.now():
                logger.warning("[SCHEDULER] Cannot reschedule to past time")
                return False
            
            try:
                query = "UPDATE schedules SET scheduled_at = ? WHERE id = ? AND status = 'pending'"
                params = [new_time, schedule_id]
                
                if user_id:
                    query += " AND user_id = ?"
                    params.append(user_id)
                
                cursor.execute(query, params)
                affected = cursor.rowcount
                self.db.commit()
                
                if user_id and affected > 0:
                    self._invalidate_cache(user_id)
                
                return affected > 0
            except Exception as e:
                logger.error(f"[SCHEDULER] Reschedule failed: {e}")
                return False

    def _invalidate_cache(self, user_id: str):
        keys_to_remove = [k for k in self._cache.keys() if user_id in k]
        for key in keys_to_remove:
            del self._cache[key]

    def clear_cache(self):
        self._cache.clear()
        logger.info("[SCHEDULER] Cache cleared")