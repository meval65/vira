import json
import os
import logging
import asyncio
import time
import datetime
from collections import deque
from typing import List, Optional, Dict, Any, Callable

import PIL.Image
from google.genai import types

logger = logging.getLogger(__name__)


class SessionManager:
    """Async-compatible session manager for chat history and metadata."""
    
    def __init__(self, max_history: int = 40):
        self.sessions: Dict[str, deque] = {}
        self.meta_data: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        
        self.MAX_HISTORY = max_history
        self.SESSION_DIR = "storage/sessions"
        self.IMG_PREFIX = ":::IMG_PATH:::"
        
        self._cleanup_lock = asyncio.Lock()
        self._last_cleanup = time.time()
        
        self.CLEANUP_INTERVAL = 1800
        self.SESSION_TIMEOUT = 7200
        self.MEMORY_ANALYSIS_INTERVAL = 600
        self.NEW_MEMORY_THRESHOLD = 5
        
        os.makedirs(self.SESSION_DIR, exist_ok=True)
    
    def _get_lock(self, user_id: str) -> asyncio.Lock:
        """Get or create an async lock for a user."""
        if user_id not in self._locks:
            self._locks[user_id] = asyncio.Lock()
        return self._locks[user_id]

    # Backward compatibility - sync lock wrapper for non-critical operations
    # For truly sync contexts that can't use async
    _sync_locks: Dict[str, object] = {}
    
    def get_lock(self, user_id: str):
        """Get a simple context manager for backward compatibility.
        Note: For new code, prefer async patterns."""
        import threading
        if user_id not in SessionManager._sync_locks:
            SessionManager._sync_locks[user_id] = threading.Lock()
        return SessionManager._sync_locks[user_id]

    async def cleanup_inactive_sessions(self):
        """Clean up inactive sessions to free memory."""
        now = time.time()
        if now - self._last_cleanup < self.CLEANUP_INTERVAL:
            return
        
        if self._cleanup_lock.locked():
            return
        
        async with self._cleanup_lock:
            try:
                inactive_users = self._get_inactive_users()
                for user_id in inactive_users:
                    await self._save_and_unload(user_id)
                
                self._last_cleanup = now
                if inactive_users:
                    logger.info(f"Unloaded {len(inactive_users)} sessions")
            
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def _get_inactive_users(self) -> List[str]:
        current_time = datetime.datetime.now()
        inactive_users = []
        
        for uid in list(self.meta_data.keys()):
            last_ts = self.meta_data[uid].get('last_interaction')
            if last_ts and (current_time - last_ts).total_seconds() > self.SESSION_TIMEOUT:
                inactive_users.append(uid)
        
        return inactive_users
    
    async def _save_and_unload(self, user_id: str):
        """Save session to disk and unload from memory."""
        lock = self._get_lock(user_id)
        async with lock:
            if user_id in self.sessions:
                await asyncio.to_thread(self._save_session_to_disk_sync, user_id)
                self.sessions.pop(user_id, None)
                self.meta_data.pop(user_id, None)
                self._locks.pop(user_id, None)

    def get_session(self, user_id: str) -> deque:
        """Get session history for a user (sync, loads from disk if needed)."""
        if user_id not in self.sessions:
            self._load_session(user_id)
        return self.sessions.get(user_id, deque())
    
    def get_metadata(self, user_id: str, key: str, default: Any = None) -> Any:
        if user_id not in self.meta_data:
            self._load_session(user_id)
        return self.meta_data.get(user_id, {}).get(key, default)

    def update_metadata(self, user_id: str, updates: Dict[str, Any]):
        with self.get_lock(user_id):
            if user_id not in self.meta_data:
                self._load_session(user_id)
            self.meta_data[user_id].update(updates)
    
    def update_session(self, user_id: str, user_text: str, ai_text: str, 
                        image_path: str = None, on_limit_reached=None):
        with self.get_lock(user_id):
            if user_id not in self.sessions:
                self._load_session(user_id)
            
            history = self.sessions[user_id]
            
            u_parts = self._prepare_user_parts(user_text, image_path)
            history.append({"role": "user", "parts": u_parts})
            history.append({"role": "model", "parts": [ai_text]})
            
            self._handle_history_limit(user_id, history, on_limit_reached)
            self._update_interaction_time(user_id)
            self._save_session_to_disk(user_id)
    
    def _prepare_user_parts(self, user_text: str, image_path: Optional[str]) -> List[str]:
        u_parts = [user_text]
        if image_path and os.path.exists(image_path):
            u_parts.append(f"{self.IMG_PREFIX}{image_path}")
        return u_parts
    
    def _handle_history_limit(self, user_id: str, history: deque, callback):
        if len(history) > self.MAX_HISTORY:
            if callback:
                try:
                    callback(user_id)
                except Exception as e:
                    logger.error(f"Callback error: {e}")
            
            while len(history) > self.MAX_HISTORY:
                history.popleft()
    
    def _update_interaction_time(self, user_id: str):
        if user_id not in self.meta_data:
            self.meta_data[user_id] = {}
        self.meta_data[user_id]['last_interaction'] = datetime.datetime.now()

    def delete_last_n_messages(self, user_id: str, n: int = 5):
        with self.get_lock(user_id):
            history = self.sessions.get(user_id)
            if history:
                limit = max(0, len(history) - n)
                self.sessions[user_id] = deque(list(history)[:limit])
                self._save_session_to_disk(user_id)

    def clear_session(self, user_id: str):
        with self.get_lock(user_id):
            self._init_empty_session(user_id)
            path = self._get_session_path(user_id)
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError as e:
                    logger.error(f"Clear error for {user_id}: {e}")

    def _get_session_path(self, user_id: str) -> str:
        safe_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        return os.path.join(self.SESSION_DIR, f"{safe_id}.json")
    
    def _load_session(self, user_id: str):
        path = self._get_session_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.sessions[user_id] = deque(data.get("history", []))
                self.meta_data[user_id] = self._parse_metadata(data)
                return

            except (json.JSONDecodeError, OSError) as e:
                logger.error(f"Load error for {user_id}: {e}")
        
        self._init_empty_session(user_id)
    
    def _parse_metadata(self, data: Dict) -> Dict[str, Any]:
        meta = {
            "summary": data.get("summary", ""),
            "memory_summary": data.get("memory_summary", ""),
            "schedule_summary": data.get("schedule_summary", ""),
            "memory_count": data.get("memory_count", 0),
            "last_memory_analysis": data.get("last_memory_analysis", 0),
            "last_interaction": None
        }
        
        ts = data.get("last_interaction_ts")
        if ts:
            try:
                meta["last_interaction"] = datetime.datetime.fromisoformat(ts)
            except ValueError:
                pass
        
        return meta

    def _init_empty_session(self, user_id: str):
        self.sessions[user_id] = deque()
        self.meta_data[user_id] = {
            "summary": "",
            "memory_summary": "",
            "schedule_summary": "",
            "memory_count": 0,
            "last_memory_analysis": 0,
            "last_interaction": None
        }

    def _save_session_to_disk_sync(self, user_id: str):
        """Synchronous version of disk save - called via asyncio.to_thread."""
        try:
            meta = self.meta_data.get(user_id, {})
            last_int = meta.get("last_interaction")
            
            data = {
                "summary": meta.get("summary", ""),
                "memory_summary": meta.get("memory_summary", ""),
                "schedule_summary": meta.get("schedule_summary", ""),
                "memory_count": meta.get("memory_count", 0),
                "last_memory_analysis": meta.get("last_memory_analysis", 0),
                "last_interaction_ts": last_int.isoformat() if last_int else None,
                "history": list(self.sessions.get(user_id, []))
            }
            
            self._atomic_write(self._get_session_path(user_id), data)
        
        except Exception as e:
            logger.error(f"Save error for {user_id}: {e}")

    # Alias for backward compatibility
    def _save_session_to_disk(self, user_id: str):
        """Save session to disk (sync). Alias for _save_session_to_disk_sync."""
        self._save_session_to_disk_sync(user_id)
    
    def _atomic_write(self, path: str, data: Dict):
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass
            raise

    def prepare_history_for_gemini(self, history: deque, img_prefix: str) -> List[types.Content]:
        gemini_history = []
        for msg in history:
            parts = self._process_message_parts(msg.get("parts", []), img_prefix)
            if parts:
                gemini_history.append(types.Content(role=msg.get("role", "user"), parts=parts))
        return gemini_history
    
    def _process_message_parts(self, parts: List, img_prefix: str) -> List:
        processed_parts = []
        for item in parts:
            if isinstance(item, str):
                if item.startswith(img_prefix):
                    img_part = self._load_image_part(item, img_prefix)
                    if img_part:
                        processed_parts.append(img_part)
                else:
                    processed_parts.append(types.Part(text=item))
            else:
                processed_parts.append(item)
        return processed_parts
    
    def _load_image_part(self, item: str, img_prefix: str) -> Optional[PIL.Image.Image]:
        path = item.replace(img_prefix, "")
        if os.path.exists(path):
            try:
                img = PIL.Image.open(path)
                img.load()
                return img
            except Exception as e:
                logger.warning(f"Image load failed, skipping {path}: {e}")
        return None
    
    def get_summary(self, uid): 
        return self.get_metadata(uid, "summary", "")
    
    def get_memory_summary(self, uid): 
        return self.get_metadata(uid, "memory_summary", "")
    
    def get_schedule_summary(self, uid): 
        return self.get_metadata(uid, "schedule_summary", "")

    def should_run_memory_analysis(self, uid, current_count):
        last_time = self.get_metadata(uid, "last_memory_analysis", 0)
        last_count = self.get_metadata(uid, "memory_count", 0)
        
        time_elapsed = time.time() - last_time > self.MEMORY_ANALYSIS_INTERVAL
        count_threshold = current_count - last_count >= self.NEW_MEMORY_THRESHOLD
        
        return time_elapsed or count_threshold
    
    def mark_memory_analysis_done(self, uid, count):
        self.update_metadata(uid, {
            "last_memory_analysis": time.time(),
            "memory_count": count
        })
        self._save_session_to_disk(uid)
    
    def update_summary(self, uid, summary):
        self.update_metadata(uid, {"summary": summary})
        self._save_session_to_disk(uid)
    
    def update_memory_summary(self, uid, summary):
        self.update_metadata(uid, {"memory_summary": summary})
        self._save_session_to_disk(uid)
        
    def update_schedule_summary(self, uid, summary):
        self.update_metadata(uid, {"schedule_summary": summary})
        self._save_session_to_disk(uid)