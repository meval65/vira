import json
import os
import logging
import threading
import time
import datetime
from collections import deque
from typing import List, Optional

import PIL.Image
from google.genai import types

logger = logging.getLogger(__name__)

class SessionManager:
    def __init__(self, max_history: int = 20):
        self.sessions = {}
        self.summaries = {}
        self.memory_summary = {}
        self.memory_counts = {}
        self.last_memory_analysis = {}
        self.last_interactions = {}
        self.locks = {}
        self.MAX_HISTORY = max_history
        self.SESSION_DIR = "storage/sessions"
        self.IMG_PREFIX = ":::IMG_PATH:::"
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = time.time()
        self.CLEANUP_INTERVAL = 1800
        self.MEMORY_ANALYSIS_INTERVAL = 600
        self.NEW_MEMORY_THRESHOLD = 5
        self.SESSION_TIMEOUT = 7200
        os.makedirs(self.SESSION_DIR, exist_ok=True)
    
    def get_lock(self, user_id: str) -> threading.Lock:
        if user_id not in self.locks:
            self.locks[user_id] = threading.Lock()
        return self.locks[user_id]
    
    def cleanup_inactive_sessions(self):
        now = time.time()
        if now - self._last_cleanup < self.CLEANUP_INTERVAL:
            return
        
        if not self._cleanup_lock.acquire(blocking=False):
            return
        
        try:
            current_time = datetime.datetime.now()
            inactive_users = []
            
            for uid, last_time in list(self.last_interactions.items()):
                if last_time and (current_time - last_time).total_seconds() > self.SESSION_TIMEOUT:
                    inactive_users.append(uid)
            
            for user_id in inactive_users:
                self._save_and_unload(user_id)
            
            self._last_cleanup = now
            
            if inactive_users:
                logger.info(f"[SESSION-CLEANUP] Unloaded {len(inactive_users)} inactive sessions")
        
        except Exception as e:
            logger.error(f"[SESSION-CLEANUP-ERROR] {e}")
        
        finally:
            self._cleanup_lock.release()
    
    def _save_and_unload(self, user_id: str):
        with self.get_lock(user_id):
            if user_id in self.sessions:
                self._save_session(user_id)
                
                self.sessions.pop(user_id, None)
                self.summaries.pop(user_id, None)
                self.memory_summary.pop(user_id, None)
                self.memory_counts.pop(user_id, None)
                self.last_memory_analysis.pop(user_id, None)
                self.last_interactions.pop(user_id, None)
                self.locks.pop(user_id, None)
    
    def delete_last_n_messages(self, user_id: str, n: int = 5):
        with self.get_lock(user_id):
            history = self.sessions.get(user_id)
            if not history or len(history) == 0:
                return
            
            n = min(n, len(history))
            new_history = deque(list(history)[:-n]) if n < len(history) else deque()
            
            self.sessions[user_id] = new_history
            self.last_interactions[user_id] = datetime.datetime.now()
            self._save_session(user_id)
    
    def get_session(self, user_id: str) -> deque:
        if user_id not in self.sessions:
            self._load_session(user_id)
        return self.sessions.get(user_id, deque())
    
    def get_memory_summary(self, user_id: str) -> str:
        if user_id not in self.memory_summary:
            self._load_session(user_id)
        return self.memory_summary.get(user_id, "")
    
    def get_summary(self, user_id: str) -> str:
        if user_id not in self.summaries:
            self._load_session(user_id)
        return self.summaries.get(user_id, "")
    
    def should_run_memory_analysis(self, user_id: str, current_count: int) -> bool:
        if not self.memory_summary.get(user_id):
            return True
        
        last_analysis_time = self.last_memory_analysis.get(user_id, 0)
        time_elapsed = time.time() - last_analysis_time
        
        if time_elapsed > self.MEMORY_ANALYSIS_INTERVAL:
            return True
        
        last_count = self.memory_counts.get(user_id, 0)
        new_memories = current_count - last_count
        
        if new_memories >= self.NEW_MEMORY_THRESHOLD:
            return True
        
        return False
    
    def mark_memory_analysis_done(self, user_id: str, memory_count: int):
        self.last_memory_analysis[user_id] = time.time()
        self.memory_counts[user_id] = memory_count
        self._save_session(user_id)
    
    def update_session(self, user_id: str, user_text: str, ai_text: str, 
                      image_path: str = None, on_limit_reached=None):
        with self.get_lock(user_id):
            if user_id not in self.sessions:
                self.sessions[user_id] = deque()
            
            history = self.sessions[user_id]
            
            u_parts = [user_text]
            if image_path and os.path.exists(image_path):
                u_parts.append(f"{self.IMG_PREFIX}{image_path}")
            
            history.append({"role": "user", "parts": u_parts})
            history.append({"role": "model", "parts": [ai_text]})
            
            if len(history) >= self.MAX_HISTORY and on_limit_reached:
                try:
                    on_limit_reached(user_id)
                except Exception as e:
                    logger.error(f"[LIMIT-CALLBACK-ERROR] {e}")
            
            while len(history) > self.MAX_HISTORY:
                history.popleft()
            
            self.last_interactions[user_id] = datetime.datetime.now()
            self._save_session(user_id)
    
    def update_memory_summary(self, user_id: str, summary: str):
        with self.get_lock(user_id):
            self.memory_summary[user_id] = summary
            self._save_session(user_id)
    
    def update_summary(self, user_id: str, summary: str):
        with self.get_lock(user_id):
            self.summaries[user_id] = summary
            self._save_session(user_id)
    
    def clear_session(self, user_id: str):
        with self.get_lock(user_id):
            self._init_empty_session(user_id)
            
            path = self._get_session_path(user_id)
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"[SESSION-CLEAR] User: {user_id}")
                except Exception as e:
                    logger.error(f"[SESSION-CLEAR-ERROR] {e}")
    
    def _get_session_path(self, user_id: str) -> str:
        safe_user_id = "".join(c for c in user_id if c.isalnum() or c in ('-', '_'))
        return os.path.join(self.SESSION_DIR, f"{safe_user_id}.json")
    
    def _load_session(self, user_id: str):
        path = self._get_session_path(user_id)
        
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.memory_summary[user_id] = data.get("memory_summary", "")
                self.summaries[user_id] = data.get("summary", "")
                self.sessions[user_id] = deque(data.get("history", []))
                self.memory_counts[user_id] = data.get("memory_count", 0)
                self.last_memory_analysis[user_id] = data.get("last_memory_analysis", 0)
                
                last_ts = data.get("last_interaction_ts")
                if last_ts:
                    try:
                        self.last_interactions[user_id] = datetime.datetime.fromisoformat(last_ts)
                    except ValueError:
                        self.last_interactions[user_id] = None
                else:
                    self.last_interactions[user_id] = None
            
            except Exception as e:
                logger.error(f"[SESSION-LOAD-ERROR] {user_id}: {e}")
                self._init_empty_session(user_id)
        else:
            self._init_empty_session(user_id)
    
    def _init_empty_session(self, user_id: str):
        self.sessions[user_id] = deque()
        self.summaries[user_id] = ""
        self.memory_summary[user_id] = ""
        self.memory_counts[user_id] = 0
        self.last_memory_analysis[user_id] = 0
        self.last_interactions[user_id] = None
    
    def _save_session(self, user_id: str):
        try:
            last_int = self.last_interactions.get(user_id)
            
            data = {
                "summary": self.summaries.get(user_id, ""),
                "memory_summary": self.memory_summary.get(user_id, ""),
                "history": list(self.sessions.get(user_id, [])),
                "memory_count": self.memory_counts.get(user_id, 0),
                "last_memory_analysis": self.last_memory_analysis.get(user_id, 0),
                "last_interaction_ts": last_int.isoformat() if last_int else None
            }
            
            path = self._get_session_path(user_id)
            tmp_path = path + '.tmp'
            
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            os.replace(tmp_path, path)
        
        except Exception as e:
            logger.error(f"[SESSION-SAVE-ERROR] {user_id}: {e}")
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except:
                    pass

    def prepare_history_for_gemini(self, history: deque, img_prefix: str) -> List[types.Content]:
        gemini_history = []
        
        for msg in history:
            parts = []
            
            for item in msg.get("parts", []):
                if isinstance(item, str):
                    if item.startswith(img_prefix):
                        path = item.replace(img_prefix, "")
                        if os.path.exists(path):
                            try:
                                parts.append(PIL.Image.open(path))
                            except Exception as e:
                                logger.error(f"[IMG-LOAD-ERROR] {path}: {e}")
                    else:
                        parts.append(types.Part(text=item))
                else:
                    parts.append(item)
            
            if parts:
                role = msg.get("role", "user")
                gemini_history.append(types.Content(role=role, parts=parts))
        
        return gemini_history