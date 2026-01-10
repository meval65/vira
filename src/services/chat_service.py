import datetime
import json
import os
import logging
import asyncio
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional

import PIL.Image
from dateutil import parser
from google import genai
from google.genai import types
from pymeteosource.api import Meteosource
from pymeteosource.types import tiers

from src.config import (
    GOOGLE_API_KEYS,
    CHAT_MODEL,
    TIER_1_MODEL,
    TIER_2_MODEL,
    TIER_3_MODEL,
    INSTRUCTION,
    FIRST_LEVEL_ANALYSIS_INSTRUCTION,
    SECOND_LEVEL_ANALYSIS_INSTRUCTION,
    MEMORY_ANALYSIS_INSTRUCTION,
    CHAT_ANALYSIS_INSTUCTION,
    GET_RELEVANT_CONTEXT_INSTRUCTION,
    AVAILABLE_CHAT_MODELS,
    METEOSOURCE_API_KEY,
    set_chat_model
)

from src.services.memory_service import MemoryManager
from src.services.analyzer_service import MemoryAnalyzer
from src.services.scheduler_service import SchedulerService

logger = logging.getLogger(__name__)

class APIKeyHealthMonitor:
    def __init__(self, num_keys: int):
        self.health = {i: {'healthy': True, 'failures': 0, 'last_fail': None} for i in range(num_keys)}
        self.lock = threading.Lock()
        self.FAILURE_THRESHOLD = 3
        self.RECOVERY_TIME = 600
    
    def mark_failure(self, key_index: int):
        with self.lock:
            self.health[key_index]['failures'] += 1
            self.health[key_index]['last_fail'] = time.time()
            if self.health[key_index]['failures'] >= self.FAILURE_THRESHOLD:
                self.health[key_index]['healthy'] = False
                logger.warning(f"[API-HEALTH] Key {key_index} marked unhealthy")
    
    def mark_success(self, key_index: int):
        with self.lock:
            if self.health[key_index]['failures'] > 0:
                self.health[key_index]['failures'] = 0
                self.health[key_index]['healthy'] = True
    
    def get_healthy_key(self, current_index: int, total_keys: int) -> Optional[int]:
        with self.lock:
            now = time.time()
            for offset in range(1, total_keys + 1):
                candidate = (current_index + offset) % total_keys
                key_health = self.health[candidate]
                if key_health['healthy']:
                    return candidate
                if key_health['last_fail'] and (now - key_health['last_fail']) > self.RECOVERY_TIME:
                    key_health['healthy'] = True
                    key_health['failures'] = 0
                    logger.info(f"[API-HEALTH] Key {candidate} recovered")
                    return candidate
            return None

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
        os.makedirs(self.SESSION_DIR, exist_ok=True)
    
    def get_lock(self, user_id: str) -> threading.Lock:
        return self.locks.setdefault(user_id, threading.Lock())
    
    def cleanup_inactive_sessions(self):
        now = time.time()
        if now - self._last_cleanup < self.CLEANUP_INTERVAL:
            return
        if not self._cleanup_lock.acquire(blocking=False):
            return
        try:
            timeout = 7200
            inactive_users = [
                uid for uid, last_time in self.last_interactions.items()
                if last_time and (datetime.datetime.now() - last_time).total_seconds() > timeout
            ]
            for user_id in inactive_users:
                self._save_and_unload(user_id)
            self._last_cleanup = now
            if inactive_users:
                logger.info(f"[SESSION-CLEANUP] Unloaded {len(inactive_users)} sessions")
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
    
    def delete_last_n_messages(self, user_id: str, n: int = 5):
        history = self.get_session(user_id)
        if not history:
            return
        with self.get_lock(user_id):
            new_history_list = list(history)[:-n]
            self.sessions[user_id] = deque(new_history_list)
            self.last_interactions[user_id] = datetime.datetime.now()
            self._save_session(user_id)
    
    def get_session(self, user_id: str) -> deque:
        if user_id not in self.sessions:
            self._load_session(user_id)
        return self.sessions[user_id]
    
    def get_memory_summary(self, user_id: str) -> str:
        if user_id not in self.memory_summary:
            self._load_session(user_id)
        return self.memory_summary.get(user_id, "")
    
    def get_summary(self, user_id: str) -> str:
        if user_id not in self.summaries:
            self._load_session(user_id)
        return self.summaries.get(user_id, "User baru.")
    
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
    
    def update_session(self, user_id: str, user_text: str, ai_text: str, image_path: str = None, on_limit_reached=None):
        if user_id not in self.sessions:
            self.sessions[user_id] = deque()
        history = self.sessions[user_id]
        u_parts = [user_text]
        if image_path and os.path.exists(image_path):
            u_parts.append(f"{self.IMG_PREFIX}{image_path}")
        history.append({"role": "user", "parts": u_parts})
        history.append({"role": "model", "parts": [ai_text]})
        
        if len(history) >= self.MAX_HISTORY and on_limit_reached:
            on_limit_reached(user_id)
        
        while len(history) > self.MAX_HISTORY:
            history.popleft()
        
        self.last_interactions[user_id] = datetime.datetime.now()
        self._save_session(user_id)
    
    def update_memory_summary(self, user_id: str, summary: str):
        self.memory_summary[user_id] = summary
        self._save_session(user_id)
    
    def update_summary(self, user_id: str, summary: str):
        self.summaries[user_id] = summary
        self._save_session(user_id)
    
    def clear_session(self, user_id: str):
        self._init_empty_session(user_id)
        path = self._get_session_path(user_id)
        if os.path.exists(path):
            try:
                os.remove(path)
                logger.info(f"[SESSION-CLEAR] {user_id}")
            except Exception as e:
                logger.error(f"[SESSION-CLEAR-ERROR] {e}")
    
    def _get_session_path(self, user_id: str) -> str:
        return os.path.join(self.SESSION_DIR, f"{user_id}.json")
    
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
                self.last_interactions[user_id] = datetime.datetime.fromisoformat(last_ts) if last_ts else None
            except Exception as e:
                logger.error(f"[SESSION-LOAD-ERROR] {e}")
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
            logger.error(f"[SESSION-SAVE-ERROR] {e}")

    def prepare_history_for_gemini(self, history: deque, img_prefix: str) -> List[types.Content]:
        gemini_history = []
        for msg in history:
            parts = []
            for item in msg["parts"]:
                if isinstance(item, str):
                    if item.startswith(img_prefix):
                        path = item.replace(img_prefix, "")
                        if os.path.exists(path):
                            try:
                                parts.append(PIL.Image.open(path))
                            except Exception:
                                pass
                    else:
                        parts.append(types.Part(text=item))
                else:
                    parts.append(item)
            if parts:
                gemini_history.append(types.Content(role=msg["role"], parts=parts))
        return gemini_history

class ChatHandler:
    def __init__(self, memory_manager: MemoryManager, analyzer: MemoryAnalyzer, scheduler_service: SchedulerService):
        self.memory_manager = memory_manager
        self.analyzer = analyzer
        self.scheduler_service = scheduler_service
        self.api_keys = GOOGLE_API_KEYS
        self.current_key_index = 0
        self.health_monitor = APIKeyHealthMonitor(len(self.api_keys))
        self.client: Optional[genai.Client] = None
        self._configure_genai()
        self.chat_model_name = CHAT_MODEL
        self.tier_1_model_name = TIER_1_MODEL
        self.tier_2_model_name = TIER_2_MODEL
        self.tier_3_model_name = TIER_3_MODEL
        self.session_manager = SessionManager()
        self.context_cache: Dict[str, Tuple[str, datetime.datetime]] = {}
        self.DEFAULT_LAT = -7.6398581
        self.DEFAULT_LON = 112.2395766
        self._cached_weather = None
        self._last_weather_fetch = None
        self._weather_cache_duration = 900
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._session_processing_flags: Dict[str, bool] = {}
        self._flag_lock = threading.Lock()

    def _configure_genai(self):
        if not self.api_keys:
            raise ValueError("No API keys")
        try:
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            logger.info(f"[SYSTEM] Client initialized with Key-{self.current_key_index}")
        except Exception as e:
            logger.error(f"[INIT-ERROR] {e}")
            raise

    def _rotate_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            return False
        next_key = self.health_monitor.get_healthy_key(self.current_key_index, len(self.api_keys))
        if next_key is None:
            logger.error("[SYSTEM] No healthy API keys")
            return False
        self.current_key_index = next_key
        try:
            self._configure_genai()
            logger.warning(f"[SYSTEM] Rotated to Key-{self.current_key_index}")
            return True
        except Exception as e:
            logger.error(f"[ROTATION-ERROR] {e}")
            return False

    def change_model(self, model_name: str):
        if model_name not in AVAILABLE_CHAT_MODELS:
            raise ValueError(f"Unavailable model: {model_name}")
        self.chat_model_name = model_name
        set_chat_model(model_name)
        logger.info(f"[SYSTEM] Model changed to: {model_name}")

    def _format_time_gap(self, last_time: Optional[datetime.datetime]) -> str:
        if not last_time:
            return "Ini interaksi pertama."
        diff = datetime.datetime.now() - last_time
        seconds = diff.total_seconds()
        if seconds < 60:
            return "Baru saja."
        elif seconds < 3600:
            return f"{int(seconds/60)} menit lalu."
        elif seconds < 86400:
            return f"{int(seconds/3600)} jam lalu."
        elif diff.days == 1:
            return "Kemarin."
        elif diff.days < 7:
            return f"{diff.days} hari lalu."
        return f"{diff.days // 7} minggu lalu."

    def _get_weather_data(self) -> str:
        if self._cached_weather and self._last_weather_fetch:
            if (datetime.datetime.now() - self._last_weather_fetch).total_seconds() < self._weather_cache_duration:
                return self._cached_weather
        if not METEOSOURCE_API_KEY:
            return "Data cuaca tidak dikonfigurasi."
        try:
            ms = Meteosource(METEOSOURCE_API_KEY, tiers.FREE)
            forecast = ms.get_point_forecast(lat=self.DEFAULT_LAT, lon=self.DEFAULT_LON, sections=['current'])
            if forecast and hasattr(forecast, 'current'):
                curr = forecast.current
                w_str = f"{curr.summary}, {curr.temperature}°C"
                if hasattr(curr, 'feels_like'):
                    w_str += f" (terasa {curr.feels_like}°C)"
                self._cached_weather = w_str
                self._last_weather_fetch = datetime.datetime.now()
                return w_str
        except Exception as e:
            logger.error(f"[WEATHER] {e}")
            if self._cached_weather:
                return f"{self._cached_weather} (Cached)"
        return "Cuaca tidak tersedia."

    def _run_chat_analysis(self, user_id: str) -> str:
        history = list(self.session_manager.get_session(user_id))[-10:]
        prompt = (
            f"{CHAT_ANALYSIS_INSTUCTION}\n\n"
            f"[OLD SUMMARY]: {self.session_manager.get_summary(user_id)}\n\n"
            f"[HISTORY CHAT]: {history}\n\n"
        )
        try:
            mem_res = self.client.models.generate_content(
                model=self.tier_1_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=512)
            )
            response = mem_res.text if mem_res.text else ""
            logger.info(f"[MODEL-OUTPUT: {self.tier_1_model_name}] (Chat Analysis): {response}")
            self.session_manager.update_summary(user_id, response)
            self.session_manager.delete_last_n_messages(user_id, 10)
            return response
        except Exception as e:
            logger.error(f"[CHAT-ANALYSIS-ERROR] {e}")
            return ""

    def _run_memory_analysis(self, user_id: str, data_number: int) -> str:
        result = self.memory_manager.search_memories(user_id, "", data_number)
        if not result:
            return ""
        
        current_count = len(self.memory_manager.search_memories(user_id, "", 1000))
        if not self.session_manager.should_run_memory_analysis(user_id, current_count):
            logger.info(f"[MEMORY-ANALYSIS-SKIP] {user_id}: conditions not met")
            return self.session_manager.get_memory_summary(user_id)
        
        mem_lines = [f"{i+1}:{m.get('summary', '')})" for i, m in enumerate(result)]
        prompt = (
            f"{MEMORY_ANALYSIS_INSTRUCTION}\n\n"
            f"[Old_Memory_Summary]:\n{self.session_manager.get_memory_summary(user_id)}\n\n"
            f"[New_Memory_Data]:\n{chr(10).join(mem_lines)}\n"
        )
        try:
            mem_res = self.client.models.generate_content(
                model=self.tier_1_model_name,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=512)
            )
            response = mem_res.text if mem_res.text else ""
            logger.info(f"[MODEL-OUTPUT: {self.tier_1_model_name}] (Memory Analysis): {response}")
            self.session_manager.update_memory_summary(user_id, response)
            self.session_manager.mark_memory_analysis_done(user_id, current_count)
            return response
        except Exception as e:
            logger.error(f"[MEMORY-ANALYSIS-ERROR] {e}")
            return ""

    def _build_context(self, user_id: str, summary: str, memories: List[str], user_text: str, schedule_context: str = None) -> str:
        cache_key = f"{user_id}:{hashlib.md5(summary.encode()).hexdigest()[:8]}"
        if cache_key in self.context_cache and not schedule_context:
            cached_ctx, cached_time = self.context_cache[cache_key]
            if (datetime.datetime.now() - cached_time).total_seconds() < 300:
                return cached_ctx
        
        now_str = datetime.datetime.now().strftime('%A, %d %B %Y, %H:%M')
        mem_str = "\n".join([f"• {m}" for m in memories[:10]]) if memories else "Tidak ada data spesifik."
        memory_summary = self.session_manager.get_memory_summary(user_id)
        
        schedule_alert = f"{schedule_context}\n" if schedule_context else ""
        context = (
            f"{GET_RELEVANT_CONTEXT_INSTRUCTION}\n\n"
            f"[USER INPUT]: {user_text}\n\n"
            f"TIME: {now_str}\n"
            f"WEATHER: {self._get_weather_data()}\n"
            f"STATUS: {self._format_time_gap(self.session_manager.last_interactions.get(user_id))}\n"
            f"[Schedule / Agenda]: {schedule_alert}\n"
            f"[Conversation Summary]: {summary}\n\n"
            f"[Long-term Memory Summary]: {memory_summary}\n\n"
            f"[Relevant Memories]: {mem_str}\n\n"
        )
        if not schedule_context:
            self.context_cache[cache_key] = (context, datetime.datetime.now())
            if len(self.context_cache) > 50:
                oldest = min(self.context_cache.keys(), key=lambda k: self.context_cache[k][1])
                del self.context_cache[oldest]
        
        relevant = self.client.models.generate_content(
            model=self.tier_2_model_name,
            contents=context,
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=512)
        )
        return relevant.text

    def process_message(self, user_id: str, user_text: str, image_path: str = None) -> str:
        with self._flag_lock:
            if self._session_processing_flags.get(user_id, False):
                return "Mohon tunggu, pesan sebelumnya masih diproses..."
            self._session_processing_flags[user_id] = True
        try:
            self.session_manager.cleanup_inactive_sessions()
            return self._process_message_internal(user_id, user_text, image_path)
        finally:
            with self._flag_lock:
                self._session_processing_flags[user_id] = False

    def _process_message_internal(self, user_id: str, user_text: str, image_path: str = None) -> str:
        history = self.session_manager.get_session(user_id)
        current_summary = self.session_manager.get_summary(user_id)
        pending_sched = self.scheduler_service.get_pending_schedule_for_user(user_id)
        schedule_ctx = None
        if pending_sched:
            schedule_ctx = pending_sched['context']
            self.scheduler_service.mark_as_executed(pending_sched['id'])
        
        relevant_memories = []
        if user_text and len(user_text.strip()) > 3:
            try:
                emb = self.analyzer.get_embedding(user_text)
                if emb:
                    relevant_memories = self.memory_manager.get_relevant_memories(user_id, emb)
            except Exception as e:
                logger.error(f"[EMBEDDING-ERROR] {e}")
        
        system_context = self._build_context(user_id, current_summary, relevant_memories, user_text, schedule_ctx)
        full_prompt = f"{system_context}\n\n[PESAN BARU]\nUser: {user_text}"
        chat_input = [types.Part(text=full_prompt)]
        if image_path and os.path.exists(image_path):
            try:
                chat_input.append(PIL.Image.open(image_path))
            except Exception as e:
                logger.error(f"[IMG-ERROR] {e}")
        
        gemini_history = self.session_manager.prepare_history_for_gemini(history, self.session_manager.IMG_PREFIX)
        response = self._generate_with_retry(gemini_history, chat_input)
        
        if response and response != "ERROR":
            self.session_manager.update_session(user_id, user_text, response, image_path, lambda uid: self.executor.submit(self._run_chat_analysis, uid))
            hist_snapshot = list(self.session_manager.get_session(user_id))
            self.executor.submit(self._run_advanced_analysis, user_id, user_text, response, hist_snapshot)
        return response

    def _generate_with_retry(self, history: List[types.Content], chat_input: List, max_retries: int = 3) -> str:
        config = types.GenerateContentConfig(
            temperature=0.55, top_p=0.85, top_k=40, max_output_tokens=2048, system_instruction=INSTRUCTION
        )
        for attempt in range(max_retries + 1):
            try:
                chat = self.client.chats.create(model=self.chat_model_name, history=history, config=config)
                resp = chat.send_message(chat_input)
                if resp.text:
                    self.health_monitor.mark_success(self.current_key_index)
                    logger.info(f"[MODEL-OUTPUT: {self.chat_model_name}] (Main Chat): {resp.text}")
                    return resp.text
                return ""
            except Exception as e:
                err = str(e).lower()
                logger.error(f"[CHAT-FAIL] Attempt {attempt+1}: {e}")
                if any(x in err for x in ["quota", "429", "exhausted"]):
                    self.health_monitor.mark_failure(self.current_key_index)
                    if self._rotate_api_key():
                        continue
                    return "Maaf, sistem sibuk."
                if attempt < max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
        return "Maaf, terjadi kesalahan sistem."

    def _run_advanced_analysis(self, user_id: str, user_text: str, vira_text: str, history_snapshot: List[Any]):
        if not user_text or not vira_text:
            return
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prompt_l1 = (
            f"{FIRST_LEVEL_ANALYSIS_INSTRUCTION}\n\n"
            f"[SYSTEM TIME]: {now_str}\n"
            f"[INTERACTION]\nUser: {user_text}\nAI: {vira_text}\n"
        )
        try:
            gemma_res = self.client.chats.create(
                model=self.tier_2_model_name,
                config=types.GenerateContentConfig(temperature=0.5, max_output_tokens=512)
            )
            gemma_out = gemma_res.send_message(prompt_l1)
            logger.info(f"[MODEL-OUTPUT: {self.tier_1_model_name}] (Analysis L1): {gemma_out.text}")
            
            prompt_l2 = f"{SECOND_LEVEL_ANALYSIS_INSTRUCTION}\n\n[SOURCE]\n{gemma_out.text}\n"
            flash_res = self.client.models.generate_content(
                model=self.tier_3_model_name, contents=prompt_l2,
                config=types.GenerateContentConfig(temperature=0.1)
            )
            logger.info(f"[MODEL-OUTPUT: {self.tier_3_model_name}] (Analysis L2): {flash_res.text}")
            
            data = self._safe_json_parse(flash_res.text)
            if data:
                self._process_analysis_results(user_id, data)
        except Exception as e:
            logger.error(f"[ADV-ANALYSIS-ERROR] {e}")

    def _safe_json_parse(self, text: str) -> Optional[Dict]:
        try:
            clean = text.strip()
            if clean.startswith("```"):
                lines = clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean = "\n".join(lines)
            return json.loads(clean.strip())
        except Exception as e:
            logger.error(f"[JSON-PARSE-ERROR] {e}")
            return None

    def _process_analysis_results(self, user_id: str, data: Dict):
        mem = data.get("memory")
        if mem:
            items = mem if isinstance(mem, list) else [mem]
            for m in items:
                if isinstance(m, dict) and m.get("should_store"):
                    self._save_memory_to_db(user_id, m)
        sched = data.get("schedules")
        if sched:
            items = sched if isinstance(sched, list) else [sched]
            for s in items:
                if isinstance(s, dict) and s.get("should_schedule"):
                    self._process_new_schedule(user_id, s)

    def _process_new_schedule(self, user_id: str, data: Dict):
        try:
            t_str, ctx = data.get("time_str"), data.get("context")
            if not t_str or not ctx or len(ctx) < 5:
                return
            trigger = parser.parse(t_str, fuzzy=True)
            now = datetime.datetime.now()
            if trigger.year < 2000:
                trigger = trigger.replace(year=now.year, month=now.month, day=now.day)
            if trigger.date() == now.date() and trigger < now:
                trigger += datetime.timedelta(days=1)
            if trigger < now:
                return
            self.scheduler_service.add_schedule(user_id, trigger, ctx)
            logger.info(f"[SCHEDULE-ADDED] {user_id} @ {trigger}")
        except Exception as e:
            logger.error(f"[SCHEDULE-ERROR] {e}")

    def _save_memory_to_db(self, user_id: str, data: Dict):
        summary = data.get("summary", "").strip()
        if not summary or len(summary) < 5:
            return
        try:
            vec = self.analyzer.get_embedding(summary)
            if vec:
                self.memory_manager.add_memory(
                    user_id, summary, data.get("type", "preference"), data.get("priority", 0.5), vec
                )
                logger.info(f"[MEMORY-SAVED] {user_id}: {summary}...")
                self.executor.submit(self._run_memory_analysis, user_id, 40)
        except Exception as e:
            logger.error(f"[MEMORY-SAVE-FAIL] {e}")

    def trigger_proactive_message(self, user_id: str, context: str) -> Optional[str]:
        try:
            history = self.session_manager.prepare_history_for_gemini(
                self.session_manager.get_session(user_id), self.session_manager.IMG_PREFIX
            )
            prompt = (
                f"[SYSTEM TRIGGER] Jadwal pengingat.\nKonteks: {context}\n"
                f"Summary: {self.session_manager.get_summary(user_id)}\n"
                f"Tugas: Sapa dan ingatkan user."
            )
            chat = self.client.chats.create(
                model=self.chat_model_name, history=history,
                config=types.GenerateContentConfig(temperature=0.6, system_instruction=INSTRUCTION)
            )
            resp = chat.send_message(prompt)
            if resp.text:
                logger.info(f"[MODEL-OUTPUT: {self.chat_model_name}] (Proactive): {resp.text}")
                self.session_manager.update_session(user_id, "(System Reminder)", resp.text, None)
                return resp.text
        except Exception as e:
            logger.error(f"[PROACTIVE-FAIL] {e}")
        return None

    def get_system_stats(self) -> Dict:
        return {
            "active_sessions": len(self.session_manager.sessions),
            "current_key": self.current_key_index,
            "api_health": {k: v['healthy'] for k, v in self.health_monitor.health.items()},
            "cache_size": len(self.context_cache),
            "models": {
                "chat": self.chat_model_name, 
                "L1": self.tier_1_model_name, 
                "L2": self.tier_2_model_name,
                "L3": self.tier_3_model_name
            }
        }

    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)