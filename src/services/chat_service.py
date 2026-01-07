import datetime
from datetime import timedelta
import json
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
import PIL.Image 
from typing import List, Dict, Any, Tuple, Optional
from collections import deque
from dateutil import parser
from functools import lru_cache
import hashlib

from pymeteosource.api import Meteosource
from pymeteosource.types import tiers

from src.config import (
    GOOGLE_API_KEYS, 
    CHAT_MODEL, 
    ANALYSIS_MODEL,
    FIRST_LEVEL_ANALYZER_MODEL,
    INSTRUCTION, 
    ANALYSIS_PROMPT, 
    FIRST_LEVEL_ANALYSIS_INSTRUCTION,
    AVAILABLE_CHAT_MODELS,
    METEOSOURCE_API_KEY, 
    set_chat_model
)

from src.services.memory_service import MemoryManager
from src.services.analyzer_service import MemoryAnalyzer
from src.services.scheduler_service import SchedulerService

logger = logging.getLogger(__name__)

class ChatHandler:
    def __init__(self, memory_manager: MemoryManager, analyzer: MemoryAnalyzer, scheduler_service: SchedulerService):
        self.memory_manager = memory_manager
        self.analyzer = analyzer
        self.scheduler_service = scheduler_service
        
        self.api_keys = GOOGLE_API_KEYS
        self.current_key_index = 0
        self.key_health_status = {i: True for i in range(len(self.api_keys))}
        self._configure_genai()
        
        self.chat_model_name = CHAT_MODEL
        self.analysis_model_name = ANALYSIS_MODEL
        self.first_level_model_name = FIRST_LEVEL_ANALYZER_MODEL
        
        self.chat_instance = None
        self.analysis_instance = None
        self.first_level_instance = None
        
        self._init_models()
        
        self.active_sessions: Dict[str, deque] = {}
        self.session_summaries: Dict[str, str] = {}
        self.last_interactions: Dict[str, datetime.datetime] = {}
        self.context_cache: Dict[str, Tuple[str, datetime.datetime]] = {}
        
        self.MAX_HISTORY_LENGTH = 40 
        self.SESSION_DIR = "storage/sessions"
        self.IMG_PREFIX = ":::IMG_PATH:::" 
        
        self.DEFAULT_LAT = -7.6398581
        self.DEFAULT_LON = 112.2395766
        
        self._cached_weather = None      
        self._last_weather_fetch = None
        self._weather_cache_duration = 900
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        
        self._analysis_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else None
        self._session_lock = {}
        
        if not os.path.exists(self.SESSION_DIR):
            os.makedirs(self.SESSION_DIR)

    def _configure_genai(self):
        if not self.api_keys:
            logger.critical("[SYSTEM] No Google API KEY available!")
            raise ValueError("No API keys configured")
        genai.configure(api_key=self.api_keys[self.current_key_index])

    def _rotate_api_key(self) -> bool:
        if len(self.api_keys) <= 1: 
            return False
        
        original_index = self.current_key_index
        attempts = 0
        
        while attempts < len(self.api_keys):
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            
            if self.key_health_status.get(self.current_key_index, True):
                self._configure_genai()
                logger.warning(f"[SYSTEM] Rotated to API key index {self.current_key_index}")
                self._init_models()
                return True
            
            attempts += 1
        
        self.current_key_index = original_index
        logger.error("[SYSTEM] All API keys exhausted or unhealthy")
        return False

    def _mark_key_unhealthy(self, key_index: int):
        self.key_health_status[key_index] = False
        logger.warning(f"[SYSTEM] Marked API key {key_index} as unhealthy")

    def _init_models(self):
        try:
            self.chat_instance = genai.GenerativeModel(
                model_name=self.chat_model_name,
                system_instruction=INSTRUCTION
            )
            
            self.first_level_instance = genai.GenerativeModel(
                model_name=self.first_level_model_name
            )

            self.analysis_instance = genai.GenerativeModel(
                model_name=self.analysis_model_name,
                system_instruction=ANALYSIS_PROMPT
            )
            
            logger.info(f"[INIT] Models loaded - Chat: {self.chat_model_name}, L1: {self.first_level_model_name}, L2: {self.analysis_model_name}")
        except Exception as e:
            logger.error(f"[INIT-ERROR] Failed to load models: {e}")
            raise

    def change_model(self, model_name: str):
        if model_name not in AVAILABLE_CHAT_MODELS:
            raise ValueError(f"Model {model_name} not available. Available: {AVAILABLE_CHAT_MODELS}")
        
        self.chat_model_name = model_name
        set_chat_model(model_name)
        self._init_models()
        logger.info(f"[SYSTEM] Chat model changed to: {model_name}")

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
        else:
            return f"{diff.days // 7} minggu lalu."

    @lru_cache(maxsize=32)
    def _get_cached_weather_key(self, lat: float, lon: float, timestamp: int) -> str:
        return f"{lat}:{lon}:{timestamp}"

    def _get_weather_data(self) -> str:
        if self._cached_weather and self._last_weather_fetch:
            elapsed = (datetime.datetime.now() - self._last_weather_fetch).total_seconds()
            if elapsed < self._weather_cache_duration:
                return self._cached_weather

        if not METEOSOURCE_API_KEY:
            return "Data cuaca tidak dikonfigurasi."

        try:
            ms = Meteosource(METEOSOURCE_API_KEY, tiers.FREE)
            forecast = ms.get_point_forecast(
                lat=self.DEFAULT_LAT, 
                lon=self.DEFAULT_LON, 
                sections=['current'],
            )
            
            if forecast and hasattr(forecast, 'current'):
                weather_str = f"{forecast.current.summary}, {forecast.current.temperature}°C"
                if hasattr(forecast.current, 'feels_like'):
                    weather_str += f" (terasa {forecast.current.feels_like}°C)"
                
                self._cached_weather = weather_str
                self._last_weather_fetch = datetime.datetime.now()
                return self._cached_weather
                
        except Exception as e:
            logger.error(f"[WEATHER] Error: {e}")
            if self._cached_weather: 
                return f"{self._cached_weather} (Cached)"
        
        return "Cuaca tidak tersedia."

    def _build_context(self, user_id: str, summary: str, memories: List[str], schedule_context: str = None) -> str:
        cache_key = f"{user_id}:{hashlib.md5(summary.encode()).hexdigest()[:8]}"
        
        if cache_key in self.context_cache:
            cached_context, cached_time = self.context_cache[cache_key]
            if (datetime.datetime.now() - cached_time).total_seconds() < 300:
                if not schedule_context:
                    return cached_context
        
        now = datetime.datetime.now()
        now_str = now.strftime('%A, %d %B %Y, %H:%M')
        weather = self._get_weather_data()
        last_seen = self.last_interactions.get(user_id)
        gap_str = self._format_time_gap(last_seen)
        
        mem_str = "\n".join([f"• {m}" for m in memories[:10]]) if memories else "Tidak ada data spesifik."
        
        upcoming_schedules = self.scheduler_service.get_upcoming_schedules(user_id, limit=7)
        schedule_list_str = "\n".join(upcoming_schedules) if upcoming_schedules else "Tidak ada jadwal tercatat."

        schedule_alert = ""
        if schedule_context:
            schedule_alert = (
                f"\n[SYSTEM REMINDER ACTIVE]\n"
                f"Sistem mendeteksi jadwal jatuh tempo: '{schedule_context}'.\n"
                f"TUGAS: Ingatkan user secara natural dan empati.\n"
            )

        context = (
            f"[SITUASI SAAT INI]\n"
            f"Waktu: {now_str}\n"
            f"Cuaca: {weather}\n"
            f"Terakhir Chat: {gap_str}\n"
            f"{schedule_alert}\n"
            f"[AGENDA/JADWAL USER MENDATANG]:\n{schedule_list_str}\n\n"
            f"[RINGKASAN MASA LALU]: {summary}\n\n"
            f"[FAKTA PENTING USER]:\n{mem_str}"
        )
        
        if not schedule_context:
            self.context_cache[cache_key] = (context, now)
            if len(self.context_cache) > 50:
                oldest_key = min(self.context_cache.keys(), key=lambda k: self.context_cache[k][1])
                del self.context_cache[oldest_key]
        
        return context

    def process_message(self, user_id: str, user_text: str, image_path: str = None) -> str:
        if user_id not in self._session_lock:
            self._session_lock[user_id] = False
        
        if self._session_lock[user_id]:
            return "Mohon tunggu, pesan sebelumnya masih diproses..."
        
        self._session_lock[user_id] = True
        
        try:
            return self._process_message_internal(user_id, user_text, image_path)
        finally:
            self._session_lock[user_id] = False

    def _process_message_internal(self, user_id: str, user_text: str, image_path: str = None) -> str:
        history = self.get_session_history(user_id)
        current_summary = self.session_summaries.get(user_id, "User baru.")
        
        relevant_memories = []
        if user_text and len(user_text.strip()) > 3:
            try:
                emb = self.analyzer.get_embedding(user_text)
                if emb: 
                    relevant_memories = self.memory_manager.get_relevant_memories(user_id, emb)
            except Exception as e:
                logger.error(f"[EMBEDDING-ERROR] {e}")
        
        pending_sched = self.scheduler_service.get_pending_schedule_for_user(user_id)
        schedule_ctx = None
        if pending_sched:
            schedule_ctx = pending_sched['context']
            self.scheduler_service.mark_as_executed(pending_sched['id'])
            logger.info(f"[COLLISION] Merging schedule {pending_sched['id']} into chat")

        system_context = self._build_context(user_id, current_summary, relevant_memories, schedule_ctx)
        full_prompt = f"{system_context}\n\n[PESAN BARU]\nUser: {user_text}"
        
        chat_input = [full_prompt]
        if image_path and os.path.exists(image_path):
            try: 
                img = PIL.Image.open(image_path)
                chat_input.append(img)
            except Exception as e:
                logger.error(f"[IMAGE-ERROR] {e}")

        gemini_history = self._prepare_history_for_gemini(history)
        
        vira_response = self._generate_with_retry(gemini_history, chat_input, user_id, user_text, image_path)
        
        if vira_response and vira_response != "ERROR":
            self.executor.submit(
                self._run_advanced_analysis, 
                user_id, user_text, vira_response, current_summary
            )
            
            self._update_local_history(user_id, user_text, vira_response, image_path)
        
        return vira_response

    def _generate_with_retry(self, gemini_history: List[Dict], chat_input: List, 
                            user_id: str, user_text: str, image_path: str, 
                            max_retries: int = 2) -> str:
        
        from google.generativeai.types import HarmCategory, HarmBlockThreshold
        
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        for attempt in range(max_retries + 1):
            try:
                chat_session = self.chat_instance.start_chat(history=gemini_history)
                response = chat_session.send_message(
                    chat_input,
                    generation_config=GenerationConfig(
                        temperature=0.85,
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=2048
                    ),
                    safety_settings=safety_settings
                )
                return response.text
                
            except Exception as e:
                error_str = str(e).lower()
                logger.error(f"[CHAT-FAIL] Attempt {attempt + 1}: {e}")
                
                if "quota" in error_str or "429" in error_str or "resource_exhausted" in error_str:
                    self._mark_key_unhealthy(self.current_key_index)
                    
                    if self._rotate_api_key():
                        continue
                    else:
                        return "Maaf, semua API sedang sibuk. Coba lagi sebentar lagi."
                
                elif "500" in error_str or "503" in error_str:
                    if attempt < max_retries:
                        import time
                        time.sleep(2 ** attempt)
                        continue
                
                elif attempt < max_retries:
                    continue
                    
        return "Maaf, kepalaku agak pusing. Bisa ulangi?"

    def _run_advanced_analysis(self, user_id: str, user_text: str, vira_text: str, old_summary: str):
        if not user_text or not vira_text:
            return
        
        gemma_input_prompt = (
            f"{FIRST_LEVEL_ANALYSIS_INSTRUCTION}\n\n"
            f"--- START INTERACTION TO ANALYZE ---\n"
            f"User: {user_text}\n"
            f"AI: {vira_text}\n"
            f"--- END INTERACTION ---\n"
        )

        try:
            gemma_res = self.first_level_instance.generate_content(
                gemma_input_prompt,
                generation_config=GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=2048
                )
            )
            gemma_insight = gemma_res.text
            logger.debug(f"[GEMMA] {gemma_insight}...")
            
            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            flash_prompt = (
                f"[CURRENT SYSTEM TIME]: {now_str}\n"
                f"[EXPERT ANALYSIS SOURCE]\n{gemma_insight}\n\n"
                f"[ORIGINAL CONTEXT]\nUser: {user_text}\nAI: {vira_text}\nSummary: {old_summary}\n\n"
                f"TASK: Extract structured data based on expert analysis.\n"
                f"RULES:\n"
                f"1. Convert relative time to ISO 8601 format (YYYY-MM-DDTHH:MM:SS)\n"
                f"2. Be conservative - only extract high-confidence information\n"
                f"3. Prioritize quality over quantity\n"
            )
            
            flash_res = self.analysis_instance.generate_content(
                flash_prompt,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                )
            )
            
            logger.debug(f"[FLASH] {flash_res.text[:200]}...")
            data = self._safe_json_parse(flash_res.text)
            
            if data:
                self._process_analysis_results(user_id, data)
                
        except Exception as e:
            logger.error(f"[ANALYSIS-PIPELINE] Error: {e}", exc_info=True)

    def _safe_json_parse(self, text: str) -> Optional[Dict]:
        try:
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"[JSON-PARSE] Failed: {e}")
            return None

    def _process_analysis_results(self, user_id: str, data: Dict):
        if "updated_summary" in data and data["updated_summary"]:
            new_summary = data["updated_summary"].strip()
            if new_summary and len(new_summary) > 10:
                self.session_summaries[user_id] = new_summary
        
        mem_data = data.get("memory", {})
        if isinstance(mem_data, dict):
            if mem_data.get("should_store"):
                self._save_memory_to_db(user_id, mem_data)
        elif isinstance(mem_data, list):
            for m in mem_data:
                if isinstance(m, dict) and m.get("should_store"):
                    self._save_memory_to_db(user_id, m)

        schedules_list = data.get("schedules", [])
        if isinstance(schedules_list, dict):
            schedules_list = [schedules_list]

        for item in schedules_list:
            if isinstance(item, dict) and item.get("should_schedule"):
                self._process_new_schedule(user_id, item)

    def _process_new_schedule(self, user_id: str, data: Dict):
        try:
            raw_time = data.get("time_str")
            context = data.get("context")
            
            if not raw_time or not context:
                return
            
            context = context.strip()
            if len(context) < 5:
                return
            
            trigger_time = parser.parse(raw_time, fuzzy=True)
            now = datetime.datetime.now()
            
            if trigger_time.year < 2000:
                trigger_time = trigger_time.replace(year=now.year, month=now.month, day=now.day)
            
            if trigger_time.year == now.year and trigger_time.month == now.month and trigger_time.day == now.day:
                if trigger_time < now:
                    trigger_time += datetime.timedelta(days=1)
            
            if trigger_time < now:
                logger.warning(f"[SCHEDULE-SKIP] Time in past: {trigger_time}")
                return

            self.scheduler_service.add_schedule(user_id, trigger_time, context)
            logger.info(f"[SCHEDULE-SAVED] {user_id} @ {trigger_time.isoformat()}")
            
        except Exception as e:
            logger.warning(f"[SCHEDULE-FAIL] {e}")

    def trigger_proactive_message(self, user_id: str, context: str) -> Optional[str]:
        history = self.get_session_history(user_id)
        current_summary = self.session_summaries.get(user_id, "")
        
        proactive_prompt = (
            f"[SYSTEM TRIGGER] Jadwal pengingat tiba.\n"
            f"Konteks: {context}\n"
            f"Ringkasan User: {current_summary}\n"
            f"Tugas: Sapa user dan ingatkan dengan natural, hangat, dan empati."
        )
        
        gemini_history = self._prepare_history_for_gemini(history)
        
        try:
            chat_session = self.chat_instance.start_chat(history=gemini_history)
            response = chat_session.send_message(
                proactive_prompt,
                generation_config=GenerationConfig(
                    temperature=0.9,
                    max_output_tokens=512
                )
            )
            vira_text = response.text
            self._update_local_history(user_id, "(System Reminder)", vira_text, None)
            return vira_text
        except Exception as e:
            logger.error(f"[PROACTIVE-FAIL] {e}")
            return None

    def _save_memory_to_db(self, user_id: str, mem_data: Dict):
        summary = mem_data.get("summary", "").strip()
        if not summary or len(summary) < 5:
            return
        
        try:
            vector = self.analyzer.get_embedding(summary)
            self.memory_manager.add_memory(
                user_id, 
                summary, 
                mem_data.get("type", "preference"), 
                mem_data.get("priority", 0.5), 
                vector
            )
        except Exception as e:
            logger.error(f"[MEMORY-SAVE-ERROR] {e}")

    def _update_local_history(self, user_id: str, u_text: str, v_text: str, img_path: str):
        if user_id not in self.active_sessions:
            self.active_sessions[user_id] = deque()
        
        history = self.active_sessions[user_id]
        
        u_parts = [u_text]
        if img_path and os.path.exists(img_path): 
            u_parts.append(f"{self.IMG_PREFIX}{img_path}")
        
        history.append({"role": "user", "parts": u_parts})
        history.append({"role": "model", "parts": [v_text]})
        
        while len(history) > self.MAX_HISTORY_LENGTH: 
            history.popleft()
        
        self.last_interactions[user_id] = datetime.datetime.now()
        self._save_session_to_disk(user_id)

    def _prepare_history_for_gemini(self, raw_history: deque) -> List[Dict]:
        gemini_history = []
        for msg in raw_history:
            parts = []
            for item in msg["parts"]:
                if isinstance(item, str):
                    if item.startswith(self.IMG_PREFIX):
                        path = item.replace(self.IMG_PREFIX, "")
                        if os.path.exists(path):
                            try: 
                                parts.append(PIL.Image.open(path))
                            except Exception as e:
                                logger.error(f"[IMG-LOAD-ERROR] {e}")
                    else:
                        parts.append(item)
                else:
                    parts.append(item)
            
            if parts:
                gemini_history.append({"role": msg["role"], "parts": parts})
        
        return gemini_history

    def _get_session_path(self, user_id: str) -> str:
        return os.path.join(self.SESSION_DIR, f"{user_id}.json")

    def _load_session_from_disk(self, user_id: str):
        path = self._get_session_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f: 
                    data = json.load(f)
                
                self.session_summaries[user_id] = data.get("summary", "")
                self.active_sessions[user_id] = deque(data.get("history", []))
                
                last_ts = data.get("last_interaction_ts")
                if last_ts and last_ts != "None":
                    try:
                        self.last_interactions[user_id] = datetime.datetime.fromisoformat(last_ts)
                    except:
                        self.last_interactions[user_id] = None
                else:
                    self.last_interactions[user_id] = None
                    
            except Exception as e:
                logger.error(f"[SESSION-LOAD-ERROR] {e}")
                self._init_empty_session(user_id)
        else: 
            self._init_empty_session(user_id)

    def _init_empty_session(self, user_id: str):
        self.active_sessions[user_id] = deque()
        self.session_summaries[user_id] = ""
        self.last_interactions[user_id] = None

    def _save_session_to_disk(self, user_id: str):
        try:
            data = {
                "summary": self.session_summaries.get(user_id, ""),
                "history": list(self.active_sessions.get(user_id, [])),
                "last_interaction_ts": self.last_interactions.get(user_id).isoformat() 
                    if self.last_interactions.get(user_id) else None
            }
            
            path = self._get_session_path(user_id)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            logger.error(f"[SESSION-SAVE-ERROR] {e}")

    def get_session_history(self, user_id: str) -> deque:
        if user_id not in self.active_sessions: 
            self._load_session_from_disk(user_id)
        return self.active_sessions[user_id]
        
    def clear_session(self, user_id: str):
        self._init_empty_session(user_id)
        path = self._get_session_path(user_id)
        if os.path.exists(path): 
            try:
                os.remove(path)
                logger.info(f"[SESSION-CLEAR] {user_id}")
            except Exception as e:
                logger.error(f"[SESSION-CLEAR-ERROR] {e}")
    
    def get_system_stats(self) -> Dict:
        return {
            "active_sessions": len(self.active_sessions),
            "current_api_key_index": self.current_key_index,
            "api_key_health": self.key_health_status,
            "cache_size": len(self.context_cache),
            "models": {
                "chat": self.chat_model_name,
                "analysis": self.analysis_model_name,
                "first_level": self.first_level_model_name
            }
        }
    
    def __del__(self):
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)