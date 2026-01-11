import os
import logging
import time
import threading
import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional

import PIL.Image
from dateutil import parser
from google import genai
from google.genai import types

from src.config import (
    GOOGLE_API_KEYS,
    CHAT_MODEL,
    TIER_1_MODEL,
    TIER_2_MODEL,
    TIER_3_MODEL,
    INSTRUCTION,
    FIRST_LEVEL_ANALYSIS_INSTRUCTION,
    CHAT_ANALYSIS_INSTUCTION,
    MEMORY_ANALYSIS_INSTRUCTION,
    METEOSOURCE_API_KEY,
    AVAILABLE_CHAT_MODELS,
    set_chat_model
)

from src.services.memory_service import MemoryManager
from src.services.analyzer_service import MemoryAnalyzer
from src.services.scheduler_service import SchedulerService
from src.services.handler.api_key_monitor import APIKeyHealthMonitor
from src.services.handler.session_manager import SessionManager
from src.services.handler.context_builder import ContextBuilder
from src.services.handler.analysis_service import AnalysisService

logger = logging.getLogger(__name__)

class ChatHandler:
    def __init__(self, memory_manager: MemoryManager, analyzer: MemoryAnalyzer, 
                 scheduler_service: SchedulerService):
        self.memory_manager = memory_manager
        self.analyzer = analyzer
        self.scheduler_service = scheduler_service
        
        self.api_keys = GOOGLE_API_KEYS
        if not self.api_keys:
            raise ValueError("No API keys configured")
        
        self.current_key_index = 0
        self.health_monitor = APIKeyHealthMonitor(len(self.api_keys))
        self.client: Optional[genai.Client] = None
        self._configure_genai()
        
        self.chat_model_name = CHAT_MODEL
        self.tier_1_model_name = TIER_1_MODEL
        self.tier_2_model_name = TIER_2_MODEL
        self.tier_3_model_name = TIER_3_MODEL
        
        self.session_manager = SessionManager()
        self.context_builder = ContextBuilder(METEOSOURCE_API_KEY)
        self.analysis_service = AnalysisService(
            self.client, 
            self.tier_1_model_name, 
            self.tier_2_model_name
        )
        
        self.executor = ThreadPoolExecutor(max_workers=3)
        self._session_processing_flags: Dict[str, bool] = {}
        self._flag_lock = threading.Lock()

    def clear_session(self, user_id: str):
        """Clear user session and reset all data"""
        self.session_manager.clear_session(user_id)
        self.context_builder.clear_cache()
        logger.info(f"[SESSION-CLEARED] User: {user_id}")

    def _configure_genai(self):
        try:
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            logger.info(f"[SYSTEM] Client initialized with Key-{self.current_key_index}")
        except Exception as e:
            logger.error(f"[INIT-ERROR] {e}")
            raise

    def _rotate_api_key(self) -> bool:
        if len(self.api_keys) <= 1:
            logger.warning("[SYSTEM] Cannot rotate: only one API key available")
            return False
        
        next_key = self.health_monitor.get_healthy_key(
            self.current_key_index, 
            len(self.api_keys)
        )
        
        if next_key is None:
            logger.error("[SYSTEM] No healthy API keys available")
            return False
        
        self.current_key_index = next_key
        
        try:
            self._configure_genai()
            self.analysis_service.client = self.client
            logger.warning(f"[SYSTEM] Rotated to Key-{self.current_key_index}")
            return True
        except Exception as e:
            logger.error(f"[ROTATION-ERROR] {e}")
            return False

    def change_model(self, model_name: str):
        if model_name not in AVAILABLE_CHAT_MODELS:
            raise ValueError(f"Model not available: {model_name}")
        
        self.chat_model_name = model_name
        set_chat_model(model_name)
        logger.info(f"[SYSTEM] Model changed to: {model_name}")

    def process_message(self, user_id: str, user_text: str, image_path: str = None) -> str:
        with self._flag_lock:
            if self._session_processing_flags.get(user_id, False):
                return "Mohon tunggu, pesan sebelumnya masih diproses..."
            self._session_processing_flags[user_id] = True
        
        try:
            self.session_manager.cleanup_inactive_sessions()
            return self._process_message_internal(user_id, user_text, image_path)
        
        except Exception as e:
            logger.error(f"[PROCESS-MESSAGE-ERROR] {user_id}: {e}")
            return "Maaf, terjadi kesalahan sistem."
        
        finally:
            with self._flag_lock:
                self._session_processing_flags[user_id] = False

    def _process_message_internal(self, user_id: str, user_text: str, 
                                  image_path: str = None) -> str:
        history = self.session_manager.get_session(user_id)
        current_summary = self.session_manager.get_summary(user_id)
        
        pending_sched = self.scheduler_service.get_pending_schedule_for_user(user_id)
        schedule_ctx = None
        
        if pending_sched:
            schedule_ctx = pending_sched.get('context')
            self.scheduler_service.mark_as_executed(pending_sched['id'])
        
        relevant_memories = []
        if user_text and len(user_text.strip()) > 3:
            try:
                emb = self.analyzer.get_embedding(user_text)
                if emb:
                    relevant_memories = self.memory_manager.get_relevant_memories(user_id, emb)
            except Exception as e:
                logger.error(f"[EMBEDDING-ERROR] {e}")
        
        memory_summary = self.session_manager.get_memory_summary(user_id)
        last_interaction = self.session_manager.last_interactions.get(user_id)
        
        system_context = self.context_builder.build_context(
            user_id, 
            current_summary, 
            relevant_memories,
            memory_summary,
            last_interaction,
            schedule_ctx
        )
        
        full_prompt = f"{system_context}\n\n[PESAN BARU]\nInput: {user_text}"
        chat_input = [types.Part(text=full_prompt)]
        
        if image_path and os.path.exists(image_path):
            try:
                chat_input.append(PIL.Image.open(image_path))
            except Exception as e:
                logger.error(f"[IMG-ERROR] {image_path}: {e}")
        
        gemini_history = self.session_manager.prepare_history_for_gemini(
            history, 
            self.session_manager.IMG_PREFIX
        )
        
        response = self._generate_with_retry(gemini_history, chat_input)
        
        if response and response != "ERROR":
            self.session_manager.update_session(
                user_id, 
                user_text, 
                response, 
                image_path,
                lambda uid: self.executor.submit(self._handle_chat_analysis, uid)
            )
            
            self.executor.submit(
                self._handle_interaction_analysis, 
                user_id, 
                user_text, 
                response
            )
        
        return response

    def _generate_with_retry(self, history: List[types.Content], 
                           chat_input: List, max_retries: int = 3) -> str:
        config = types.GenerateContentConfig(
            temperature=0.6,
            top_p=0.85,
            top_k=40,
            max_output_tokens=1024,
            system_instruction=INSTRUCTION
        )
        
        for attempt in range(max_retries + 1):
            try:
                chat = self.client.chats.create(
                    model=self.chat_model_name,
                    history=history,
                    config=config
                )
                
                resp = chat.send_message(chat_input)
                
                if resp.text:
                    self.health_monitor.mark_success(self.current_key_index)
                    logger.info(f"[MODEL-OUTPUT: {self.chat_model_name}] (Main Chat): {resp.text[:100]}...")
                    return resp.text
                
                logger.warning("[CHAT] Empty response from model")
                return ""
            
            except Exception as e:
                err_msg = str(e).lower()
                logger.error(f"[CHAT-FAIL] Attempt {attempt + 1}/{max_retries + 1}: {e}")
                
                is_quota_error = any(
                    keyword in err_msg 
                    for keyword in ["quota", "429", "exhausted", "rate limit"]
                )
                
                if is_quota_error:
                    self.health_monitor.mark_failure(self.current_key_index)
                    
                    if self._rotate_api_key():
                        continue
                    else:
                        return "Maaf, sistem sedang sibuk. Silakan coba beberapa saat lagi."
                
                if attempt < max_retries:
                    sleep_time = min(2 ** attempt, 8)
                    time.sleep(sleep_time)
                    continue
        
        return "Maaf, terjadi kesalahan sistem. Silakan coba lagi."

    def _handle_chat_analysis(self, user_id: str):
        try:
            history_list = list(self.session_manager.get_session(user_id))
            old_summary = self.session_manager.get_summary(user_id)
            
            new_summary = self.analysis_service.run_chat_analysis(
                history_list,
                old_summary,
                CHAT_ANALYSIS_INSTUCTION
            )
            
            if new_summary:
                self.session_manager.update_summary(user_id, new_summary)
                self.session_manager.delete_last_n_messages(user_id, 10)
        
        except Exception as e:
            logger.error(f"[CHAT-ANALYSIS-HANDLER-ERROR] {e}")

    def _handle_interaction_analysis(self, user_id: str, user_text: str, ai_text: str):
        try:
            result = self.analysis_service.run_interaction_analysis(
                user_text,
                ai_text,
                FIRST_LEVEL_ANALYSIS_INSTRUCTION
            )
            
            if result:
                self._process_analysis_results(user_id, result)
        
        except Exception as e:
            logger.error(f"[INTERACTION-ANALYSIS-HANDLER-ERROR] {e}")

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
            time_str = data.get("time_str")
            context = data.get("context")
            
            if not time_str or not context or len(context) < 5:
                return
            
            trigger = parser.parse(time_str, fuzzy=True)
            now = datetime.datetime.now()
            
            if trigger.year < 2000:
                trigger = trigger.replace(year=now.year, month=now.month, day=now.day)
            
            if trigger.date() == now.date() and trigger < now:
                trigger += datetime.timedelta(days=1)
            
            if trigger < now:
                logger.warning(f"[SCHEDULE-SKIP] Time in past: {trigger}")
                return
            
            self.scheduler_service.add_schedule(user_id, trigger, context)
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
                    user_id,
                    summary,
                    data.get("type", "preference"),
                    data.get("priority", 0.5),
                    vec
                )
                
                logger.info(f"[MEMORY-SAVED] {user_id}: {summary[:50]}...")
                
                self.executor.submit(self._handle_memory_analysis, user_id, 40)
        
        except Exception as e:
            logger.error(f"[MEMORY-SAVE-ERROR] {e}")

    def _handle_memory_analysis(self, user_id: str, data_number: int):
        try:
            result = self.memory_manager.search_memories(user_id, "", data_number)
            
            if not result:
                return
            
            current_count = len(self.memory_manager.search_memories(user_id, "", 1000))
            
            if not self.session_manager.should_run_memory_analysis(user_id, current_count):
                logger.info(f"[MEMORY-ANALYSIS-SKIP] {user_id}: conditions not met")
                return
            
            old_memory_summary = self.session_manager.get_memory_summary(user_id)
            
            new_summary = self.analysis_service.run_memory_analysis(
                result,
                old_memory_summary,
                MEMORY_ANALYSIS_INSTRUCTION
            )
            
            if new_summary:
                self.session_manager.update_memory_summary(user_id, new_summary)
                self.session_manager.mark_memory_analysis_done(user_id, current_count)
        
        except Exception as e:
            logger.error(f"[MEMORY-ANALYSIS-HANDLER-ERROR] {e}")

    def trigger_proactive_message(self, user_id: str, context: str) -> Optional[str]:
        try:
            history = self.session_manager.prepare_history_for_gemini(
                self.session_manager.get_session(user_id),
                self.session_manager.IMG_PREFIX
            )
            
            prompt = (
                f"[SYSTEM TRIGGER] Jadwal pengingat.\n"
                f"Konteks: {context}\n"
                f"Summary: {self.session_manager.get_summary(user_id)}\n"
                f"Tugas: Sapa dan ingatkan user dengan ramah."
            )
            
            resp = self._process_message_internal(user_id, prompt)
            if resp.text:
                logger.info(f"[MODEL-OUTPUT: {self.chat_model_name}] (Proactive): {resp.text[:100]}...")
                self.session_manager.update_session(
                    user_id,
                    "(System Reminder)",
                    resp.text,
                    None
                )
                return resp.text
        
        except Exception as e:
            logger.error(f"[PROACTIVE-ERROR] {e}")
        
        return None

    def get_system_stats(self) -> Dict:
        return {
            "active_sessions": len(self.session_manager.sessions),
            "current_key": self.current_key_index,
            "api_health": self.health_monitor.get_status(),
            "cache_size": self.context_builder.get_cache_size(),
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