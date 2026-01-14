import os
import logging
import time
import threading
import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional, Any

import PIL.Image
from dateutil import parser
from google import genai
from google.genai import types

from src.config import (
    GOOGLE_API_KEYS, CHAT_MODEL, TIER_1_MODEL, TIER_2_MODEL, TIER_3_MODEL,
    INSTRUCTION, CHAT_INTERACTION_ANALYSIS_INSTRUCTION, CHAT_ANALYSIS_INSTUCTION,
    MEMORY_ANALYSIS_INSTRUCTION, METEOSOURCE_API_KEY, AVAILABLE_CHAT_MODELS,
    set_chat_model
)

from src.services.handler.canonicalizer import MemoryCanonicalizer
from src.services.handler.memory_gatekeeper import MemoryGatekeeper

from src.services.memory_manager import MemoryManager
from src.services.scheduler_service import SchedulerService
from src.services.embedding import MemoryAnalyzer
from src.services.handler.intent_extractor import IntentExtractor, IntentType, RequestType
from src.services.handler.api_key_monitor import APIKeyHealthMonitor
from src.services.handler.session_manager import SessionManager
from src.services.handler.context_builder import ContextBuilder
from src.services.handler.analysis_service import AnalysisService

logger = logging.getLogger(__name__)


class ChatHandler:
    CLEANUP_INTERVAL = 300
    MAX_RETRIES = 5
    BASE_RETRY_DELAY = 1
    MAX_OUTPUT_TOKENS = 1024
    TEMPERATURE = 0.65
    TOP_P = 0.9
    
    def __init__(self, memory_manager: MemoryManager, analyzer: MemoryAnalyzer, 
                scheduler_service: SchedulerService):
        if not GOOGLE_API_KEYS:
            raise ValueError("No API keys configured")
        
        self.memory_manager = memory_manager
        self.analyzer = analyzer
        self.scheduler_service = scheduler_service
        
        self.api_keys = GOOGLE_API_KEYS
        self.current_key_index = 0
        self.health_monitor = APIKeyHealthMonitor(len(self.api_keys))
        
        self.client: Optional[genai.Client] = None
        self._initialize_client()
        
        self.chat_model_name = CHAT_MODEL
        self.tier_1_model = TIER_1_MODEL
        self.tier_2_model = TIER_2_MODEL
        self.tier_3_model = TIER_3_MODEL
        
        if hasattr(memory_manager, 'canonicalizer') and memory_manager.canonicalizer is None:
            self.canonicalizer = MemoryCanonicalizer(self.client, self.tier_2_model)
            memory_manager.gatekeeper = MemoryGatekeeper(memory_manager.db, memory_manager.emb_handler)
            memory_manager.use_canonicalization = True
        
        self.intent_extractor = IntentExtractor(self.client, self.tier_3_model)
        
        self.session_manager = SessionManager()
        self.context_builder = ContextBuilder(METEOSOURCE_API_KEY)
        self.analysis_service = AnalysisService(self.client, TIER_1_MODEL, TIER_2_MODEL)
        
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ChatWorker")
        
        self._processing_flags: Dict[str, bool] = {}
        self._flag_lock = threading.Lock()
        self._last_cleanup = time.time()

    def _initialize_client(self):
        try:
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            if hasattr(self, 'analysis_service'):
                self.analysis_service.client = self.client
            logger.info(f"Initialized with API Key #{self.current_key_index}")
        except Exception as e:
            logger.critical(f"Client initialization failed: {e}")
            raise

    def _rotate_api_key(self) -> bool:
        new_key_index = self.health_monitor.get_healthy_key(
            self.current_key_index, 
            len(self.api_keys)
        )
        
        if new_key_index is None or new_key_index == self.current_key_index:
            new_key_index = (self.current_key_index + 1) % len(self.api_keys)
        
        self.current_key_index = new_key_index
        
        try:
            self._initialize_client()
            logger.warning(f"Rotated to API Key #{self.current_key_index}")
            return True
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            return False

    def change_model(self, model_name: str):
        if model_name not in AVAILABLE_CHAT_MODELS:
            logger.warning(f"Invalid model: {model_name}")
            return
        
        self.chat_model_name = model_name
        set_chat_model(model_name)
        logger.info(f"Chat model changed to: {model_name}")

    def clear_session(self, user_id: str):
        self.session_manager.clear_session(user_id)
        self.context_builder.clear_cache()
        logger.info(f"Session cleared for user {user_id}")

    def process_message(self, user_id: str, user_text: str, image_path: str = None) -> str:
        if not self._acquire_processing_lock(user_id):
            return "⏳ Please wait, I'm still processing your previous message..."
        
        try:
            self._periodic_cleanup()
            return self._execute_chat_flow(user_id, user_text, image_path)
        except Exception as e:
            logger.error(f"Process error for user {user_id}: {e}", exc_info=True)
            return "❌ An internal error occurred. Please try again."
        finally:
            self._release_processing_lock(user_id)

    def _acquire_processing_lock(self, user_id: str) -> bool:
        with self._flag_lock:
            if self._processing_flags.get(user_id, False):
                return False
            self._processing_flags[user_id] = True
            return True

    def _release_processing_lock(self, user_id: str):
        with self._flag_lock:
            self._processing_flags[user_id] = False

    def _periodic_cleanup(self):
        current_time = time.time()
        if current_time - self._last_cleanup > self.CLEANUP_INTERVAL:
            self.executor.submit(self.session_manager.cleanup_inactive_sessions)
            self._last_cleanup = current_time

    def _execute_chat_flow(self, user_id: str, user_text: str, image_path: str) -> str:
        
        intent_data = self.intent_extractor.extract(user_text)
        logger.info(f"[INTENT] User intent: {intent_data['intent_type']} | Request: {intent_data['request_type']}")
        
        session_data = self._gather_session_data(user_id)
        schedule_context = self._process_pending_schedule(user_id)
        
        relevant_memories = []
        if intent_data.get('needs_memory', False):
            relevant_memories = self._retrieve_relevant_memories_with_intent(
                user_id, user_text, intent_data
            )
        else:
            logger.info(f"[INTENT] Skipping memory retrieval (not needed)")
        
        system_context = self.context_builder.build_context(
            user_id,
            session_data['summary'],
            relevant_memories,
            session_data['memory_summary'],
            session_data['last_interaction'],
            schedule_context,
            session_data['schedule_summary'],
            intent_context=intent_data
        )
        
        chat_input = self._build_chat_input(system_context, user_text, image_path)
        gemini_history = self.session_manager.prepare_history_for_gemini(
            session_data['history'],
            self.session_manager.IMG_PREFIX
        )
        
        response_text = self._generate_response(gemini_history, chat_input)
        
        if response_text:
            self._post_process_response(user_id, user_text, response_text, image_path)
        
        return response_text or "❌ Unable to generate response. Please try again."
    
    def _retrieve_relevant_memories_with_intent(self, user_id: str, 
                                                user_text: str,
                                                intent_data: Dict) -> List[Dict]:
        """Enhanced memory retrieval with intent context"""
        
        search_query = intent_data.get('search_query', user_text)
        memory_scope = intent_data.get('memory_scope')
        entities = intent_data.get('entities', [])
        request_type = intent_data.get('request_type')
        
        logger.info(f"[MEMORY-RETRIEVAL] Query: '{search_query}' | Scope: {memory_scope} | Entities: {entities}")
        
        try:
            query_embedding = self.analyzer.get_embedding(search_query)
            if query_embedding is None:
                logger.warning("[MEMORY-RETRIEVAL] Failed to generate embedding")
                return []
            
            embedding_list = (
                query_embedding.tolist() 
                if isinstance(query_embedding, np.ndarray) 
                else query_embedding
            )
            
            memory_type = self._map_scope_to_type(memory_scope)
            
            max_results = self._determine_max_results(request_type)
            
            memories = self.memory_manager.get_relevant_memories(
                user_id=user_id,
                query_embedding=embedding_list,
                memory_type=memory_type,
                max_results=max_results,
                use_clusters=True,
                query_text=search_query,
                entities=entities
            )
            
            logger.info(f"[MEMORY-RETRIEVAL] Found {len(memories)} relevant memories")
            return memories
        
        except Exception as e:
            logger.error(f"[MEMORY-RETRIEVAL] Failed: {e}")
            return []
    
    def _map_scope_to_type(self, memory_scope: Optional[str]) -> Optional[str]:
        """Map intent memory scope to memory type"""
        scope_mapping = {
            'preference': 'preference',
            'personal': 'general',
            'factual': 'fact',
            'general': None
        }
        return scope_mapping.get(memory_scope) if memory_scope else None
    
    def _determine_max_results(self, request_type: str) -> int:
        """Determine how many memories to retrieve based on request type"""
        limits = {
            'memory_recall': 3,
            'recommendation': 7,
            'information': 5,
            'opinion': 4,
            'general_chat': 2
        }
        return limits.get(request_type, 5)

    def _gather_session_data(self, user_id: str) -> Dict:
        return {
            'history': self.session_manager.get_session(user_id),
            'summary': self.session_manager.get_metadata(user_id, "summary", ""),
            'memory_summary': self.session_manager.get_metadata(user_id, "memory_summary", ""),
            'last_interaction': self.session_manager.get_metadata(user_id, "last_interaction"),
            'schedule_summary': self.session_manager.get_schedule_summary(user_id)
        }

    def _process_pending_schedule(self, user_id: str) -> Optional[str]:
        pending_schedule = self.scheduler_service.get_pending_schedule_for_user(user_id)
        if not pending_schedule:
            return None
        
        schedule_context = pending_schedule.get('context')
        self.scheduler_service.mark_as_executed(
            pending_schedule['id'],
            note="Triggered by user interaction"
        )
        return schedule_context

    def _retrieve_relevant_memories(self, user_id: str, user_text: str) -> List[Dict]:
        if not user_text or len(user_text) <= 3:
            return []
        
        try:
            embedding_vector = self.analyzer.get_embedding(user_text)
            if embedding_vector is None:
                return []
            
            embedding_list = (
                embedding_vector.tolist() 
                if isinstance(embedding_vector, np.ndarray) 
                else embedding_vector
            )
            
            return self.memory_manager.get_relevant_memories(
                user_id=user_id,
                query_embedding=embedding_list,
                max_results=5,
                use_clusters=True
            )
        except Exception as e:
            logger.warning(f"Memory retrieval failed for user {user_id}: {e}")
            return []

    def _build_chat_input(self, system_context: str, user_text: str, 
                          image_path: Optional[str]) -> List:
        full_prompt = f"{system_context}\n\n[USER MESSAGE]\n{user_text}"
        chat_input = [types.Part(text=full_prompt)]
        
        if image_path and os.path.exists(image_path):
            try:
                image = PIL.Image.open(image_path)
                chat_input.append(image)
            except Exception as e:
                logger.error(f"Image processing error: {e}")
        
        return chat_input

    def _generate_response(self, history: List, chat_input: List) -> str:
        config = types.GenerateContentConfig(
            temperature=self.TEMPERATURE,
            top_p=self.TOP_P,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
            system_instruction=INSTRUCTION
        )
        
        for attempt in range(self.MAX_RETRIES):
            try:
                chat = self.client.chats.create(
                    model=self.chat_model_name,
                    history=history,
                    config=config
                )
                response = chat.send_message(chat_input)
                
                if response.text:
                    self.health_monitor.mark_success(self.current_key_index)
                    return response.text
                
                logger.warning(f"Empty response on attempt {attempt + 1}")
                
            except Exception as e:
                self._handle_generation_error(e, attempt)
        
        return ""

    def _handle_generation_error(self, error: Exception, attempt: int):
        error_message = str(error).lower()
        logger.warning(f"Generation attempt {attempt + 1} failed: {error}")
        
        if any(keyword in error_message for keyword in ["429", "quota", "resource exhausted"]):
            self.health_monitor.mark_failure(self.current_key_index)
            if not self._rotate_api_key():
                return
        
        time.sleep(self.BASE_RETRY_DELAY + attempt)

    def _post_process_response(self, user_id: str, user_text: str, 
                                response_text: str, image_path: Optional[str]):
        self.session_manager.update_session(
            user_id, 
            user_text, 
            response_text, 
            image_path,
            lambda uid: self.executor.submit(self._analyze_chat_session, uid)
        )
        
        self.executor.submit(
            self._analyze_interaction, 
            user_id, 
            user_text, 
            response_text
        )

    def _analyze_chat_session(self, user_id: str):
        try:
            history = list(self.session_manager.get_session(user_id))
            current_summary = self.session_manager.get_summary(user_id)
            
            new_summary = self.analysis_service.run_chat_analysis(
                history, 
                current_summary, 
                CHAT_ANALYSIS_INSTUCTION
            )
            
            if new_summary and new_summary != current_summary:
                self.session_manager.update_summary(user_id, new_summary)
                self.session_manager.delete_last_n_messages(user_id, 8)
        except Exception as e:
            logger.error(f"Chat analysis failed for user {user_id}: {e}")

    def _analyze_interaction(self, user_id: str, user_text: str, ai_text: str):
        try:
            analysis_data = self.analysis_service.run_interaction_analysis(
                user_text, 
                ai_text, 
                CHAT_INTERACTION_ANALYSIS_INSTRUCTION
            )
            
            if not analysis_data:
                return
            
            self._process_memory_data(user_id, analysis_data.get("memory", []))
            self._process_schedule_data(user_id, analysis_data.get("schedules", []))
        except Exception as e:
            logger.error(f"Interaction analysis failed for user {user_id}: {e}")

    def _process_memory_data(self, user_id: str, memory_items: Any):
        if isinstance(memory_items, dict):
            memory_items = [memory_items]
        
        for memory_item in memory_items:
            # Added check to ensure memory_item is a dict before calling .get()
            if isinstance(memory_item, dict) and memory_item.get("should_store"):
                self._handle_memory_operation(user_id, memory_item)

    def _process_schedule_data(self, user_id: str, schedule_items: Any):
        if isinstance(schedule_items, dict):
            schedule_items = [schedule_items]
        
        for schedule_item in schedule_items:
            if isinstance(schedule_item, dict) and schedule_item.get("should_schedule"):
                self._handle_schedule_operation(user_id, schedule_item)

    def _handle_memory_operation(self, user_id: str, memory_data: Dict):
        summary = memory_data.get("summary")
        if not summary:
            return
        
        try:
            embedding_vector = self.analyzer.get_embedding(summary)
            embedding_list = None
            
            if embedding_vector is not None:
                embedding_list = (
                    embedding_vector.tolist() 
                    if isinstance(embedding_vector, np.ndarray) 
                    else embedding_vector
                )
            
            action = memory_data.get("action", "add")
            
            if action == "forget":
                self.memory_manager.forget_memory(
                    user_id=user_id,
                    query=summary,
                    embedding=embedding_list
                )
            else:
                self.memory_manager.add_memory(
                    user_id=user_id,
                    summary=summary,
                    m_type=memory_data.get("type", "general"),
                    priority=memory_data.get("priority", 0.5),
                    embedding=embedding_list
                )
                self.executor.submit(self._consolidate_memories, user_id)
        except Exception as e:
            logger.error(f"Memory operation failed for user {user_id}: {e}")

    def _consolidate_memories(self, user_id: str):
        try:
            stats = self.memory_manager.get_memory_stats(user_id)
            total_active = stats.get('active', 0)
            
            if not self.session_manager.should_run_memory_analysis(user_id, total_active):
                return
            
            recent_memories = self.memory_manager.search_memories(user_id, "", limit=20)
            
            # Fix: Pass raw memory objects (dicts) instead of formatted strings.
            # AnalysisService expects objects with .get() method, passing strings caused AttributeError.
            current_summary = self.session_manager.get_memory_summary(user_id)
            new_summary = self.analysis_service.run_memory_analysis(
                recent_memories, 
                current_summary,
                MEMORY_ANALYSIS_INSTRUCTION
            )
            
            if new_summary:
                self.session_manager.update_memory_summary(user_id, new_summary)
                self.session_manager.mark_memory_analysis_done(user_id, total_active)
        except Exception as e:
            logger.error(f"Memory consolidation failed for user {user_id}: {e}")

    def _handle_schedule_operation(self, user_id: str, schedule_data: Dict):
        try:
            intent = schedule_data.get("intent", "add")
            time_string = schedule_data.get("time_str")
            context = schedule_data.get("context")
            
            if not time_string:
                return
            
            trigger_datetime = self._parse_datetime(time_string, intent)
            if trigger_datetime is None:
                return
            
            schedule_modified = False
            
            if intent == "cancel":
                schedule_modified = self.scheduler_service.cancel_schedule_by_context(
                    user_id=user_id,
                    time_hint=trigger_datetime,
                    context_hint=context
                )
                if schedule_modified:
                    logger.info(f"Schedule cancelled for user {user_id}")
            else:
                priority = schedule_data.get("priority", 0)
                schedule_id = self.scheduler_service.add_schedule(
                    user_id=user_id,
                    trigger_time=trigger_datetime,
                    context=context,
                    priority=priority
                )
                
                if schedule_id:
                    logger.info(f"Schedule created (ID: {schedule_id}) for user {user_id}")
                    schedule_modified = True
            
            if schedule_modified:
                self.executor.submit(self._update_schedule_summary, user_id)
        except Exception as e:
            logger.error(f"Schedule operation failed for user {user_id}: {e}")

    def _parse_datetime(self, time_string: str, intent: str) -> Optional[datetime.datetime]:
        try:
            parsed_datetime = parser.parse(time_string, fuzzy=True)
            current_time = datetime.datetime.now()
            
            if parsed_datetime.year < current_time.year:
                parsed_datetime = parsed_datetime.replace(year=current_time.year)
            
            if intent == "add" and parsed_datetime < current_time:
                if parsed_datetime.date() == current_time.date():
                    parsed_datetime += datetime.timedelta(days=1)
                
                if parsed_datetime < current_time:
                    return None
            
            return parsed_datetime
        except Exception as e:
            logger.error(f"DateTime parsing failed: {e}")
            return None

    def _update_schedule_summary(self, user_id: str):
        try:
            upcoming_schedules = self.scheduler_service.get_upcoming_schedules_raw(
                user_id, 
                limit=5
            )
            new_summary = self.analysis_service.generate_schedule_summary(upcoming_schedules)
            self.session_manager.update_schedule_summary(user_id, new_summary)
            logger.info(f"Schedule summary updated for user {user_id}")
        except Exception as e:
            logger.error(f"Schedule summary update failed for user {user_id}: {e}")

    def trigger_proactive_message(self, user_id: str, context: str) -> Optional[str]:
        try:
            prompt = (
                f"[SYSTEM INSTRUCTION]\n"
                f"Task: Greet the user regarding the following reminder.\n"
                f"Reminder Context: {context}\n"
                f"Create a natural, friendly, and concise message."
            )
            
            history = self.session_manager.get_session(user_id)
            gemini_history = self.session_manager.prepare_history_for_gemini(
                history,
                self.session_manager.IMG_PREFIX
            )
            
            response = self._generate_response(
                gemini_history, 
                [types.Part(text=prompt)]
            )
            
            if response:
                self.session_manager.update_session(
                    user_id, 
                    "(System Reminder)", 
                    response
                )
                return response
        except Exception as e:
            logger.error(f"Proactive message failed for user {user_id}: {e}")
        
        return None

    def get_system_stats(self) -> Dict:
        intent_stats = self.intent_extractor.get_cache_stats()
        context_stats = self.context_builder.get_cache_stats()
        
        return {
            "sessions": len(self.session_manager.sessions),
            "current_key_index": self.current_key_index,
            "api_health": self.health_monitor.get_status(),
            "worker_queue_size": self.executor._work_queue.qsize(),
            "active_processing": sum(1 for flag in self._processing_flags.values() if flag),
            "intent_extraction": intent_stats,
            "context_builder": context_stats
        }

    def __del__(self):
        try:
            self.executor.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Executor shutdown error: {e}")