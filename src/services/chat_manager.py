import os
import logging
import time
import asyncio
import datetime
import numpy as np
from collections import defaultdict
from typing import List, Dict, Optional, Any
from dataclasses import asdict

import PIL.Image
from dateutil import parser
from google import genai
from google.genai import types

from src.database import DBConnection
from src.config import (
    GOOGLE_API_KEYS,
    TIER_1_MODEL,
    TIER_2_MODEL,
    TIER_3_MODEL,
    CHAT_MODEL,
    METEOSOURCE_API_KEY,
    INSTRUCTION,
    FORMATTING_INSTRUCTION,
    CHAT_INTERACTION_ANALYSIS_INSTRUCTION,
    CHAT_ANALYSIS_INSTUCTION,
    MEMORY_ANALYSIS_INSTRUCTION,
    AVAILABLE_CHAT_MODELS,
    set_chat_model
)

from src.services.handler.canonicalizer import MemoryCanonicalizer
from src.services.handler.memory_gatekeeper import MemoryGatekeeper
from src.services.memory_manager import MemoryManager
from src.services.scheduler_service import SchedulerService
from src.services.embedding import MemoryAnalyzer
from src.services.handler.intent_extractor import IntentExtractor
from src.services.handler.api_key_monitor import APIKeyHealthMonitor
from src.services.handler.session_manager import SessionManager
from src.services.handler.context_builder import ContextBuilder
from src.services.handler.analysis_service import AnalysisService
from src.services.proactive_insight_engine import ProactiveInsightEngine
from src.services.task_planner import TaskPlanner
from src.services.personality_engine import PersonalityEngine
from src.services.system_config_manager import SystemConfigManager
from src.services.user_profile_manager import UserProfileManager

logger = logging.getLogger(__name__)


class ConversationMetrics:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._metrics: Dict[str, Dict] = defaultdict(lambda: {
            'total_messages': 0,
            'avg_response_time': 0.0,
            'sentiment_scores': [],
            'topics': defaultdict(int),
            'last_active': None,
            'engagement_score': 0.0,
            'memory_references': 0,
            'schedule_interactions': 0
        })
    
    async def record_interaction(self, user_id: str, response_time: float, sentiment: float = 0.0, 
                          topics: List[str] = None, has_memory: bool = False, has_schedule: bool = False):
        async with self._lock:
            m = self._metrics[user_id]
            m['total_messages'] += 1
            m['last_active'] = datetime.datetime.now()
            
            total = m['total_messages']
            m['avg_response_time'] = ((m['avg_response_time'] * (total - 1)) + response_time) / total
            
            if sentiment != 0.0:
                m['sentiment_scores'].append(sentiment)
                if len(m['sentiment_scores']) > 50:
                    m['sentiment_scores'].pop(0)
            
            if topics:
                for topic in topics:
                    m['topics'][topic] += 1
            
            if has_memory:
                m['memory_references'] += 1
            if has_schedule:
                m['schedule_interactions'] += 1
            
            m['engagement_score'] = self._calculate_engagement(m)
    
    def _calculate_engagement(self, metrics: Dict) -> float:
        base_score = min(metrics['total_messages'] / 100, 1.0) * 0.3
        
        sentiment_score = 0.0
        if metrics['sentiment_scores']:
            avg_sentiment = sum(metrics['sentiment_scores']) / len(metrics['sentiment_scores'])
            sentiment_score = (avg_sentiment + 1) / 2 * 0.3
        
        feature_score = 0.0
        if metrics['total_messages'] > 0:
            memory_ratio = metrics['memory_references'] / metrics['total_messages']
            schedule_ratio = metrics['schedule_interactions'] / metrics['total_messages']
            feature_score = (memory_ratio + schedule_ratio) / 2 * 0.2
        
        recency_score = 0.2
        if metrics['last_active']:
            hours_since = (datetime.datetime.now() - metrics['last_active']).total_seconds() / 3600
            recency_score = max(0, (1 - hours_since / 168) * 0.2)
        
        return min(base_score + sentiment_score + feature_score + recency_score, 1.0)
    
    def get_user_metrics(self, user_id: str) -> Dict:
        m = self._metrics[user_id]
        return {
            'total_messages': m['total_messages'],
            'avg_response_time': round(m['avg_response_time'], 2),
            'engagement_score': round(m['engagement_score'], 2),
            'top_topics': sorted(m['topics'].items(), key=lambda x: x[1], reverse=True)[:5],
            'avg_sentiment': round(sum(m['sentiment_scores']) / len(m['sentiment_scores']), 2) if m['sentiment_scores'] else 0.0,
            'last_active': m['last_active'].isoformat() if m['last_active'] else None,
            'memory_usage': m['memory_references'],
            'schedule_usage': m['schedule_interactions']
        }
    
    def get_all_metrics(self) -> Dict[str, Dict]:
        return {uid: self.get_user_metrics(uid) for uid in self._metrics.keys()}


class ChatHandler:
    CLEANUP_INTERVAL = 300
    MAX_OUTPUT_TOKENS = 2048
    TEMPERATURE = 0.7
    TOP_P = 0.95
    METRICS_SAVE_INTERVAL = 600
    PROACTIVE_CHECK_INTERVAL = 1800  # 30 minutes
    
    def __init__(self, memory_manager: MemoryManager, analyzer: MemoryAnalyzer, 
                scheduler_service: SchedulerService):
        if not GOOGLE_API_KEYS:
            raise ValueError("No API keys configured")
        
        self.memory_manager = memory_manager
        self.analyzer = analyzer
        self.scheduler_service = scheduler_service
        
        self.personality_engine = PersonalityEngine()
        self.config_manager = SystemConfigManager(memory_manager.db)
        self.profile_manager = UserProfileManager(memory_manager.db)
        
        # Initialize Proactive Insight Engine
        self.proactive_engine = ProactiveInsightEngine(
            memory_manager.db, 
            memory_manager.knowledge_graph if hasattr(memory_manager, 'knowledge_graph') else None
        )
        
        # Initialize Task Planner
        self.task_planner = TaskPlanner(
            memory_manager.db,
            None, # Client set in _initialize_client
            CHAT_MODEL
        )
        
        self.api_keys = GOOGLE_API_KEYS
        self.current_key_index = 0
        self.health_monitor = APIKeyHealthMonitor(len(self.api_keys))
        
        self.client: Optional[genai.Client] = None
        self._initialize_client()
        
        self.chat_model_name = CHAT_MODEL
        
        if hasattr(memory_manager, 'canonicalizer') and memory_manager.canonicalizer is None:
            self.canonicalizer = MemoryCanonicalizer(self.client, TIER_2_MODEL)
            memory_manager.gatekeeper = MemoryGatekeeper(memory_manager.db, memory_manager.emb_handler)
            memory_manager.use_canonicalization = True
        
        self.intent_extractor = IntentExtractor(self.client, TIER_3_MODEL)
        self.session_manager = SessionManager()
        self.context_builder = ContextBuilder(METEOSOURCE_API_KEY)
        self.analysis_service = AnalysisService(self.client, TIER_1_MODEL, TIER_2_MODEL)
        self.metrics = ConversationMetrics()
        
        self._processing_flags: Dict[str, bool] = {}
        self._flag_lock = asyncio.Lock()
        self._last_cleanup = time.time()
        self._last_metrics_save = time.time()
        self._last_proactive_check = time.time()

    def _initialize_client(self):
        try:
            key_index = self.health_monitor.get_healthy_key(self.current_key_index, len(self.api_keys))
            self.current_key_index = key_index if key_index is not None else 0
            self.client = genai.Client(api_key=self.api_keys[self.current_key_index])
            
            if hasattr(self, 'analysis_service'):
                self.analysis_service.client = self.client
            
            # Update planner client
            if hasattr(self, 'task_planner'):
                self.task_planner.client = self.client
                
            logger.info(f"Initialized with API Key #{self.current_key_index}")
        except Exception as e:
            logger.critical(f"Client initialization failed: {e}")
            raise

    def change_model(self, model_name: str):
        if model_name in AVAILABLE_CHAT_MODELS:
            self.chat_model_name = model_name
            set_chat_model(model_name)

    def clear_session(self, user_id: str):
        self.session_manager.clear_session(user_id)
        self.context_builder.clear_cache()

    async def process_message(self, user_id: str, text: str, image_path: str = None, **kwargs) -> str:
        async with self._flag_lock:
            if self._processing_flags.get(user_id):
                return "Sebentar, Vira masih memproses pesan sebelumnya ya."
            self._processing_flags[user_id] = True
            
        start_time = time.time()
        try:
            # Initialize profile
            await self.profile_manager.initialize()
            
            # Handle Telegram Name update if provided
            user_name = kwargs.get('user_name')
            if user_name:
                await self.profile_manager.update_telegram_name(user_id, user_name)

            await self._periodic_cleanup()

            # Canonicalization (Existing)
            if self.memory_manager.use_canonicalization:
                asyncio.create_task(self._process_canonicalization(user_id, text))
            
            response = await self._execute_chat_flow(user_id, text, image_path)
            
            # Record metrics
            duration = time.time() - start_time
            asyncio.create_task(self._record_metrics(user_id, duration, text, response))
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return "Maaf, Vira mengalami gangguan sistem."
        finally:
            async with self._flag_lock:
                self._processing_flags[user_id] = False

    async def _release_processing_lock(self, user_id: str):
        async with self._flag_lock:
            self._processing_flags[user_id] = False

    async def _process_canonicalization(self, user_id: str, text: str):
        """Process message for canonical memory storage."""
        if hasattr(self.memory_manager, 'process_canonical_input'):
            await self.memory_manager.process_canonical_input(user_id, text)

    async def _periodic_cleanup(self):
        current_time = time.time()
        if current_time - self._last_cleanup > self.CLEANUP_INTERVAL:
            asyncio.create_task(self._run_session_cleanup())
            self._last_cleanup = current_time
        
        if current_time - self._last_metrics_save > self.METRICS_SAVE_INTERVAL:
            asyncio.create_task(self._save_metrics_snapshot())
            self._last_metrics_save = current_time
            
        if current_time - self._last_proactive_check > self.PROACTIVE_CHECK_INTERVAL:
            asyncio.create_task(self._check_proactive_insights())
            self._last_proactive_check = current_time

    async def _check_proactive_insights(self):
        """Check for proactive insights for all active users."""
        try:
            active_users = list(self.metrics.get_all_metrics().keys())
            
            for user_id in active_users:
                insight = await self.proactive_engine.get_best_insight(user_id)
                if insight:
                    message_text = await self.proactive_engine.generate_proactive_message(
                        user_id, insight, self.client, self.chat_model_name
                    )
                    
                    if message_text:
                        # In a real app with push notifications, we would send this via push.
                        # For now, we store it as a pending message in session or log it.
                        logger.info(f"[PROACTIVE] Generated for {user_id}: {message_text}")
                        
                        # Mark as delivered
                        await self.proactive_engine.mark_insight_delivered(user_id, insight)
                        
                        # Simulating delivery by adding to session (will appear next time user chats)
                        # Or if using WebSocket, could push directly.
                        self.session_manager.update_session(user_id, "(System Proactive)", message_text)
                        
        except Exception as e:
            logger.error(f"[PROACTIVE] Check failed: {e}")

    async def _run_session_cleanup(self):
        self.session_manager.cleanup_inactive_sessions()

    async def _save_metrics_snapshot(self):
        try:
            all_metrics = self.metrics.get_all_metrics()
            logger.info(f"Metrics snapshot: {len(all_metrics)} active users")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    async def _record_metrics(self, user_id: str, response_time: float, user_text: str, response: str):
        try:
            topics = self._extract_topics(user_text + " " + response)
            sentiment = self._estimate_sentiment(response)
            has_memory = "mengingat" in response.lower() or "ingat" in response.lower()
            has_schedule = "jadwal" in response.lower() or "reminder" in response.lower()
            
            await self.metrics.record_interaction(
                user_id, response_time, sentiment, topics, has_memory, has_schedule
            )
            
            # Also record for proactive engine
            await self.proactive_engine.record_activity(
                user_id, "interaction", 
                {"sentiment": sentiment, "has_memory": has_memory}
            )
        except Exception as e:
            logger.error(f"Metrics recording failed: {e}")

    def _extract_topics(self, text: str) -> List[str]:
        topics = []
        keywords = {
            'cuaca': ['cuaca', 'hujan', 'panas', 'dingin'],
            'jadwal': ['jadwal', 'meeting', 'reminder', 'acara'],
            'memori': ['ingat', 'lupa', 'mengingat', 'memory'],
            'pribadi': ['saya', 'aku', 'keluarga', 'teman'],
            'pekerjaan': ['kerja', 'kantor', 'tugas', 'project'],
            'kesehatan': ['sakit', 'dokter', 'obat', 'sehat']
        }
        
        text_lower = text.lower()
        for topic, words in keywords.items():
            if any(word in text_lower for word in words):
                topics.append(topic)
        
        return topics

    def _estimate_sentiment(self, text: str) -> float:
        positive_words = ['senang', 'bahagia', 'terima kasih', 'bagus', 'hebat', 'mantap', 'suka']
        negative_words = ['sedih', 'marah', 'kecewa', 'buruk', 'tidak suka', 'benci']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count + neg_count == 0:
            return 0.0
        
        return (pos_count - neg_count) / (pos_count + neg_count)

    async def _execute_chat_flow(self, user_id: str, user_text: str, image_path: str) -> str:
        # 1. Check for active plan first
        active_plan = await self.task_planner.get_active_plan(user_id)
        if active_plan:
            return await self._handle_active_plan(user_id, active_plan, user_text)

        # 2. Regular flow
        intent_data = self.intent_extractor.extract(user_text)
        
        # Check if new plan is requested (explicit keyword or complex action)
        if self._should_create_plan(user_text, intent_data):
            plan = await self.task_planner.create_plan(user_id, user_text, intent_data)
            if plan:
                return await self._handle_active_plan(user_id, plan, "start")

        async with asyncio.TaskGroup() as group:
            session_task = group.create_task(self._gather_session_data(user_id))
            schedule_task = group.create_task(self._process_pending_schedule(user_id))
            profile_task = group.create_task(self.profile_manager.get_profile(user_id))
            
            memory_task = None
            if intent_data.get('needs_memory', False):
                memory_task = group.create_task(
                    self._retrieve_relevant_memories_with_intent(user_id, user_text, intent_data)
                )
            
        session_data = session_task.result()
        schedule_context = schedule_task.result()
        relevant_memories = memory_task.result() if memory_task else []
        profile_obj = profile_task.result()
        user_profile = asdict(profile_obj) if profile_obj else None

        system_context = self.context_builder.build_context(
            user_id,
            session_data['summary'],
            relevant_memories,
            session_data['memory_summary'],
            session_data['last_interaction'],
            schedule_context,
            session_data['schedule_summary'],
            intent_context=intent_data,
            user_metrics=self.metrics.get_user_metrics(user_id),
            user_profile=user_profile
        )
        
        # Apply Personality Adaptation
        emotion = intent_data.get('detected_emotion') if intent_data else None
        self.personality_engine.adjust_personality(user_id, emotion)
        personality_instruction = self.personality_engine.get_system_instruction_modifier(user_id)
        
        # Apply System Config Override (Persona Hot-Swap)
        if hasattr(self, 'config_manager'):
            await self.config_manager.initialize()
            base_instruction = await self.config_manager.get_active_instruction()
            # If base instruction is different from config.INSTRUCTION, it replaces the core identity.
            # But context builder might be adding its own preamble. 
            # We need to see how context_builder uses instructions. 
            # Assuming context_builder builds "Context", and we prepend the Instruction.
        else:
            base_instruction = INSTRUCTION # Fallback to default imports
        
        # Combine:
        # 1. Base Instruction (Global Identity/Persona) - Overridable via DB
        # 2. System Context (Session, Memory, Schedule) - From ContextBuilder
        # 3. Personality Modifier (Ephemeral Tone) - From PersonalityEngine
        # 4. Formatting Rules (HTML Mandate)
        
        full_system_prompt = f"{base_instruction}\n\n{system_context}\n\n{personality_instruction}\n\n{FORMATTING_INSTRUCTION}"
        
        chat_input = self._build_chat_input(full_system_prompt, user_text, image_path)
        gemini_history = self.session_manager.prepare_history_for_gemini(
            session_data['history'],
            self.session_manager.IMG_PREFIX
        )
        
        response_text = await self._generate_response_with_fallback(gemini_history, chat_input)
        
        # Chance to add a prefix (e.g. "Senang mendengarnya!") directly to response
        # But maybe safer to let LLM handle it via instruction. 
        # Alternatively, self.personality_engine.get_response_prefix(user_id) could be prepended if model doesn't follow tone well.
        # For now, let's trust the adaptive instruction.
        
        if response_text:
            asyncio.create_task(self._post_process_response(user_id, user_text, response_text, image_path))
        
        return response_text or "❌ Maaf, saya tidak dapat menghasilkan respon saat ini."

    def _should_create_plan(self, text: str, intent_data: Dict) -> bool:
        """Determine if a new plan should be created."""
        text_lower = text.lower()
        keywords = ['buat rencana', 'buatkan plan', 'susun strategi', 'break down', 'tahapan']
        if any(k in text_lower for k in keywords):
            return True
        return False

    async def _handle_active_plan(self, user_id: str, plan, user_input: str) -> str:
        """
        Handle interaction with sophisticated context switching.
        Check if user input is relevant to the plan or an off-topic interruption.
        """
        # 1. Check for cancellation
        if user_input.lower() in ['batal', 'cancel', 'stop', 'batalkan']:
            await self.task_planner.cancel_plan(plan.id, "user_request")
            return "✅ Rencana dibatalkan. Ada yang lain yang bisa saya bantu?"

        # 2. Get Step Info
        current_step = next((s for s in plan.steps if s.status.value == 'pending'), None)
        if not current_step:
            return "🎉 Rencana sudah selesai! Ada lagi?"

        # 3. Analyze Intent
        intent_data = self.intent_extractor.extract(user_input)
        is_direct_command = user_input.lower() in ["lanjut", "start", "next", "ok"]
        
        # 4. Determine Context Relevance (Heuristic)
        # Relevant if:
        # - Direct command ("lanjut")
        # - Intent matches step expectation (e.g. step="ask budget", intent="information")
        # - Keywords overlap
        is_relevant = is_direct_command
        
        if not is_relevant:
            # Check keyword overlap between step description and user input (naive check)
            step_keywords = set(current_step.description.lower().split())
            user_words = set(user_input.lower().split())
            if len(step_keywords.intersection(user_words)) > 0:
                is_relevant = True
            
            # If step expects input and user provides info
            if "ask" in current_step.action_type and intent_data.get('intent_type') in ['information', 'opinion']:
                is_relevant = True

        # 5. Handle Interruption (Context Switch)
        if not is_relevant and len(user_input.split()) > 1: # Ignore single word generic inputs
            # Process as normal chat (Interruption)
            # We treat this as a separate turn WITHOUT the plan context enforcing focus
            
            # Reuse Session/Profile gathering? Or simplistic call?
            # Simplistic call to generate response
            # We create a temporary context just for this reply
            
            interruption_prompt = (
                f"You are interrupted while executing a plan: '{plan.goal}'.\n"
                f"Current step: {current_step.description}.\n"
                f"User asks: '{user_input}'.\n"
                f"Answer the user's question naturally, then politely remind them to return to the plan.\n"
                f"Use HTML formatting."
            )
            
            # Simple Generation
            chat_input = [types.Part(text=interruption_prompt)]
            history = self.session_manager.get_session(user_id) # Use full history
            gemini_history = self.session_manager.prepare_history_for_gemini(history, self.session_manager.IMG_PREFIX)
            
            response = await self._generate_response_with_fallback(gemini_history, chat_input)
            if response:
                return response
            
            # Fallback if generation fails
            return f"Maaf, bisa diulang? (Sedang fokus pada rencana: {plan.goal})"

        # 6. Execute Step (If relevant or explicit continue)
        if is_relevant or is_direct_command:
             # Simulation execution
             await self.task_planner.mark_step_complete(current_step.id, f"User Input: {user_input}")
             
             # Check next step
             next_step = next((s for s in plan.steps if s.status.value == 'pending' and s.id != current_step.id), None)
             
             msg = f"✅ <b>{current_step.description}</b> selesai.\n"
             if next_step:
                 msg += f"\nLanjut ke: <b>{next_step.description}</b>\nKetik 'lanjut' atau beri masukan."
             else:
                 msg += "\n🎉 <b>Semua langkah selesai!</b>"
             return msg
        
        # 7. Default Status View (if input was ambiguous)
        steps_str = "\n".join([
            f"{'✅' if s.status.value=='completed' else '⬜'} {s.description}" 
            for s in plan.steps
        ])
        return f"📋 <b>Rencana Aktif: {plan.goal}</b>\n\n{steps_str}\n\nKetik 'lanjut' untuk terus, atau tanya hal lain (saya akan jawab lalu kembali ke sini)."

    async def _retrieve_relevant_memories_with_intent(self, user_id: str, user_text: str, intent_data: Dict) -> List[Dict]:
        search_query = intent_data.get('search_query', user_text)
        entities = intent_data.get('entities', [])
        
        try:
            query_embedding = self.analyzer.get_embedding(search_query)
            if query_embedding is None:
                return []
            
            embedding_list = query_embedding.tolist() if isinstance(query_embedding, np.ndarray) else query_embedding
            
            return await self.memory_manager.get_relevant_memories(
                user_id=user_id,
                query_embedding=embedding_list,
                max_results=5,
                use_clusters=True,
                query_text=search_query,
                entities=entities
            )
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

    async def _gather_session_data(self, user_id: str) -> Dict:
        return {
            'history': self.session_manager.get_session(user_id),
            'summary': self.session_manager.get_metadata(user_id, "summary", ""),
            'memory_summary': self.session_manager.get_metadata(user_id, "memory_summary", ""),
            'last_interaction': self.session_manager.get_metadata(user_id, "last_interaction"),
            'schedule_summary': self.session_manager.get_schedule_summary(user_id)
        }

    async def _process_pending_schedule(self, user_id: str) -> Optional[str]:
        pending_schedule = await self.scheduler_service.get_pending_trigger(user_id)
        if not pending_schedule:
            return None
        await self.scheduler_service.mark_as_executed(pending_schedule['id'], note="Triggered by chat")
        return pending_schedule.get('context')

    def _build_chat_input(self, system_context: str, user_text: str, image_path: Optional[str]) -> List:
        full_prompt = f"{system_context}\n\n[USER MESSAGE]\n{user_text}"
        chat_input = [types.Part(text=full_prompt)]
        
        if image_path and os.path.exists(image_path):
            try:
                image = PIL.Image.open(image_path)
                chat_input.append(image)
            except Exception:
                pass
        
        return chat_input

    async def _generate_response_with_fallback(self, history: List, chat_input: List) -> str:
        # Use GLOBAL_GEN_CONFIG but allow overriding system instruction tailored for this chat
        # Actually generate_content accepts 'config' which overrides. 
        # But types.GenerateContentConfig is an object.
        # We should create a new config deriving from global if we want specific instruction?
        # GLOBAL_GEN_CONFIG already has INSTRUCTION. 
        # But we build full_system_prompt dynamically in _execute_chat_flow and pass it as user message?
        # Let's check `_build_chat_input`.
        # Code view step 1331: full_prompt = f"{system_context}\n\n[USER MESSAGE]\n{user_text}"
        # chat_input = [types.Part(text=full_prompt)]
        
        # So system context is prepended to user message. 
        # The `system_instruction` param in config is just the BASE identity.
        # So we can use GLOBAL_GEN_CONFIG as base.
        
        from src.config import GLOBAL_GEN_CONFIG
        config = GLOBAL_GEN_CONFIG

        models_to_try = []
        if self.chat_model_name:
            models_to_try.append(self.chat_model_name)
        
        for model in AVAILABLE_CHAT_MODELS:
            if model not in models_to_try:
                models_to_try.append(model)

        for model_name in models_to_try:
            attempts = 2 if model_name == self.chat_model_name else 1
            
            for attempt in range(attempts):
                try:
                    chat = self.client.chats.create(model=model_name, history=history, config=config)
                    response = chat.send_message(chat_input)
                    if response.text:
                        if model_name != self.chat_model_name:
                            logger.info(f"Fallback to model {model_name} successful")
                            self.health_monitor.mark_success(self.current_key_index)
                        return response.text
                except Exception as e:
                    logger.warning(f"Generation failed on {model_name} (Attempt {attempt+1}/{attempts}): {e}")
                    if "429" in str(e) or "quota" in str(e).lower():
                        self.health_monitor.mark_failure(self.current_key_index)
                        self._initialize_client()
                    await asyncio.sleep(0.5)
        
        return ""

    async def _post_process_response(self, user_id: str, user_text: str, response_text: str, image_path: Optional[str]):
        self.session_manager.update_session(
            user_id, user_text, response_text, image_path,
            lambda uid: asyncio.create_task(self._analyze_chat_session(uid))
        )
        asyncio.create_task(self._analyze_interaction(user_id, user_text, response_text))

    async def _analyze_chat_session(self, user_id: str):
        try:
            history = list(self.session_manager.get_session(user_id))
            current_summary = self.session_manager.get_summary(user_id)
            new_summary = self.analysis_service.run_chat_analysis(history, current_summary, CHAT_ANALYSIS_INSTUCTION)
            if new_summary and new_summary != current_summary:
                self.session_manager.update_summary(user_id, new_summary)
                self.session_manager.delete_last_n_messages(user_id, 8)
        except Exception:
            pass

    async def _analyze_interaction(self, user_id: str, user_text: str, ai_text: str):
        try:
            analysis_data = self.analysis_service.run_interaction_analysis(
                user_text, ai_text, CHAT_INTERACTION_ANALYSIS_INSTRUCTION
            )
            if not analysis_data: 
                return
            
            for item in analysis_data.get("memory", []):
                if isinstance(item, dict) and item.get("should_store"):
                    await self._handle_memory_operation(user_id, item)
            
            for item in analysis_data.get("schedules", []):
                if isinstance(item, dict) and item.get("should_schedule"):
                    await self._handle_schedule_operation(user_id, item)
                    
        except Exception as e:
            logger.error(f"Analysis failed: {e}")

    async def _handle_memory_operation(self, user_id: str, memory_data: Dict):
        summary = memory_data.get("summary")
        if not summary: 
            return
        
        try:
            emb = self.analyzer.get_embedding(summary)
            emb_list = emb.tolist() if isinstance(emb, np.ndarray) else emb
            
            if memory_data.get("action") == "forget":
                await self.memory_manager.forget_memory(user_id, summary, emb_list)
            else:
                await self.memory_manager.add_memory(
                    user_id, summary, 
                    memory_data.get("type", "general"),
                    memory_data.get("priority", 0.5), 
                    emb_list
                )
                asyncio.create_task(self._consolidate_memories(user_id))
        except Exception as e:
            logger.error(f"Memory op failed: {e}")

    async def _consolidate_memories(self, user_id: str):
        try:
            stats = await self.memory_manager.get_memory_stats(user_id)
            if not self.session_manager.should_run_memory_analysis(user_id, stats.get('active', 0)):
                return
            
            recent = await self.memory_manager.search_memories(user_id, "", limit=20)
            current_sum = self.session_manager.get_memory_summary(user_id)
            new_sum = self.analysis_service.run_memory_analysis(recent, current_sum, MEMORY_ANALYSIS_INSTRUCTION)
            
            if new_sum:
                self.session_manager.update_memory_summary(user_id, new_sum)
                self.session_manager.mark_memory_analysis_done(user_id, stats.get('active', 0))
        except Exception:
            pass

    async def _handle_schedule_operation(self, user_id: str, schedule_data: Dict):
        try:
            time_str = schedule_data.get("time_str")
            if not time_str: 
                return
            
            trigger_time = self._parse_datetime(time_str, schedule_data.get("intent", "add"))
            if not trigger_time: 
                return
            
            recurring = schedule_data.get("recurring")
            
            if schedule_data.get("intent") == "cancel":
                await self.scheduler_service.cancel_schedule_by_context(user_id, trigger_time, schedule_data.get("context"))
            else:
                await self.scheduler_service.add_schedule(
                    user_id, trigger_time, schedule_data.get("context"), 
                    schedule_data.get("priority", 0),
                    recurring=recurring
                )
            
            asyncio.create_task(self.refresh_schedule_summary(user_id))
        except Exception:
            pass

    async def refresh_schedule_summary(self, user_id: str):
        try:
            upcoming = await self.scheduler_service.get_upcoming_schedules_raw(user_id, limit=5)
            new_summary = self.analysis_service.generate_schedule_summary(upcoming)
            self.session_manager.update_schedule_summary(user_id, new_summary)
            logger.info(f"Schedule summary updated for {user_id}")
        except Exception as e:
            logger.error(f"Schedule summary update failed: {e}")

    def _parse_datetime(self, time_string: str, intent: str) -> Optional[datetime.datetime]:
        try:
            dt = parser.parse(time_string, fuzzy=True)
            now = datetime.datetime.now()
            if dt.year < now.year: 
                dt = dt.replace(year=now.year)
            if intent == "add" and dt < now:
                if dt.date() == now.date(): 
                    dt += datetime.timedelta(days=1)
                if dt < now: 
                    return None
            return dt
        except Exception:
            return None

    async def trigger_proactive_message(self, user_id: str, context: str) -> Optional[str]:
        try:
            prompt = f"Greet the user regarding this reminder: {context}. Be natural and concise."
            history = self.session_manager.get_session(user_id)
            gemini_history = self.session_manager.prepare_history_for_gemini(history, self.session_manager.IMG_PREFIX)
            
            response = await self._generate_response_with_fallback(gemini_history, [types.Part(text=prompt)])
            if response:
                self.session_manager.update_session(user_id, "(System Reminder)", response)
                return response
        except Exception:
            pass
        return None

    async def get_user_insights(self, user_id: str) -> Dict:
        try:
            metrics = self.metrics.get_user_metrics(user_id)
            memory_stats = await self.memory_manager.get_memory_stats(user_id)
            schedule_stats = await self.scheduler_service.get_schedule_stats(user_id)
            schedule_performance = self.scheduler_service.analytics.get_user_analytics(user_id)
            memory_quality = await self.memory_manager.quality_monitor.get_system_health(user_id)
            
            intent_analytics = self.intent_extractor.get_extraction_analytics()
            context_stats = self.context_builder.get_cache_stats()
            gatekeeper_stats = self.memory_manager.gatekeeper.get_gatekeeper_stats() if self.memory_manager.gatekeeper else {}
            
            return {
                'metrics': metrics,
                'memory_stats': memory_stats,
                'memory_quality': memory_quality,
                'schedule_stats': schedule_stats,
                'schedule_performance': schedule_performance,
                'session_active': user_id in self.session_manager.sessions,
                'intent_analytics': intent_analytics,
                'context_cache': context_stats,
                'gatekeeper_performance': gatekeeper_stats
            }
        except Exception as e:
            logger.error(f"Failed to get user insights: {e}")
            return {}

    def get_system_stats(self) -> Dict:
        return {
            "sessions": len(self.session_manager.sessions),
            "api_health": self.health_monitor.get_status(),
            "active_processing": sum(1 for flag in self._processing_flags.values() if flag),
            "total_users_tracked": len(self.metrics.get_all_metrics()),
            "intent_cache_stats": self.intent_extractor.get_cache_stats(),
            "context_optimizer": self.context_builder.get_cache_stats()
        }

    async def export_user_report(self, user_id: str) -> str:
        try:
            insights = await self.get_user_insights(user_id)
            
            report = f"""
=== COMPREHENSIVE USER REPORT: {user_id} ===

CONVERSATION METRICS:
- Total Messages: {insights['metrics']['total_messages']}
- Avg Response Time: {insights['metrics']['avg_response_time']}s
- Engagement Score: {insights['metrics']['engagement_score']}/1.0
- Avg Sentiment: {insights['metrics']['avg_sentiment']}
- Last Active: {insights['metrics']['last_active']}

TOP TOPICS:
{chr(10).join(f'  - {topic}: {count}' for topic, count in insights['metrics']['top_topics'])}

MEMORY USAGE:
- Active Memories: {insights['memory_stats'].get('active', 0)}
- Total References: {insights['metrics']['memory_usage']}

SCHEDULE USAGE:
- Pending: {insights['schedule_stats'].get('pending', 0)}
- Executed: {insights['schedule_stats'].get('executed', 0)}
- Recurring: {insights['schedule_stats'].get('recurring', 0)}
- Total Interactions: {insights['metrics']['schedule_usage']}

SCHEDULE PERFORMANCE:
- Total Completed: {insights.get('schedule_performance', {}).get('total_completed', 0)}
- On-Time Rate: {insights.get('schedule_performance', {}).get('on_time_rate', 0)}%
- Avg Delay: {insights.get('schedule_performance', {}).get('avg_delay_minutes', 0)} min

INTENT EXTRACTION:
- Total Extractions: {insights.get('intent_analytics', {}).get('total_extractions', 0)}
- Fallback Rate: {insights.get('intent_analytics', {}).get('fallback_rate', 0)}%
- Avg Confidence: {insights.get('intent_analytics', {}).get('avg_confidence', 0)}

CONTEXT BUILDER:
- Cache Hit Rate: {insights.get('context_cache', {}).get('cache_hit_rate', 0)}%
- Cache Size: {insights.get('context_cache', {}).get('size', 0)}

SESSION STATUS: {'Active' if insights['session_active'] else 'Inactive'}
"""
            return report.strip()
        except Exception as e:
            return f"Error generating report: {e}"