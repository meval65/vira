import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
from collections import defaultdict

from src.brain.hippocampus.models import AdminProfile, Memory, Triple
from src.brain.hippocampus.memory_operations import MemoryOperations
from src.brain.hippocampus.knowledge_graph import KnowledgeGraph
from src.brain.hippocampus.persona_manager import PersonaManager
# from src.brain.hippocampus.memory_maintenance import MemoryMaintenance
from src.brain.hippocampus.canonicalizer import Canonicalizer
from src.brain.hippocampus.insights_manager import InsightsManager
from src.brain.hippocampus.schedule_manager import ScheduleManager
from src.brain.hippocampus.visual_memory import VisualMemory
from src.brain.hippocampus.storage_utils import StorageUtils

from src.brain.db.mongo_client import get_mongo_client, MongoDBClient
from src.brain.brainstem import MAX_RETRIEVED_MEMORIES, MIN_RELEVANCE_SCORE


class Hippocampus:
    
    SIMILARITY_THRESHOLD: float = 0.90
    ARCHIVE_CHECK_DAYS: int = 90
    MIN_CONFIDENCE: float = 0.3
    
    COMPRESSION_THRESHOLD: int = 5
    TOP_MEMORIES_COUNT: int = 30
    
    WEIGHT_RECENCY: float = 0.3
    WEIGHT_FREQUENCY: float = 0.3
    WEIGHT_PRIORITY: float = 0.4
    
    BATCH_SIZE: int = 100
    CACHE_SIZE: int = 1000

    def __init__(self):
        self._mongo: Optional[MongoDBClient] = None
        self._admin_profile: AdminProfile = AdminProfile()
        self._openrouter = None
        self._evolution_history: Dict[str, List[Dict]] = defaultdict(list)
        self._conflict_history: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = asyncio.Lock()
        self._brain = None
        
        self._memory_ops: Optional[MemoryOperations] = None
        self._knowledge_graph: Optional[KnowledgeGraph] = None
        self._persona_manager: Optional[PersonaManager] = None
        # self._memory_maintenance: Optional[MemoryMaintenance] = None
        self._canonicalizer: Optional[Canonicalizer] = None
        self._insights_manager: Optional[InsightsManager] = None
        self._schedule_manager: Optional[ScheduleManager] = None
        self._visual_memory: Optional[VisualMemory] = None
        self._storage_utils: Optional[StorageUtils] = None

    def bind_brain(self, brain) -> None:
        self._brain = brain
        if brain:
            self._openrouter = brain.openrouter

    async def initialize(self) -> None:
        self._mongo = get_mongo_client()
        await self._mongo.connect()
        
        self._canonicalizer = Canonicalizer(self._openrouter)
        self._memory_ops = MemoryOperations(self._mongo, self._openrouter)
        self._knowledge_graph = KnowledgeGraph(self._mongo)
        self._persona_manager = PersonaManager(self._mongo)
        # self._memory_maintenance = MemoryMaintenance(self._mongo, self._openrouter)
        self._insights_manager = InsightsManager(self._mongo, self._openrouter)
        self._schedule_manager = ScheduleManager(self._mongo)
        self._visual_memory = VisualMemory(self._mongo)
        self._storage_utils = StorageUtils(self._mongo)
        
        await self._load_admin_profile()
        await self._persona_manager._ensure_default_persona()
        await self._create_indexes()

    async def close(self) -> None:
        if self._mongo:
            await self._mongo.close()
            self._mongo = None

    @property
    def admin_profile(self) -> AdminProfile:
        return self._admin_profile

    async def _create_indexes(self) -> None:
        try:
            await self._mongo.memories.create_index([("fingerprint", 1), ("status", 1)])
            await self._mongo.memories.create_index([("status", 1), ("priority", -1), ("last_used", -1)])
            await self._mongo.memories.create_index([("created_at", -1)])
            await self._mongo.memories.create_index([("type", 1)])
            await self._mongo.memories.create_index([("tags", 1)])
            await self._mongo.knowledge_graph.create_index([("subject", 1)])
            await self._mongo.knowledge_graph.create_index([("confidence", 1), ("last_accessed", 1)])
            await self._mongo.entities.create_index([("name", 1)], unique=True)
        except Exception:
            pass

    async def _load_admin_profile(self) -> None:
        doc = await self._mongo.admin_profile.find_one({"_id": "admin"})
        if doc:
            self._admin_profile = AdminProfile(
                telegram_name=doc.get("telegram_name"),
                full_name=doc.get("full_name"),
                additional_info=doc.get("additional_info"),
                preferences=doc.get("preferences", {}),
                last_updated=doc.get("last_updated", datetime.now())
            )

    async def update_admin_profile(
        self,
        telegram_name: Optional[str] = None,
        full_name: Optional[str] = None,
        additional_info: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> None:
        async with self._lock:
            update_fields = {"last_updated": datetime.now()}
            if telegram_name is not None:
                update_fields["telegram_name"] = telegram_name
                self._admin_profile.telegram_name = telegram_name
            if full_name is not None:
                update_fields["full_name"] = full_name
                self._admin_profile.full_name = full_name
            if additional_info is not None:
                update_fields["additional_info"] = additional_info
                self._admin_profile.additional_info = additional_info
            if preferences is not None:
                update_fields["preferences"] = preferences
                self._admin_profile.preferences = preferences
            
            self._admin_profile.last_updated = update_fields["last_updated"]
            
            await self._mongo.admin_profile.update_one(
                {"_id": "admin"},
                {"$set": update_fields},
                upsert=True
            )

    async def store(
        self,
        summary: str,
        memory_type: str,
        priority: float = 0.5,
        embedding: Optional[List[float]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        return await self._memory_ops.store(
            summary, memory_type, priority, embedding, tags, self._canonicalizer
        )

    async def recall(
        self,
        query: str,
        limit: int = MAX_RETRIEVED_MEMORIES,
        query_embedding: Optional[List[float]] = None,
        memory_type: Optional[str] = None
    ) -> List[Memory]:
        return await self._memory_ops.recall(query, limit, query_embedding, memory_type)

    async def delete_memory(self, memory_id: str, hard_delete: bool = False) -> bool:
        return await self._memory_ops.delete_memory(memory_id, hard_delete)
    
    async def list_memories(
        self,
        limit: int = 50,
        skip: int = 0,
        status: Optional[str] = None,
        memory_type: Optional[str] = None
    ) -> List[Dict]:
        return await self._memory_ops.list_memories(limit, skip, status, memory_type)
    
    async def update_memory(self, memory_id: str, updates: Dict) -> bool:
        return await self._memory_ops.update_memory(memory_id, updates)

    async def get_memory_by_id(self, memory_id: str) -> Optional[Memory]:
        return await self._memory_ops.get_memory_by_id(memory_id)

    async def update_memory_tags(self, memory_id: str, tags: List[str]) -> bool:
        return await self._memory_ops.update_memory_tags(memory_id, tags)

    async def store_visual_memory(
        self,
        image_description: str,
        image_path: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        additional_context: Optional[str] = None,
        priority: float = 0.6
    ) -> str:
        return await self._visual_memory.store_visual_memory(
            image_description, image_path, embedding, additional_context, priority
        )
    
    async def search_by_tag(
        self,
        tag: str,
        limit: int = 10,
        query_embedding: Optional[List[float]] = None
    ) -> List[Memory]:
        return await self._visual_memory.search_by_tag(
            tag, limit, query_embedding, self._memory_ops._cosine_similarity
        )
    
    async def search_visual_memories(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        limit: int = 5
    ) -> List[Memory]:
        return await self._visual_memory.search_visual_memories(
            query, query_embedding, limit, self._memory_ops._cosine_similarity
        )

    async def search_memories_by_tags(self, tags: List[str], limit: int = 50) -> List[Memory]:
        return await self._visual_memory.search_memories_by_tags(tags, limit)

    async def archive_old_memories(self, days_threshold: int = ARCHIVE_CHECK_DAYS) -> int:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=days_threshold)
        result = await self._mongo.memories.update_many(
            {"status": "active", "last_used": {"$lt": cutoff}},
            {"$set": {"status": "archived"}}
        )
        return result.modified_count

    async def compress_memories(self) -> int:
        return 0

    async def check_and_compress_memories(self) -> bool:
        return False

    async def apply_memory_decay(self, batch_size: int = None) -> int:
        return 0

    async def consolidate_memories(self) -> int:
        return 0

    async def optimize_knowledge_graph(self) -> None:
        pass

    async def get_memory_stats(self) -> Dict[str, int]:
        active = await self._mongo.memories.count_documents({"status": "active"})
        archived = await self._mongo.memories.count_documents({"status": "archived"})
        total = await self._mongo.memories.count_documents({})
        return {"active": active, "archived": archived, "total": total}

    async def run_maintenance(self) -> Dict[str, Any]:
        archived = await self.archive_old_memories()
        return {"archived": archived, "status": "completed"}

    async def store_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source_memory_id: Optional[str] = None
    ) -> str:
        return await self._knowledge_graph.store_triple(
            subject, predicate, obj, confidence, source_memory_id
        )

    async def query_triples(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        obj: Optional[str] = None,
        min_confidence: float = MIN_CONFIDENCE
    ) -> List[Triple]:
        return await self._knowledge_graph.query_triples(
            subject, predicate, obj, min_confidence
        )
    
    async def add_triple(self, subject: str, predicate: str, obj: str, confidence: float = 0.8, source_memory_id: str = None) -> str:
        return await self.store_triple(subject, predicate, obj, confidence, source_memory_id)

    async def list_triples(self, limit: int = 100, skip: int = 0) -> List[Dict]:
        return await self._knowledge_graph.list_triples(limit, skip)

    async def update_triple(self, triple_id: str, updates: Dict) -> bool:
        return await self._knowledge_graph.update_triple(triple_id, updates)

    async def delete_triple(self, triple_id: str) -> bool:
        return await self._knowledge_graph.delete_triple(triple_id)

    async def get_entity_relations(self, entity: str, limit: int = 50) -> List[Triple]:
        return await self._knowledge_graph.get_entity_relations(entity, limit)

    async def infer_knowledge(self, entity: str) -> Dict[str, List[str]]:
        return await self._knowledge_graph.infer_knowledge(entity)

    async def find_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 3
    ) -> Optional[List[Triple]]:
        return await self._knowledge_graph.find_path(start_entity, end_entity, max_depth)

    async def create_persona(
        self,
        name: str,
        description: str,
        traits: Dict[str, float],
        voice_tone: str = "friendly",
        identity_anchor: Optional[str] = None
    ) -> str:
        return await self._persona_manager.create_persona(
            name, description, traits, voice_tone, identity_anchor
        )

    async def update_persona(
        self,
        persona_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        traits: Optional[Dict[str, float]] = None,
        voice_tone: Optional[str] = None
    ) -> bool:
        return await self._persona_manager.update_persona(
            persona_id, name, description, traits, voice_tone
        )

    async def delete_persona(self, persona_id: str) -> bool:
        return await self._persona_manager.delete_persona(persona_id)

    async def list_personas(self) -> List[Dict]:
        return await self._persona_manager.list_personas()
    
    async def get_personas(self) -> List[Dict]:
        return await self.list_personas()

    async def get_active_persona(self) -> Optional[Dict]:
        return await self._persona_manager.get_active_persona()

    async def switch_persona(self, persona_id: str) -> bool:
        return await self._persona_manager.switch_persona(persona_id)

    async def get_persona_calibration(self, persona_id: str) -> Optional[Dict]:
        return await self._persona_manager.get_persona_calibration(persona_id)

    async def update_persona_calibration(
        self,
        persona_id: str,
        emotional_inertia: Optional[float] = None,
        base_arousal: Optional[float] = None,
        base_valence: Optional[float] = None,
        calibration_status: Optional[bool] = None,
        identity_anchor: Optional[str] = None
    ) -> bool:
        return await self._persona_manager.update_persona_calibration(
            persona_id, emotional_inertia, base_arousal, base_valence, 
            calibration_status, identity_anchor
        )

    async def save_emotional_state(
        self,
        mood: str,
        empathy: float,
        satisfaction: float,
        mood_history: List[Dict]
    ) -> None:
        return await self._storage_utils.save_emotional_state(
            mood, empathy, satisfaction, mood_history
        )

    async def load_emotional_state(self) -> Optional[Dict]:
        return await self._storage_utils.load_emotional_state()

    async def get_global_context(self) -> Optional[str]:
        return await self._storage_utils.get_global_context()

    async def query_entity(self, entity_name: str) -> Dict[str, Any]:
        return await self._storage_utils.query_entity(
            entity_name, self._knowledge_graph.get_entity_relations
        )

    async def add_schedule(
        self,
        trigger_time: datetime,
        context: str,
        priority: int = 0,
        repeat: Optional[str] = None
    ) -> str:
        return await self._schedule_manager.add_schedule(
            trigger_time, context, priority, repeat
        )

    async def update_schedule(self, schedule_id: str, updates: Dict) -> bool:
        return await self._schedule_manager.update_schedule(schedule_id, updates)

    async def delete_schedule(self, schedule_id: str) -> bool:
        return await self._schedule_manager.delete_schedule(schedule_id)

    async def get_pending_schedules(self, limit: int = 10) -> List[Dict]:
        return await self._schedule_manager.get_pending_schedules(limit)

    async def get_upcoming_schedules(self, hours_ahead: int = 24) -> List[Dict]:
        return await self._schedule_manager.get_upcoming_schedules(hours_ahead)

    async def store_tool_output(
        self,
        tool_name: str,
        output_type: str,
        content: str,
        file_path: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        return await self._storage_utils.store_tool_output(
            tool_name, output_type, content, file_path, metadata
        )

    async def get_tool_output(self, output_id: str) -> Optional[Dict]:
        return await self._storage_utils.get_tool_output(output_id)

    async def get_recent_tool_outputs(
        self,
        tool_name: Optional[str] = None,
        output_type: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict]:
        return await self._storage_utils.get_recent_tool_outputs(
            tool_name, output_type, limit
        )

    async def get_recent_chat_history(self, limit: int = 25) -> List[Dict]:
        return await self._storage_utils.get_recent_chat_history(limit)

    async def get_related_memories(self, memory_id: str, limit: int = 10) -> List[Memory]:
        return await self._storage_utils.get_related_memories(
            memory_id, self._memory_ops.get_memory_by_id, 
            self._memory_ops._cosine_similarity
        )

    async def generate_insight(self, chat_summary: str) -> Optional[Dict]:
        return await self._insights_manager.generate_insight(
            chat_summary, self.recall
        )

    async def store_insight(self, insight_data: Dict) -> str:
        return await self._insights_manager.store_insight(insight_data)

    async def get_relevant_insights(self, query: str, limit: int = 3) -> List[Dict]:
        return await self._insights_manager.get_relevant_insights(query, limit)

    async def mark_insight_used(self, insight_id: str) -> bool:
        return await self._insights_manager.mark_insight_used(insight_id)

    async def run_daydream_cycle(self) -> Optional[Dict]:
        return await self._insights_manager.run_daydream_cycle(
            self.get_recent_chat_history, self.recall
        )
