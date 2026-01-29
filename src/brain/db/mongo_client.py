"""
MongoDB Client Module for Vira Personal Life OS.

This module provides the core database connection and collection management
for all brain modules. It uses Motor (async MongoDB driver) for compatibility
with the asyncio-based architecture.

Collections:
- memories: Long-term memory storage with embeddings
- chat_logs: Conversation history with TTL
- knowledge_graph: Entity relationship triples
- entities: Named entity registry
- schedules: Scheduled reminders and tasks
- admin_profile: User profile information
- emotional_state: AI emotional state tracking
- personas: Dynamic persona configurations
- global_context: Compressed memory summaries
- compression_log: Memory compression history
- session_metadata: Session state tracking
- memory_evolution_log: Memory change history

Fixed to handle multiple event loops (main app + FastAPI dashboard thread).
"""

import os
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import CollectionInvalid

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "vira_os")

# TTL for chat logs (90 days in seconds)
CHAT_LOG_TTL_SECONDS = 90 * 24 * 60 * 60  # 7,776,000 seconds


class MongoDBClient:
    """
    MongoDB client manager for Vira.
    
    Creates a new client per event loop to avoid cross-loop issues.
    """
    
    _clients: Dict[int, AsyncIOMotorClient] = {}
    _indexes_created: bool = False
    
    def _get_loop_id(self) -> int:
        """Get current event loop ID."""
        try:
            loop = asyncio.get_running_loop()
            return id(loop)
        except RuntimeError:
            return 0
    
    def _get_client(self) -> AsyncIOMotorClient:
        """Get or create client for current event loop."""
        loop_id = self._get_loop_id()
        if loop_id not in self._clients or self._clients[loop_id] is None:
            self._clients[loop_id] = AsyncIOMotorClient(MONGO_URI)
        return self._clients[loop_id]
    
    async def connect(self) -> None:
        """Establish connection to MongoDB and initialize collections."""
        client = self._get_client()
        
        # Verify connection
        await client.admin.command('ping')
        print(f"  ✓ MongoDB connected: {MONGO_URI}/{MONGO_DB_NAME}")
        
        # Initialize collections and indexes (only once)
        if not MongoDBClient._indexes_created:
            await self._setup_collections()
            MongoDBClient._indexes_created = True
    
    async def close(self) -> None:
        """Close the MongoDB connection for current loop."""
        loop_id = self._get_loop_id()
        if loop_id in self._clients and self._clients[loop_id]:
            self._clients[loop_id].close()
            self._clients[loop_id] = None
            print("  ✓ MongoDB connection closed")
    
    @property
    def db(self) -> AsyncIOMotorDatabase:
        """Get the database instance."""
        return self._get_client()[MONGO_DB_NAME]
    
    # ==================== COLLECTION ACCESSORS ====================
    
    @property
    def memories(self):
        """Access the memories collection."""
        return self.db["memories"]
    
    @property
    def chat_logs(self):
        """Access the chat_logs collection (with TTL)."""
        return self.db["chat_logs"]
    
    @property
    def knowledge_graph(self):
        """Access the knowledge_graph collection (triples)."""
        return self.db["knowledge_graph"]
    
    @property
    def entities(self):
        """Access the entities collection."""
        return self.db["entities"]
    
    @property
    def schedules(self):
        """Access the schedules collection."""
        return self.db["schedules"]
    
    @property
    def admin_profile(self):
        """Access the admin_profile collection (single-doc)."""
        return self.db["admin_profile"]
    
    @property
    def emotional_state(self):
        """Access the emotional_state collection (single-doc)."""
        return self.db["emotional_state"]
    
    @property
    def personas(self):
        """Access the personas collection."""
        return self.db["personas"]
    
    @property
    def memory_evolution_log(self):
        """Access the memory_evolution_log collection."""
        return self.db["memory_evolution_log"]
    
    @property
    def global_context(self):
        """Access the global_context collection (compressed memory summaries)."""
        return self.db["global_context"]
    
    @property
    def compression_log(self):
        """Access the compression_log collection (compression history)."""
        return self.db["compression_log"]
    
    @property
    def session_metadata(self):
        """Access the session_metadata collection."""
        return self.db["session_metadata"]
    
    @property
    def conversation_topics(self):
        """Access the conversation_topics collection."""
        return self.db["conversation_topics"]
    
    @property
    def insights(self):
        """Access the insights collection for daydream-generated insights."""
        return self.db["insights"]
    
    @property
    def tool_outputs(self):
        """Access the tool_outputs collection."""
        return self.db["tool_outputs"]
    
    # ==================== COLLECTION SETUP ====================
    
    async def _setup_collections(self) -> None:
        """Create collections and indexes."""
        
        # --- memories ---
        await self._create_indexes("memories", [
            IndexModel([("fingerprint", ASCENDING)], unique=True, sparse=True),
            IndexModel([("priority", DESCENDING), ("last_used", DESCENDING)]),
            IndexModel([("status", ASCENDING), ("type", ASCENDING)]),
            IndexModel([("entity", ASCENDING)], sparse=True),
            IndexModel([("is_compressed", ASCENDING)]),  # New: for compression tracking
        ])
        
        # --- chat_logs (with TTL) ---
        await self._create_indexes("chat_logs", [
            IndexModel([("timestamp", DESCENDING)]),
            IndexModel(
                [("timestamp", ASCENDING)],
                expireAfterSeconds=CHAT_LOG_TTL_SECONDS,
                name="ttl_90_days"
            ),
        ])
        
        # --- knowledge_graph ---
        await self._create_indexes("knowledge_graph", [
            IndexModel([("subject", ASCENDING)]),
            IndexModel([("object", ASCENDING)]),
            IndexModel([("subject", ASCENDING), ("predicate", ASCENDING), ("object", ASCENDING)], unique=True),
        ])
        
        # --- entities ---
        await self._create_indexes("entities", [
            IndexModel([("name", ASCENDING)], unique=True),
            IndexModel([("mention_count", DESCENDING)]),
        ])
        
        # --- schedules ---
        await self._create_indexes("schedules", [
            IndexModel([("status", ASCENDING), ("scheduled_at", ASCENDING)]),
        ])
        
        # --- personas ---
        await self._create_indexes("personas", [
            IndexModel([("name", ASCENDING)], unique=True),
            IndexModel([("is_active", ASCENDING)]),
        ])
        
        # --- global_context ---
        await self._create_indexes("global_context", [
            IndexModel([("version", DESCENDING)]),
        ])
        
        # --- compression_log ---
        await self._create_indexes("compression_log", [
            IndexModel([("timestamp", DESCENDING)]),
        ])
        
        # --- conversation_topics ---
        await self._create_indexes("conversation_topics", [
            IndexModel([("topic", ASCENDING)]),
            IndexModel([("last_mentioned", DESCENDING)]),
            IndexModel([("frequency", DESCENDING)]),
        ])
        
        # --- insights (daydream-generated) ---
        await self._create_indexes("insights", [
            IndexModel([("created_at", DESCENDING)]),
            IndexModel([("is_used", ASCENDING)]),
            IndexModel([("relevance_score", DESCENDING)]),
            IndexModel(
                [("expires_at", ASCENDING)],
                expireAfterSeconds=0,
                name="ttl_insights"
            ),
        ])
        
        print("  ✓ MongoDB indexes initialized")
    
    async def _create_indexes(self, collection_name: str, indexes: List[IndexModel]) -> None:
        """Create indexes for a collection, ignoring if they already exist."""
        try:
            collection = self.db[collection_name]
            await collection.create_indexes(indexes)
        except Exception as e:
            # Log but don't fail - indexes might already exist
            print(f"  ⚠ Index warning for {collection_name}: {e}")


# ==================== GLOBAL INSTANCE ====================

_mongo_client: Optional[MongoDBClient] = None


def get_mongo_client() -> MongoDBClient:
    """Get the global MongoDB client instance."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoDBClient()
    return _mongo_client


async def init_mongodb() -> MongoDBClient:
    """Initialize and return the MongoDB client."""
    client = get_mongo_client()
    await client.connect()
    return client


async def close_mongodb() -> None:
    """Close the MongoDB connection."""
    global _mongo_client
    if _mongo_client:
        await _mongo_client.close()
