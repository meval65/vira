import asyncio
import aiosqlite
import sqlite3
import os
import logging
import json
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

class DBConnection:
    """Asynchronous SQLite connection using aiosqlite.
    Provides async context manager and helper methods for common operations.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.getenv("DB_PATH", "storage/memory.db")
        self._conn: Optional[aiosqlite.Connection] = None
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        await self._ensure_connection()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.close()

    async def _ensure_connection(self):
        if self._conn is None:
            # Ensure directory exists
            directory = os.path.dirname(self.db_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            self._conn = await aiosqlite.connect(
                self.db_path,
                timeout=60.0,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._conn.row_factory = aiosqlite.Row
            await self._tune_database()
            await self._create_tables()
            await self._create_indexes()
            await self._migrate_schema()
            logger.info(f"[DATABASE] Async connection established to {self.db_path}")

    async def _tune_database(self):
        async with self._conn.execute("PRAGMA journal_mode=WAL;") as cur:
            await cur.fetchone()
        await self._conn.executescript(
            """
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-64000;
            PRAGMA temp_store=MEMORY;
            PRAGMA mmap_size=268435456;
            PRAGMA page_size=4096;
            PRAGMA foreign_keys=ON;
            """
        )
        await self._conn.commit()

    async def _create_tables(self):
        await self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                priority REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                embedding BLOB,
                metadata TEXT,
                cluster_id TEXT,
                fingerprint TEXT,
                entity TEXT,
                relation TEXT,
                value TEXT,
                confidence REAL DEFAULT 0.5,
                source_count INTEGER DEFAULT 1,
                CHECK(priority >= 0.0 AND priority <= 1.0),
                CHECK(confidence >= 0.0 AND confidence <= 1.0),
                CHECK(status IN ('active', 'archived', 'deleted'))
            );
            CREATE TABLE IF NOT EXISTS canonical_memory_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                fingerprint TEXT,
                old_value TEXT,
                new_value TEXT,
                confidence_before REAL,
                confidence_after REAL,
                source_count INTEGER,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK(action IN ('create', 'merge', 'resurrect', 'update', 'archive', 'delete')),
                FOREIGN KEY(memory_id) REFERENCES memories(id)
            );
            CREATE TABLE IF NOT EXISTS schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                scheduled_at TIMESTAMP NOT NULL,
                context TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                recurrence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed_at TIMESTAMP,
                execution_note TEXT,
                metadata TEXT,
                tags TEXT,
                CHECK(status IN ('pending', 'executed', 'cancelled', 'failed'))
            );
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                timezone TEXT DEFAULT 'Asia/Jakarta',
                language TEXT DEFAULT 'id',
                notification_enabled INTEGER DEFAULT 1,
                preferences_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS system_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                user_id TEXT,
                message TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS memory_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                source_memory_id TEXT NOT NULL,
                target_memory_id TEXT NOT NULL,
                relation_type TEXT NOT NULL,
                strength REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK(strength >= 0.0 AND strength <= 1.0),
                FOREIGN KEY(source_memory_id) REFERENCES memories(id),
                FOREIGN KEY(target_memory_id) REFERENCES memories(id),
                UNIQUE(source_memory_id, target_memory_id, relation_type)
            );
            """
        )
        await self._conn.commit()

    async def _create_indexes(self):
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_core ON memories(user_id, status, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_ranking ON memories(priority DESC, last_used_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(user_id) WHERE embedding IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_fingerprint ON memories(user_id, fingerprint) WHERE fingerprint IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(user_id, entity) WHERE entity IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_relation ON memories(relation) WHERE relation IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence DESC) WHERE confidence IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_source_count ON memories(source_count DESC)",
            "CREATE INDEX IF NOT EXISTS idx_canonical_log_memory ON canonical_memory_log(memory_id, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_canonical_log_user ON canonical_memory_log(user_id, action, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_canonical_log_fingerprint ON canonical_memory_log(fingerprint)",
            "CREATE INDEX IF NOT EXISTS idx_schedules_pending ON schedules(user_id, status, scheduled_at)",
            "CREATE INDEX IF NOT EXISTS idx_schedules_lookup ON schedules(user_id, scheduled_at)",
            "CREATE INDEX IF NOT EXISTS idx_logs_events ON system_logs(event_type, created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_relations_source ON memory_relations(source_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_target ON memory_relations(target_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_relations_user ON memory_relations(user_id, relation_type)"
        ]
        for sql in indexes:
            try:
                await self._conn.execute(sql)
            except Exception:
                pass
        await self._conn.commit()

    async def _migrate_schema(self):
        # Simple migration logic similar to original but async
        async with self._conn.execute("PRAGMA table_info(memories)") as cur:
            existing = {row[1] for row in await cur.fetchall()}
        # Define new columns per original migration
        new_columns = {
            "memories": [
                ("metadata", "TEXT"),
                ("cluster_id", "TEXT"),
                ("fingerprint", "TEXT"),
                ("entity", "TEXT"),
                ("relation", "TEXT"),
                ("value", "TEXT"),
                ("confidence", "REAL DEFAULT 0.5"),
                ("source_count", "INTEGER DEFAULT 1"),
            ]
        }
        for table, cols in new_columns.items():
            for col_name, col_type in cols:
                if col_name not in existing:
                    try:
                        await self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                        logger.info(f"[DATABASE] Migrated {table}: Added {col_name}")
                    except Exception as e:
                        logger.debug(f"[DATABASE] Column {col_name} migration skipped: {e}")
        await self._conn.commit()

    async def execute(self, query: str, params: Tuple = ()) -> aiosqlite.Cursor:
        async with self._lock:
            await self._ensure_connection()
            async with self._conn.execute(query, params) as cur:
                await self._conn.commit()
                return cur

    async def fetchall(self, query: str, params: Tuple = ()) -> List[aiosqlite.Row]:
        async with self._lock:
            await self._ensure_connection()
            async with self._conn.execute(query, params) as cur:
                rows = await cur.fetchall()
                return rows

    async def fetchone(self, query: str, params: Tuple = ()) -> Optional[aiosqlite.Row]:
        async with self._lock:
            await self._ensure_connection()
            async with self._conn.execute(query, params) as cur:
                row = await cur.fetchone()
                return row

    async def executemany(self, query: str, seq_of_params: List[Tuple]):
        async with self._lock:
            await self._ensure_connection()
            await self._conn.executemany(query, seq_of_params)
            await self._conn.commit()

    async def rollback(self):
        """Rollback the current transaction if needed."""
        if self._conn:
            await self._conn.rollback()

    async def close(self):
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("[DATABASE] Async connection closed")

    # Compatibility helpers for legacy code (sync wrappers) – they run the async calls in an event loop
    def get_cursor(self):
        raise NotImplementedError("Synchronous cursor access is not supported in async DBConnection. Use async methods.")

    def execute_query(self, query: str, params: Tuple = ()) -> List[Dict]:
        raise NotImplementedError("Use async fetchall instead.")

    def execute_update(self, query: str, params: Tuple = ()) -> int:
        raise NotImplementedError("Use async execute instead.")

    def execute_many(self, query: str, params_list: List[Tuple]) -> int:
        raise NotImplementedError("Use async executemany instead.")