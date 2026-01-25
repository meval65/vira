import os
import uuid
import json
import hashlib
import asyncio
import numpy as np
import aiosqlite
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field, asdict

from pydantic import BaseModel, Field
from google import genai
from google.genai import types

from src.brainstem import (
    DB_PATH, GOOGLE_API_KEY, TIER_2_MODEL, TIER_3_MODEL,
    MemoryType, MAX_RETRIEVED_MEMORIES, MIN_RELEVANCE_SCORE,
    DECAY_DAYS_EMOTION, DECAY_DAYS_GENERAL, CANONICALIZATION_INSTRUCTION
)


class TripleRelation(str, Enum):
    HAS = "has"
    IS = "is"
    LIKES = "likes"
    DISLIKES = "dislikes"
    WORKS_AT = "works_at"
    LIVES_IN = "lives_in"
    KNOWS = "knows"
    RELATED_TO = "related_to"
    CREATED = "created"
    OWNS = "owns"


@dataclass
class Memory:
    id: str
    summary: str
    memory_type: str
    priority: float = 0.5
    confidence: float = 0.5
    fingerprint: Optional[str] = None
    entity: Optional[str] = None
    relation: Optional[str] = None
    value: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    use_count: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_used_at: datetime = field(default_factory=datetime.now)
    status: str = "active"


@dataclass
class Triple:
    id: Optional[int]
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_memory_id: Optional[str] = None
    created_at: Optional[datetime] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0


class AdminProfile(BaseModel):
    telegram_name: Optional[str] = None
    full_name: Optional[str] = None
    additional_info: Optional[str] = None
    last_updated: datetime = Field(default_factory=datetime.now)


class Hippocampus:
    SIMILARITY_THRESHOLD: float = 0.90
    ARCHIVE_CHECK_DAYS: int = 90
    MIN_CONFIDENCE: float = 0.3

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._admin_profile: AdminProfile = AdminProfile()
        self._embedding_cache: Dict[int, np.ndarray] = {}
        self._embedding_dim: Optional[int] = None
        self._genai_client = None
        self._evolution_history: Dict[str, List[Dict]] = defaultdict(list)
        self._conflict_history: Dict[str, List[Dict]] = defaultdict(list)

    async def initialize(self) -> None:
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

        self._conn = await aiosqlite.connect(
            self.db_path,
            timeout=60.0,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        self._conn.row_factory = aiosqlite.Row

        await self._tune_database()
        await self._create_tables()
        await self._create_indexes()
        await self._load_admin_profile()

        if GOOGLE_API_KEY:
            self._genai_client = genai.Client(api_key=GOOGLE_API_KEY)

    async def _tune_database(self) -> None:
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.executescript("""
            PRAGMA synchronous=NORMAL;
            PRAGMA cache_size=-64000;
            PRAGMA temp_store=MEMORY;
            PRAGMA mmap_size=268435456;
            PRAGMA page_size=4096;
            PRAGMA foreign_keys=ON;
        """)
        await self._conn.commit()

    async def _create_tables(self) -> None:
        await self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
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

            CREATE TABLE IF NOT EXISTS triples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                source_memory_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                access_count INTEGER DEFAULT 0,
                metadata TEXT,
                UNIQUE(subject, predicate, object)
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                entity_type TEXT DEFAULT 'unknown',
                aliases TEXT,
                properties TEXT,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                mention_count INTEGER DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS admin_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                telegram_name TEXT,
                full_name TEXT,
                additional_info TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheduled_at TIMESTAMP NOT NULL,
                context TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                recurrence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed_at TIMESTAMP,
                execution_note TEXT,
                metadata TEXT,
                CHECK(status IN ('pending', 'executed', 'cancelled', 'failed'))
            );

            CREATE TABLE IF NOT EXISTS memory_evolution_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_id TEXT NOT NULL,
                action TEXT NOT NULL,
                fingerprint TEXT,
                old_value TEXT,
                new_value TEXT,
                confidence_before REAL,
                confidence_after REAL,
                update_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS emotional_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                current_mood TEXT DEFAULT 'neutral',
                empathy_level REAL DEFAULT 0.5,
                satisfaction_level REAL DEFAULT 0.0,
                last_interaction TIMESTAMP,
                mood_history TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        await self._conn.commit()

    async def _create_indexes(self) -> None:
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority DESC, last_used_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_fingerprint ON memories(fingerprint) WHERE fingerprint IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity) WHERE entity IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_triples_subject ON triples(subject)",
            "CREATE INDEX IF NOT EXISTS idx_triples_object ON triples(object)",
            "CREATE INDEX IF NOT EXISTS idx_triples_predicate ON triples(predicate)",
            "CREATE INDEX IF NOT EXISTS idx_schedules_pending ON schedules(status, scheduled_at)",
            "CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name)"
        ]
        for sql in indexes:
            try:
                await self._conn.execute(sql)
            except Exception:
                pass
        await self._conn.commit()

    async def _load_admin_profile(self) -> None:
        row = await self._conn.execute_fetchall(
            "SELECT telegram_name, full_name, additional_info, last_updated FROM admin_profile WHERE id = 1"
        )
        if row:
            r = row[0]
            self._admin_profile = AdminProfile(
                telegram_name=r[0],
                full_name=r[1],
                additional_info=r[2],
                last_updated=r[3] if r[3] else datetime.now()
            )

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def admin_profile(self) -> AdminProfile:
        return self._admin_profile

    async def update_admin_profile(
        self,
        telegram_name: Optional[str] = None,
        full_name: Optional[str] = None,
        additional_info: Optional[str] = None
    ) -> None:
        if telegram_name:
            self._admin_profile.telegram_name = telegram_name
        if full_name:
            self._admin_profile.full_name = full_name
        if additional_info:
            self._admin_profile.additional_info = additional_info
        self._admin_profile.last_updated = datetime.now()

        await self._conn.execute("""
            INSERT INTO admin_profile (id, telegram_name, full_name, additional_info, last_updated)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                telegram_name=COALESCE(excluded.telegram_name, telegram_name),
                full_name=COALESCE(excluded.full_name, full_name),
                additional_info=COALESCE(excluded.additional_info, additional_info),
                last_updated=excluded.last_updated
        """, (telegram_name, full_name, additional_info, datetime.now()))
        await self._conn.commit()

    async def store(
        self,
        summary: str,
        memory_type: str,
        priority: float = 0.5,
        embedding: Optional[List[float]] = None
    ) -> str:
        canonical = await self._canonicalize(summary, memory_type)

        fingerprint = canonical.get("fingerprint")
        if fingerprint:
            existing = await self._find_by_fingerprint(fingerprint)
            if existing:
                merged = await self._merge_memories(existing, canonical)
                await self._update_memory(existing["id"], merged)
                self._record_evolution(fingerprint, existing.get("summary"), summary, "merge")
                return existing["id"]

        memory_id = str(uuid.uuid4())
        emb_blob = None
        if embedding:
            emb_array = np.array(embedding, dtype=np.float32)
            emb_blob = emb_array.tobytes()

        await self._conn.execute("""
            INSERT INTO memories (id, summary, memory_type, priority, embedding, fingerprint, entity, relation, value, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory_id,
            summary,
            memory_type,
            priority,
            emb_blob,
            canonical.get("fingerprint"),
            canonical.get("entity"),
            canonical.get("relation"),
            json.dumps(canonical.get("value")) if canonical.get("value") else None,
            canonical.get("confidence", 0.5)
        ))
        await self._conn.commit()

        if fingerprint:
            self._record_evolution(fingerprint, None, summary, "create")

        return memory_id

    async def recall(
        self,
        query: str,
        limit: int = MAX_RETRIEVED_MEMORIES,
        query_embedding: Optional[List[float]] = None
    ) -> List[Memory]:
        rows = await self._conn.execute_fetchall("""
            SELECT id, summary, memory_type, priority, confidence, fingerprint, 
                   entity, relation, value, embedding, use_count, created_at, last_used_at, status
            FROM memories
            WHERE status = 'active'
            ORDER BY priority DESC, last_used_at DESC
            LIMIT ?
        """, (limit * 3,))

        memories = []
        for r in rows:
            emb = None
            if r[9]:
                try:
                    emb = np.frombuffer(r[9], dtype=np.float32)
                except Exception:
                    pass

            memories.append(Memory(
                id=r[0],
                summary=r[1],
                memory_type=r[2],
                priority=r[3],
                confidence=r[4],
                fingerprint=r[5],
                entity=r[6],
                relation=r[7],
                value=r[8],
                embedding=emb,
                use_count=r[10],
                created_at=r[11],
                last_used_at=r[12],
                status=r[13]
            ))

        if query_embedding and memories:
            query_vec = np.array(query_embedding, dtype=np.float32)
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)

            scored = []
            for m in memories:
                if m.embedding is not None:
                    m_norm = m.embedding / (np.linalg.norm(m.embedding) + 1e-10)
                    sim = float(np.dot(query_norm, m_norm))
                    if sim >= MIN_RELEVANCE_SCORE:
                        scored.append((m, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            memories = [m for m, _ in scored[:limit]]

        for m in memories[:limit]:
            await self._mark_memory_used(m.id)

        return memories[:limit]

    async def query_entity(self, entity: str) -> Dict[str, Any]:
        entity_lower = entity.lower().strip()

        outgoing = await self._conn.execute_fetchall("""
            SELECT id, subject, predicate, object, confidence
            FROM triples WHERE LOWER(subject) = ?
            ORDER BY confidence DESC LIMIT 10
        """, (entity_lower,))

        incoming = await self._conn.execute_fetchall("""
            SELECT id, subject, predicate, object, confidence
            FROM triples WHERE LOWER(object) = ?
            ORDER BY confidence DESC LIMIT 10
        """, (entity_lower,))

        related_memories = await self._conn.execute_fetchall("""
            SELECT id, summary, memory_type, confidence
            FROM memories WHERE LOWER(entity) = ? AND status = 'active'
            ORDER BY priority DESC LIMIT 5
        """, (entity_lower,))

        for row in outgoing + incoming:
            await self._conn.execute(
                "UPDATE triples SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
                (datetime.now(), row[0])
            )
        await self._conn.commit()

        return {
            "entity": entity,
            "outgoing": [{"subject": r[1], "predicate": r[2], "object": r[3], "confidence": r[4]} for r in outgoing],
            "incoming": [{"subject": r[1], "predicate": r[2], "object": r[3], "confidence": r[4]} for r in incoming],
            "memories": [{"id": r[0], "summary": r[1], "type": r[2], "confidence": r[3]} for r in related_memories]
        }

    async def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        confidence: float = 0.8,
        source_memory_id: Optional[str] = None
    ) -> int:
        subject = subject.lower().strip()
        predicate = predicate.lower().strip().replace(" ", "_")
        obj = obj.lower().strip()

        await self._ensure_entity(subject)
        await self._ensure_entity(obj)

        existing = await self._conn.execute_fetchall(
            "SELECT id, confidence FROM triples WHERE subject = ? AND predicate = ? AND object = ?",
            (subject, predicate, obj)
        )

        if existing:
            new_conf = min(1.0, existing[0][1] + 0.1)
            await self._conn.execute(
                "UPDATE triples SET confidence = ?, last_accessed = ? WHERE id = ?",
                (new_conf, datetime.now(), existing[0][0])
            )
            await self._conn.commit()
            return existing[0][0]

        cursor = await self._conn.execute("""
            INSERT INTO triples (subject, predicate, object, confidence, source_memory_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (subject, predicate, obj, confidence, source_memory_id, datetime.now()))
        await self._conn.commit()
        return cursor.lastrowid

    async def traverse(
        self,
        start_entity: str,
        max_hops: int = 2,
        min_confidence: float = 0.4
    ) -> Dict[str, List[Triple]]:
        start = start_entity.lower().strip()
        visited = {start}
        result: Dict[str, List[Triple]] = {start: []}

        current_level = [start]
        for hop in range(max_hops):
            next_level = []
            for entity in current_level:
                rows = await self._conn.execute_fetchall("""
                    SELECT id, subject, predicate, object, confidence, source_memory_id, created_at, last_accessed, access_count
                    FROM triples
                    WHERE (LOWER(subject) = ? OR LOWER(object) = ?) AND confidence >= ?
                """, (entity, entity, min_confidence))

                for r in rows:
                    triple = Triple(
                        id=r[0], subject=r[1], predicate=r[2], object=r[3],
                        confidence=r[4], source_memory_id=r[5], created_at=r[6],
                        last_accessed=r[7], access_count=r[8]
                    )
                    if entity not in result:
                        result[entity] = []
                    result[entity].append(triple)

                    neighbor = r[3] if r[1].lower() == entity else r[1]
                    if neighbor.lower() not in visited:
                        visited.add(neighbor.lower())
                        next_level.append(neighbor.lower())

            current_level = next_level

        return result

    async def get_memory_stats(self) -> Dict[str, int]:
        active = await self._conn.execute_fetchall(
            "SELECT COUNT(*) FROM memories WHERE status = 'active'"
        )
        archived = await self._conn.execute_fetchall(
            "SELECT COUNT(*) FROM memories WHERE status = 'archived'"
        )
        triples = await self._conn.execute_fetchall(
            "SELECT COUNT(*) FROM triples"
        )

        return {
            "active": active[0][0] if active else 0,
            "archived": archived[0][0] if archived else 0,
            "triples": triples[0][0] if triples else 0
        }

    async def add_schedule(
        self,
        trigger_time: datetime,
        context: str,
        priority: int = 0,
        recurring: Optional[str] = None
    ) -> int:
        cursor = await self._conn.execute("""
            INSERT INTO schedules (scheduled_at, context, priority, recurrence, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (trigger_time, context, priority, recurring, datetime.now()))
        await self._conn.commit()
        return cursor.lastrowid

    async def get_pending_schedules(self, limit: int = 10) -> List[Dict]:
        rows = await self._conn.execute_fetchall("""
            SELECT id, scheduled_at, context, priority, recurrence
            FROM schedules
            WHERE status = 'pending' AND scheduled_at <= ?
            ORDER BY scheduled_at ASC
            LIMIT ?
        """, (datetime.now(), limit))

        return [
            {"id": r[0], "scheduled_at": r[1], "context": r[2], "priority": r[3], "recurrence": r[4]}
            for r in rows
        ]

    async def get_upcoming_schedules(self, hours_ahead: int = 72) -> List[Dict]:
        future = datetime.now() + timedelta(hours=hours_ahead)
        rows = await self._conn.execute_fetchall("""
            SELECT id, scheduled_at, context, priority, recurrence
            FROM schedules
            WHERE status = 'pending' AND scheduled_at <= ?
            ORDER BY scheduled_at ASC
        """, (future,))

        return [
            {"id": r[0], "scheduled_at": r[1], "context": r[2], "priority": r[3], "recurrence": r[4]}
            for r in rows
        ]

    async def mark_schedule_executed(self, schedule_id: int, note: Optional[str] = None) -> None:
        await self._conn.execute("""
            UPDATE schedules SET status = 'executed', executed_at = ?, execution_note = ?
            WHERE id = ?
        """, (datetime.now(), note, schedule_id))
        await self._conn.commit()

    async def get_schedule_stats(self) -> Dict[str, int]:
        pending = await self._conn.execute_fetchall(
            "SELECT COUNT(*) FROM schedules WHERE status = 'pending'"
        )
        executed = await self._conn.execute_fetchall(
            "SELECT COUNT(*) FROM schedules WHERE status = 'executed'"
        )

        return {
            "pending": pending[0][0] if pending else 0,
            "executed": executed[0][0] if executed else 0
        }

    async def _ensure_entity(self, name: str) -> None:
        name_lower = name.lower().strip()
        existing = await self._conn.execute_fetchall(
            "SELECT id FROM entities WHERE LOWER(name) = ?", (name_lower,)
        )
        if not existing:
            await self._conn.execute(
                "INSERT INTO entities (name) VALUES (?)", (name_lower,)
            )
            await self._conn.commit()
        else:
            await self._conn.execute(
                "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
                (existing[0][0],)
            )
            await self._conn.commit()

    async def _find_by_fingerprint(self, fingerprint: str) -> Optional[Dict]:
        row = await self._conn.execute_fetchall("""
            SELECT id, summary, memory_type, priority, confidence, entity, relation, value
            FROM memories WHERE fingerprint = ? AND status = 'active'
        """, (fingerprint,))

        if row:
            r = row[0]
            return {
                "id": r[0], "summary": r[1], "memory_type": r[2], "priority": r[3],
                "confidence": r[4], "entity": r[5], "relation": r[6], "value": r[7]
            }
        return None

    async def _update_memory(self, memory_id: str, data: Dict) -> None:
        await self._conn.execute("""
            UPDATE memories SET
                summary = ?,
                priority = ?,
                confidence = ?,
                value = ?,
                last_used_at = ?,
                use_count = use_count + 1
            WHERE id = ?
        """, (
            data.get("summary"),
            data.get("priority", 0.5),
            data.get("confidence", 0.5),
            json.dumps(data.get("value")) if data.get("value") else None,
            datetime.now(),
            memory_id
        ))
        await self._conn.commit()

    async def _mark_memory_used(self, memory_id: str) -> None:
        await self._conn.execute("""
            UPDATE memories SET last_used_at = ?, use_count = use_count + 1 WHERE id = ?
        """, (datetime.now(), memory_id))
        await self._conn.commit()

    async def _canonicalize(self, summary: str, memory_type: str) -> Dict:
        if not self._genai_client:
            return self._fallback_canonicalize(summary, memory_type)

        try:
            prompt = f"{CANONICALIZATION_INSTRUCTION}\n\nInput: \"{summary}\""
            response = self._genai_client.models.generate_content(
                model=TIER_3_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256
                )
            )
            text = response.text.strip()
            return self._extract_json(text) or self._fallback_canonicalize(summary, memory_type)
        except Exception:
            return self._fallback_canonicalize(summary, memory_type)

    def _fallback_canonicalize(self, summary: str, memory_type: str) -> Dict:
        words = summary.lower().split()
        entity = words[-1] if words else "unknown"
        relation = "related_to"

        for w in ["likes", "loves", "enjoys", "prefers"]:
            if w in words:
                relation = "likes"
                break
        for w in ["hates", "dislikes"]:
            if w in words:
                relation = "dislikes"
                break
        for w in ["is", "am", "are"]:
            if w in words:
                relation = "is"
                break

        fingerprint = hashlib.md5(f"{memory_type}:{relation}:{entity}".encode()).hexdigest()[:16]

        return {
            "fingerprint": f"{memory_type}:{relation}:{fingerprint}",
            "type": memory_type,
            "entity": entity,
            "relation": relation,
            "value": True,
            "confidence": 0.6
        }

    async def _merge_memories(self, existing: Dict, new: Dict) -> Dict:
        new_conf = min(1.0, existing.get("confidence", 0.5) + 0.1)
        new_priority = min(1.0, existing.get("priority", 0.5) + 0.05)

        return {
            "summary": new.get("summary", existing.get("summary")),
            "priority": new_priority,
            "confidence": new_conf,
            "value": new.get("value", existing.get("value"))
        }

    def _record_evolution(
        self,
        fingerprint: str,
        old_value: Optional[str],
        new_value: str,
        action: str
    ) -> None:
        self._evolution_history[fingerprint].append({
            "old_value": old_value,
            "new_value": new_value,
            "action": action,
            "timestamp": datetime.now().isoformat()
        })

    def _extract_json(self, text: str) -> Optional[Dict]:
        import re
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return None

    async def save_emotional_state(
        self,
        mood: str,
        empathy: float,
        satisfaction: float,
        mood_history: List[Dict]
    ) -> None:
        await self._conn.execute("""
            INSERT INTO emotional_state (id, current_mood, empathy_level, satisfaction_level, last_interaction, mood_history, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                current_mood = excluded.current_mood,
                empathy_level = excluded.empathy_level,
                satisfaction_level = excluded.satisfaction_level,
                last_interaction = excluded.last_interaction,
                mood_history = excluded.mood_history,
                updated_at = excluded.updated_at
        """, (mood, empathy, satisfaction, datetime.now(), json.dumps(mood_history[-10:]), datetime.now()))
        await self._conn.commit()

    async def load_emotional_state(self) -> Optional[Dict]:
        rows = await self._conn.execute_fetchall(
            "SELECT current_mood, empathy_level, satisfaction_level, last_interaction, mood_history FROM emotional_state WHERE id = 1"
        )
        if rows:
            r = rows[0]
            return {
                "mood": r[0],
                "empathy": r[1],
                "satisfaction": r[2],
                "last_interaction": r[3],
                "history": json.loads(r[4]) if r[4] else []
            }
        return None
