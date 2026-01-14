import sqlite3
import os
import logging
import threading
import json
import shutil
from typing import Optional, Dict, List, Any, Union, Generator
from contextlib import contextmanager
from datetime import datetime
from src.config import DB_PATH

logger = logging.getLogger(__name__)

class DBConnection:
    def __init__(self):
        self.db_path = DB_PATH
        self._lock = threading.RLock()
        self._ensure_directory_exists()
        
        try:
            self.conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=60.0,
                detect_types=sqlite3.PARSE_DECLTYPES
            )
            self.conn.row_factory = sqlite3.Row
            
            self._tune_database()
            self._create_tables()
            self._create_indexes()
            self._migrate_schema()
            
            logger.info(f"[DATABASE] Connected to {self.db_path}")
            
        except sqlite3.Error as e:
            logger.critical(f"[DATABASE] Fatal connection error: {e}", exc_info=True)
            raise e

    def _ensure_directory_exists(self):
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

    def _tune_database(self):
        with self._lock:
            try:
                cursor = self.conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA cache_size=-64000;")
                cursor.execute("PRAGMA temp_store=MEMORY;")
                cursor.execute("PRAGMA mmap_size=268435456;")
                cursor.execute("PRAGMA page_size=4096;")
                cursor.execute("PRAGMA foreign_keys=ON;")
                self.conn.commit()
            except sqlite3.Error as e:
                logger.warning(f"[DATABASE] Tuning warning: {e}")

    def _create_tables(self):
        with self._lock:
            cursor = self.conn.cursor()
            
            cursor.execute("""
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
                )
            """)
            
            cursor.execute("""
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
                )
            """)
            
            cursor.execute("""
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
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_preferences (
                    user_id TEXT PRIMARY KEY,
                    timezone TEXT DEFAULT 'Asia/Jakarta',
                    language TEXT DEFAULT 'id',
                    notification_enabled INTEGER DEFAULT 1,
                    preferences_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    user_id TEXT,
                    message TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
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
                )
            """)
            
            self.conn.commit()

    def _create_indexes(self):
        with self._lock:
            cursor = self.conn.cursor()
            
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
            
            for idx_sql in indexes:
                try:
                    cursor.execute(idx_sql)
                except sqlite3.Error:
                    pass
            
            self.conn.commit()

    def _migrate_schema(self):
        with self._lock:
            cursor = self.conn.cursor()
            try:
                table_columns = {
                    'schedules': [
                        ('recurrence', 'TEXT'),
                        ('executed_at', 'TIMESTAMP'),
                        ('execution_note', 'TEXT'),
                        ('metadata', 'TEXT'),
                        ('tags', 'TEXT')
                    ],
                    'memories': [
                        ('metadata', 'TEXT'),
                        ('cluster_id', 'TEXT'),
                        ('fingerprint', 'TEXT'),
                        ('entity', 'TEXT'),
                        ('relation', 'TEXT'),
                        ('value', 'TEXT'),
                        ('confidence', 'REAL DEFAULT 0.5'),
                        ('source_count', 'INTEGER DEFAULT 1')
                    ]
                }
                
                for table, columns in table_columns.items():
                    cursor.execute(f"PRAGMA table_info({table})")
                    existing = {col[1] for col in cursor.fetchall()}
                    
                    for col_name, col_type in columns:
                        if col_name not in existing:
                            try:
                                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                                logger.info(f"[DATABASE] Migrated {table}: Added {col_name}")
                            except sqlite3.OperationalError as e:
                                logger.debug(f"[DATABASE] Column {col_name} migration skipped: {e}")
                
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row[0] for row in cursor.fetchall()}
                
                if 'canonical_memory_log' not in existing_tables:
                    cursor.execute("""
                        CREATE TABLE canonical_memory_log (
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
                            CHECK(action IN ('create', 'merge', 'resurrect', 'update', 'archive', 'delete'))
                        )
                    """)
                    logger.info("[DATABASE] Created canonical_memory_log table")
                
                if 'memory_relations' not in existing_tables:
                    cursor.execute("""
                        CREATE TABLE memory_relations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT NOT NULL,
                            source_memory_id TEXT NOT NULL,
                            target_memory_id TEXT NOT NULL,
                            relation_type TEXT NOT NULL,
                            strength REAL DEFAULT 0.5,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            CHECK(strength >= 0.0 AND strength <= 1.0),
                            UNIQUE(source_memory_id, target_memory_id, relation_type)
                        )
                    """)
                    logger.info("[DATABASE] Created memory_relations table")
                
                self.conn.commit()
            except Exception as e:
                logger.error(f"[DATABASE] Schema migration failed: {e}")

    def get_cursor(self) -> sqlite3.Cursor:
        return self.conn.cursor()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Cursor, None, None]:
        with self._lock:
            cursor = self.conn.cursor()
            try:
                yield cursor
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise
            finally:
                cursor.close()

    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, params)
                return cursor.fetchall()
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] Query Error: {e}")
                raise
            finally:
                cursor.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(query, params)
                affected = cursor.rowcount
                self.conn.commit()
                return affected
            except sqlite3.Error as e:
                self.conn.rollback()
                logger.error(f"[DATABASE] Update Error: {e}")
                raise
            finally:
                cursor.close()

    def execute_many(self, query: str, params_list: List[tuple]) -> int:
        if not params_list:
            return 0
            
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.executemany(query, params_list)
                affected = cursor.rowcount
                self.conn.commit()
                return affected
            except sqlite3.Error as e:
                self.conn.rollback()
                logger.error(f"[DATABASE] Batch Execute Error: {e}")
                raise
            finally:
                cursor.close()

    def log_canonical_action(self, memory_id: str, user_id: str, action: str,
                            fingerprint: Optional[str] = None,
                            old_value: Any = None, new_value: Any = None,
                            confidence_before: Optional[float] = None,
                            confidence_after: Optional[float] = None,
                            source_count: Optional[int] = None,
                            metadata: Optional[Dict] = None):
        try:
            meta_str = json.dumps(metadata) if metadata else None
            old_val_str = json.dumps(old_value) if old_value is not None else None
            new_val_str = json.dumps(new_value) if new_value is not None else None
            
            self.execute_update("""
                INSERT INTO canonical_memory_log (
                    memory_id, user_id, action, fingerprint,
                    old_value, new_value, confidence_before, confidence_after,
                    source_count, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id, user_id, action, fingerprint,
                old_val_str, new_val_str, confidence_before, confidence_after,
                source_count, meta_str
            ))
            logger.debug(f"[CANONICAL-LOG] {action} logged for memory {memory_id}")
        except Exception as e:
            logger.error(f"[CANONICAL-LOG] Failed to log action: {e}")

    def get_memory_by_fingerprint(self, user_id: str, fingerprint: str, 
                                  status: str = 'active') -> Optional[Dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, summary, memory_type, priority, embedding, 
                       use_count, last_used_at, metadata, fingerprint,
                       entity, relation, value, confidence, source_count
                FROM memories
                WHERE user_id=? AND fingerprint=? AND status=?
                LIMIT 1
            """, (user_id, fingerprint, status))
            
            row = cursor.fetchone()
            if row:
                return {
                    'id': row[0],
                    'summary': row[1],
                    'memory_type': row[2],
                    'priority': row[3],
                    'embedding': row[4],
                    'use_count': row[5],
                    'last_used_at': row[6],
                    'metadata': json.loads(row[7]) if row[7] else {},
                    'fingerprint': row[8],
                    'entity': row[9],
                    'relation': row[10],
                    'value': row[11],
                    'confidence': row[12],
                    'source_count': row[13]
                }
        except Exception as e:
            logger.error(f"[DATABASE] Get by fingerprint failed: {e}")
        return None

    def search_memories_by_entity(self, user_id: str, entity: str, 
                                  limit: int = 10) -> List[Dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, summary, fingerprint, entity, relation, value, 
                       confidence, source_count, priority
                FROM memories
                WHERE user_id=? AND entity=? AND status='active'
                ORDER BY confidence DESC, source_count DESC
                LIMIT ?
            """, (user_id, entity, limit))
            
            return [
                {
                    'id': row[0],
                    'summary': row[1],
                    'fingerprint': row[2],
                    'entity': row[3],
                    'relation': row[4],
                    'value': row[5],
                    'confidence': row[6],
                    'source_count': row[7],
                    'priority': row[8]
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"[DATABASE] Search by entity failed: {e}")
            return []

    def search_memories_by_relation(self, user_id: str, relation: str,
                                    limit: int = 10) -> List[Dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT id, summary, fingerprint, entity, relation, value,
                       confidence, source_count
                FROM memories
                WHERE user_id=? AND relation=? AND status='active'
                ORDER BY confidence DESC, source_count DESC
                LIMIT ?
            """, (user_id, relation, limit))
            
            return [
                {
                    'id': row[0],
                    'summary': row[1],
                    'fingerprint': row[2],
                    'entity': row[3],
                    'relation': row[4],
                    'value': row[5],
                    'confidence': row[6],
                    'source_count': row[7]
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"[DATABASE] Search by relation failed: {e}")
            return []

    def get_canonical_memory_stats(self, user_id: str) -> Dict:
        stats = {}
        try:
            cursor = self.conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*), AVG(confidence), AVG(source_count)
                FROM memories
                WHERE user_id=? AND status='active' AND fingerprint IS NOT NULL
            """, (user_id,))
            row = cursor.fetchone()
            stats['total_canonical'] = row[0] or 0
            stats['avg_confidence'] = round(row[1], 3) if row[1] else 0.0
            stats['avg_source_count'] = round(row[2], 2) if row[2] else 0.0
            
            cursor.execute("""
                SELECT memory_type, COUNT(*) 
                FROM memories
                WHERE user_id=? AND status='active' AND fingerprint IS NOT NULL
                GROUP BY memory_type
            """, (user_id,))
            stats['by_type'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT relation, COUNT(*)
                FROM memories
                WHERE user_id=? AND status='active' AND relation IS NOT NULL
                GROUP BY relation
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """, (user_id,))
            stats['top_relations'] = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT entity, COUNT(*)
                FROM memories
                WHERE user_id=? AND status='active' AND entity IS NOT NULL
                GROUP BY entity
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """, (user_id,))
            stats['top_entities'] = {row[0]: row[1] for row in cursor.fetchall()}
            
        except Exception as e:
            logger.error(f"[DATABASE] Canonical stats failed: {e}")
        return stats

    def get_memory_change_log(self, memory_id: str, limit: int = 20) -> List[Dict]:
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT action, fingerprint, old_value, new_value,
                       confidence_before, confidence_after, source_count,
                       created_at
                FROM canonical_memory_log
                WHERE memory_id=?
                ORDER BY created_at DESC
                LIMIT ?
            """, (memory_id, limit))
            
            return [
                {
                    'action': row[0],
                    'fingerprint': row[1],
                    'old_value': json.loads(row[2]) if row[2] else None,
                    'new_value': json.loads(row[3]) if row[3] else None,
                    'confidence_before': row[4],
                    'confidence_after': row[5],
                    'source_count': row[6],
                    'created_at': row[7]
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"[DATABASE] Get change log failed: {e}")
            return []

    def set_preference(self, user_id: str, key: str, value: Any):
        with self._lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute("SELECT preferences_json FROM user_preferences WHERE user_id = ?", (user_id,))
                row = cursor.fetchone()
                
                prefs = {}
                if row and row[0]:
                    try:
                        prefs = json.loads(row[0])
                    except json.JSONDecodeError:
                        pass
                
                prefs[key] = value
                json_str = json.dumps(prefs)
                
                cursor.execute("""
                    INSERT INTO user_preferences (user_id, preferences_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id) DO UPDATE SET
                    preferences_json = excluded.preferences_json,
                    updated_at = CURRENT_TIMESTAMP
                """, (user_id, json_str))
                
                self.conn.commit()
            except Exception as e:
                logger.error(f"[DATABASE] Set preference failed: {e}")

    def get_preference(self, user_id: str, key: str, default: Any = None) -> Any:
        try:
            cursor = self.conn.cursor()
            cursor.execute("SELECT preferences_json FROM user_preferences WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            
            if row and row[0]:
                prefs = json.loads(row[0])
                return prefs.get(key, default)
            return default
        except Exception:
            return default

    def log_event(self, event_type: str, user_id: Optional[str] = None, 
                  message: str = "", metadata: Optional[Dict] = None):
        try:
            meta_str = json.dumps(metadata) if metadata else None
            self.execute_update("""
                INSERT INTO system_logs (event_type, user_id, message, metadata)
                VALUES (?, ?, ?, ?)
            """, (event_type, user_id, message, meta_str))
        except Exception:
            pass

    def get_database_stats(self) -> Dict:
        stats = {}
        try:
            with self._lock:
                cursor = self.conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM memories WHERE status='active'")
                stats['active_memories'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM memories WHERE status='active' AND fingerprint IS NOT NULL")
                stats['canonical_memories'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM schedules WHERE status='pending'")
                stats['pending_schedules'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT user_id) FROM memories")
                stats['total_users'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM canonical_memory_log")
                stats['total_canonical_actions'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
                size_bytes = cursor.fetchone()[0]
                stats['db_size_mb'] = round(size_bytes / (1024 * 1024), 2)
                
                cursor.execute("PRAGMA integrity_check")
                stats['integrity'] = cursor.fetchone()[0]
                
        except Exception as e:
            logger.error(f"[DATABASE] Stats error: {e}")
            return {"error": str(e)}
        return stats

    def maintenance(self):
        with self._lock:
            try:
                self.conn.execute("VACUUM")
                self.conn.execute("ANALYZE")
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.info("[DATABASE] Maintenance complete")
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] Maintenance failed: {e}")

    def backup(self, backup_dir: str = "backups") -> bool:
        with self._lock:
            try:
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                    
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"backup_{timestamp}_{os.path.basename(self.db_path)}"
                dst = os.path.join(backup_dir, filename)
                
                self.conn.execute("PRAGMA wal_checkpoint(FULL)")
                shutil.copy2(self.db_path, dst)
                logger.info(f"[DATABASE] Backup created: {dst}")
                return True
            except Exception as e:
                logger.error(f"[DATABASE] Backup failed: {e}")
                return False

    def close(self):
        with self._lock:
            if self.conn:
                try:
                    self.conn.close()
                except Exception:
                    pass

    def commit(self):
        with self._lock:
            self.conn.commit()

    def rollback(self):
        with self._lock:
            self.conn.rollback()

    def __del__(self):
        self.close()