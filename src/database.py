import sqlite3
import os
import logging
import threading
from typing import Optional, Dict, List
from contextlib import contextmanager
from src.config import DB_PATH

logger = logging.getLogger(__name__)

class DBConnection:
    def __init__(self):
        self.db_path = DB_PATH
        self._ensure_directory_exists()
        self._lock = threading.RLock()
        self._connection_pool = {}
        self._max_connections = 5
        
        try:
            self.conn = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            self.conn.row_factory = sqlite3.Row
            self._tune_database()
            self._create_tables()
            self._create_indexes()
            self._migrate_schema()
            logger.info(f"[DATABASE] Connected to {self.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"[DATABASE] Connection failed: {e}", exc_info=True)
            raise e

    def _ensure_directory_exists(self):
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"[DATABASE] Created directory: {directory}")

    def _tune_database(self):
        with self._lock:
            cursor = self.conn.cursor()
            
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA cache_size=-64000;")
            cursor.execute("PRAGMA temp_store=MEMORY;")
            cursor.execute("PRAGMA mmap_size=268435456;")
            cursor.execute("PRAGMA page_size=4096;")
            
            self.conn.commit()
            logger.info("[DATABASE] Performance tuning applied")

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
                    CHECK(priority >= 0.0 AND priority <= 1.0),
                    CHECK(status IN ('active', 'archived', 'deleted'))
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed_at TIMESTAMP,
                    execution_note TEXT,
                    metadata TEXT,
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
            
            self.conn.commit()
            logger.info("[DATABASE] Tables created/verified")

    def _create_indexes(self):
        with self._lock:
            cursor = self.conn.cursor()
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_memories_user_status ON memories(user_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type, status)",
                "CREATE INDEX IF NOT EXISTS idx_memories_priority ON memories(priority DESC)",
                "CREATE INDEX IF NOT EXISTS idx_memories_last_used ON memories(last_used_at DESC)",
                "CREATE INDEX IF NOT EXISTS idx_memories_composite ON memories(user_id, status, priority, last_used_at)",
                
                "CREATE INDEX IF NOT EXISTS idx_schedules_user ON schedules(user_id, status)",
                "CREATE INDEX IF NOT EXISTS idx_schedules_time ON schedules(scheduled_at, status)",
                "CREATE INDEX IF NOT EXISTS idx_schedules_status ON schedules(status, scheduled_at)",
                "CREATE INDEX IF NOT EXISTS idx_schedules_composite ON schedules(user_id, status, scheduled_at)",
                
                "CREATE INDEX IF NOT EXISTS idx_logs_user ON system_logs(user_id, created_at)",
                "CREATE INDEX IF NOT EXISTS idx_logs_event ON system_logs(event_type, created_at)"
            ]
            
            for index_sql in indexes:
                try:
                    cursor.execute(index_sql)
                except sqlite3.Error as e:
                    logger.warning(f"[DATABASE] Index creation warning: {e}")
            
            self.conn.commit()
            logger.info("[DATABASE] Indexes created/verified")

    def _migrate_schema(self):
        with self._lock:
            cursor = self.conn.cursor()
            
            try:
                cursor.execute("PRAGMA table_info(schedules)")
                columns = [col[1] for col in cursor.fetchall()]
                
                if 'executed_at' not in columns:
                    cursor.execute("ALTER TABLE schedules ADD COLUMN executed_at TIMESTAMP")
                    logger.info("[DATABASE] Added executed_at column to schedules")
                
                if 'execution_note' not in columns:
                    cursor.execute("ALTER TABLE schedules ADD COLUMN execution_note TEXT")
                    logger.info("[DATABASE] Added execution_note column to schedules")
                
                if 'metadata' not in columns:
                    cursor.execute("ALTER TABLE schedules ADD COLUMN metadata TEXT")
                    logger.info("[DATABASE] Added metadata column to schedules")
                
                cursor.execute("PRAGMA table_info(memories)")
                mem_columns = [col[1] for col in cursor.fetchall()]
                
                if 'metadata' not in mem_columns:
                    cursor.execute("ALTER TABLE memories ADD COLUMN metadata TEXT")
                    logger.info("[DATABASE] Added metadata column to memories")
                
                self.conn.commit()
                
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] Migration error: {e}")

    def get_cursor(self) -> sqlite3.Cursor:
        return self.conn.cursor()

    def commit(self):
        with self._lock:
            try:
                self.conn.commit()
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] Commit failed: {e}")
                self.conn.rollback()
                raise

    def rollback(self):
        with self._lock:
            try:
                self.conn.rollback()
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] Rollback failed: {e}")

    @contextmanager
    def transaction(self):
        cursor = self.get_cursor()
        try:
            yield cursor
            self.commit()
        except Exception as e:
            self.rollback()
            logger.error(f"[DATABASE] Transaction failed: {e}", exc_info=True)
            raise
        finally:
            cursor.close()

    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        with self._lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(query, params)
                return cursor.fetchall()
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] Query failed: {query[:100]}... Error: {e}")
                raise
            finally:
                cursor.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        with self._lock:
            cursor = self.get_cursor()
            try:
                cursor.execute(query, params)
                affected = cursor.rowcount
                self.commit()
                return affected
            except sqlite3.Error as e:
                self.rollback()
                logger.error(f"[DATABASE] Update failed: {query[:100]}... Error: {e}")
                raise
            finally:
                cursor.close()

    def log_event(self, event_type: str, user_id: Optional[str] = None, 
                  message: str = "", metadata: Optional[str] = None):
        try:
            cursor = self.get_cursor()
            cursor.execute("""
                INSERT INTO system_logs (event_type, user_id, message, metadata)
                VALUES (?, ?, ?, ?)
            """, (event_type, user_id, message, metadata))
            self.commit()
        except Exception as e:
            logger.error(f"[DATABASE] Log event failed: {e}")

    def get_database_stats(self) -> Dict:
        stats = {}
        
        try:
            cursor = self.get_cursor()
            
            cursor.execute("SELECT COUNT(*) FROM memories WHERE status='active'")
            stats['active_memories'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM memories WHERE status='archived'")
            stats['archived_memories'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM schedules WHERE status='pending'")
            stats['pending_schedules'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM schedules WHERE status='executed'")
            stats['executed_schedules'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM memories")
            stats['total_users'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            stats['db_size_bytes'] = cursor.fetchone()[0]
            stats['db_size_mb'] = round(stats['db_size_bytes'] / (1024 * 1024), 2)
            
        except Exception as e:
            logger.error(f"[DATABASE] Stats error: {e}")
        
        return stats

    def vacuum(self):
        with self._lock:
            try:
                logger.info("[DATABASE] Running VACUUM...")
                cursor = self.get_cursor()
                cursor.execute("VACUUM")
                logger.info("[DATABASE] VACUUM completed")
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] VACUUM failed: {e}")

    def analyze(self):
        with self._lock:
            try:
                cursor = self.get_cursor()
                cursor.execute("ANALYZE")
                self.commit()
                logger.info("[DATABASE] ANALYZE completed")
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] ANALYZE failed: {e}")

    def optimize(self):
        self.analyze()
        logger.info("[DATABASE] Optimization completed")

    def backup(self, backup_path: str):
        with self._lock:
            try:
                import shutil
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"[DATABASE] Backup created: {backup_path}")
                return True
            except Exception as e:
                logger.error(f"[DATABASE] Backup failed: {e}")
                return False

    def integrity_check(self) -> bool:
        try:
            cursor = self.get_cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()[0]
            
            if result == "ok":
                logger.info("[DATABASE] Integrity check passed")
                return True
            else:
                logger.error(f"[DATABASE] Integrity check failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"[DATABASE] Integrity check error: {e}")
            return False

    def close(self):
        with self._lock:
            if self.conn:
                try:
                    self.conn.close()
                    logger.info("[DATABASE] Connection closed")
                except sqlite3.Error as e:
                    logger.error(f"[DATABASE] Close error: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            if hasattr(self, 'conn') and self.conn:
                self.conn.close()
        except:
            pass