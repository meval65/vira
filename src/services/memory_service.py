import uuid
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Any, Optional, Dict
from functools import lru_cache
import threading
from collections import defaultdict

from src.database import DBConnection
from src.config import (
    MemoryType, 
    MAX_RETRIEVED_MEMORIES, 
    DECAY_DAYS_EMOTION, 
    DECAY_DAYS_GENERAL
)

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, db: DBConnection):
        self.db = db
        self.SIMILARITY_THRESHOLD = 0.92
        self.DEDUP_BATCH_SIZE = 500
        self._lock = threading.RLock()
        self._embedding_cache = {}
        self._stats_cache = {}
        self.MAX_CACHE_SIZE = 1000
        self.CACHE_TTL = 300

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    def _parse_embedding(self, emb_data: Any) -> Optional[np.ndarray]:
        if emb_data is None: 
            return None
        
        cache_key = None
        if isinstance(emb_data, bytes):
            cache_key = hash(emb_data)
            if cache_key in self._embedding_cache:
                return self._embedding_cache[cache_key]
        
        try:
            if isinstance(emb_data, bytes):
                vec = np.frombuffer(emb_data, dtype=np.float32)
            elif isinstance(emb_data, str):
                vec = np.array(json.loads(emb_data), dtype=np.float32)
            else:
                return None
            
            if cache_key and len(self._embedding_cache) < self.MAX_CACHE_SIZE:
                self._embedding_cache[cache_key] = vec
            
            return vec
            
        except Exception as e:
            logger.error(f"[MEM-ERR] Parse embedding failed: {e}")
            return None

    def add_memory(self, user_id: str, summary: str, m_type: str, priority: float = 0.5, embedding: List[float] = None):
        clean_summary = summary.strip()
        if not clean_summary or len(clean_summary) < 5:
            return

        priority = max(0.0, min(1.0, priority))

        with self._lock:
            cursor = self.db.get_cursor()
            
            vec_new = None
            embedding_blob = None
            
            if embedding:
                vec_new = np.array(embedding, dtype=np.float32)
                vec_new = self._normalize(vec_new)
                embedding_blob = vec_new.tobytes()

            if vec_new is not None:
                duplicate_id = self._find_semantic_duplicate(user_id, vec_new)
                if duplicate_id:
                    logger.info(f"[MEMORY-SKIP] Semantic duplicate: '{clean_summary[:30]}...'")
                    self._mark_batch_as_used([duplicate_id])
                    return

            cursor.execute(
                "SELECT id FROM memories WHERE user_id=? AND lower(summary)=? AND status='active'", 
                (user_id, clean_summary.lower())
            )
            if cursor.fetchone():
                logger.info(f"[MEMORY-SKIP] Exact text duplicate: {clean_summary[:30]}...")
                return 

            mem_id = str(uuid.uuid4())
            try:
                cursor.execute("""
                    INSERT INTO memories (id, user_id, summary, memory_type, priority, last_used_at, use_count, status, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?)
                """, (mem_id, user_id, clean_summary, m_type, priority, datetime.now(), 0, embedding_blob, datetime.now()))
                self.db.commit()
                
                self._invalidate_cache(user_id)
                
                logger.info(f"[MEMORY-STORE] [{m_type}] {clean_summary[:50]}")
            except Exception as e:
                logger.error(f"[MEMORY-ERROR] Failed to save: {e}")

    def _find_semantic_duplicate(self, user_id: str, new_vector: np.ndarray) -> Optional[str]:
        cursor = self.db.get_cursor()
        cursor.execute(
            "SELECT id, embedding FROM memories WHERE user_id=? AND status='active' ORDER BY last_used_at DESC LIMIT ?", 
            (user_id, self.DEDUP_BATCH_SIZE)
        )
        rows = cursor.fetchall()
        
        if not rows: 
            return None

        embeddings = []
        ids = []
        
        for r_id, r_emb in rows:
            vec_old = self._parse_embedding(r_emb)
            if vec_old is not None and len(vec_old) == len(new_vector):
                embeddings.append(vec_old)
                ids.append(r_id)
        
        if not embeddings:
            return None
        
        matrix = np.array(embeddings)
        scores = np.dot(matrix, new_vector)
        
        max_idx = np.argmax(scores)
        max_score = scores[max_idx]
        
        if max_score >= self.SIMILARITY_THRESHOLD:
            return ids[max_idx]
        
        return None

    def wipe_all_memories(self, user_id: str) -> int:
        with self._lock:
            cursor = self.db.get_cursor()
            try:
                cursor.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
                count = cursor.rowcount
                self.db.commit()
                
                self._invalidate_cache(user_id)
                
                logger.warning(f"[MEMORY-WIPE] Deleted {count} memories for {user_id}")
                return count
            except Exception as e:
                logger.error(f"[MEMORY-ERR] Wipe failed: {e}")
                return 0

    def get_relevant_memories(self, user_id: str, query_embedding: List[float] = None, 
                             memory_type: str = None, min_priority: float = 0.0,
                             max_results: int = None) -> List[str]:
        
        if max_results is None:
            max_results = MAX_RETRIEVED_MEMORIES
            
        cursor = self.db.get_cursor()
        
        query = """
            SELECT id, summary, priority, embedding, use_count, last_used_at, memory_type
            FROM memories 
            WHERE user_id = ? AND status = 'active'
        """
        params = [user_id]
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        
        if min_priority > 0:
            query += " AND priority >= ?"
            params.append(min_priority)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows: 
            return []

        memories_with_emb = []
        memories_no_emb = []
        all_data = []

        for i, row in enumerate(rows):
            m_id, summary, priority, raw_emb, use_count, last_used, mem_type = row
            all_data.append((m_id, summary, priority, use_count, last_used, mem_type))
            
            vec = self._parse_embedding(raw_emb)
            if vec is not None and len(vec) > 0:
                memories_with_emb.append((i, vec, priority, use_count))
            else:
                memories_no_emb.append((i, priority, use_count))

        scored_results = []

        if query_embedding and memories_with_emb:
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_vec = self._normalize(q_vec)

            vectors = [m[1] for m in memories_with_emb]
            priorities = np.array([m[2] for m in memories_with_emb])
            use_counts = np.array([m[3] for m in memories_with_emb])
            indices = [m[0] for m in memories_with_emb]
            
            matrix = np.array(vectors)
            sim_scores = np.dot(matrix, q_vec)
            
            recency_scores = self._calculate_recency_scores([all_data[i][4] for i in indices])
            use_count_scores = np.log1p(use_counts) / 10.0
            
            final_scores = (
                sim_scores * 0.60 + 
                priorities * 0.20 + 
                recency_scores * 0.15 + 
                use_count_scores * 0.05
            )

            for idx, score in zip(indices, final_scores):
                scored_results.append((score, idx))

        for idx, prio, use_count in memories_no_emb:
            recency = self._calculate_recency_score(all_data[idx][4])
            fallback_score = (prio * 0.4) + (recency * 0.3) + (np.log1p(use_count) / 10.0 * 0.1)
            scored_results.append((fallback_score, idx))

        if not query_embedding and not scored_results:
            scored_results = [(row[2] + (np.log1p(row[3]) / 10.0), i) for i, row in enumerate(all_data)]

        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        ids_to_update = []
        current_char_count = 0
        MAX_CHARS = 2500
        type_distribution = defaultdict(int)
        MAX_PER_TYPE = max(4, max_results // 3)

        for score, idx in scored_results:
            if len(results) >= max_results: 
                break
            
            m_id, summary, priority, use_count, last_used, mem_type = all_data[idx]
            
            if current_char_count + len(summary) > MAX_CHARS:
                continue
            
            if type_distribution[mem_type] >= MAX_PER_TYPE:
                continue
            
            results.append(summary)
            ids_to_update.append(m_id)
            current_char_count += len(summary)
            type_distribution[mem_type] += 1
            
        if ids_to_update:
            self._mark_batch_as_used(ids_to_update)
        
        return results

    def _calculate_recency_score(self, last_used: Any) -> float:
        if not last_used:
            return 0.0
        
        try:
            if isinstance(last_used, str):
                last_used = datetime.fromisoformat(last_used)
            
            days_ago = (datetime.now() - last_used).total_seconds() / 86400
            
            return np.exp(-days_ago / 30.0)
        except:
            return 0.0

    def _calculate_recency_scores(self, last_used_list: List[Any]) -> np.ndarray:
        scores = []
        for last_used in last_used_list:
            scores.append(self._calculate_recency_score(last_used))
        return np.array(scores)

    def _mark_batch_as_used(self, memory_ids: List[str]):
        if not memory_ids: 
            return
        
        with self._lock:
            cursor = self.db.get_cursor()
            now = datetime.now()
            placeholders = ','.join('?' * len(memory_ids))
            query = f"UPDATE memories SET use_count = use_count + 1, last_used_at = ? WHERE id IN ({placeholders})"
            
            try:
                cursor.execute(query, [now] + memory_ids)
                self.db.commit()
            except Exception as e:
                logger.error(f"[MEMORY-ERROR] Batch update failed: {e}")

    def apply_decay_rules(self, user_id: str = None):
        with self._lock:
            cursor = self.db.get_cursor()
            now = datetime.now()
            
            limit_date_emo = now - timedelta(days=DECAY_DAYS_EMOTION)
            limit_date_gen = now - timedelta(days=DECAY_DAYS_GENERAL)

            query_emo = "UPDATE memories SET status = 'archived' WHERE memory_type = ? AND last_used_at < ? AND status = 'active'"
            params_emo = [MemoryType.EMOTION.value, limit_date_emo]
            
            query_gen = "UPDATE memories SET status = 'archived' WHERE memory_type != ? AND last_used_at < ? AND status = 'active'"
            params_gen = [MemoryType.EMOTION.value, limit_date_gen]

            if user_id:
                query_emo += " AND user_id = ?"
                params_emo.append(user_id)
                query_gen += " AND user_id = ?"
                params_gen.append(user_id)

            try:
                cursor.execute(query_emo, tuple(params_emo))
                c1 = cursor.rowcount
                cursor.execute(query_gen, tuple(params_gen))
                c2 = cursor.rowcount
                
                if c1 > 0 or c2 > 0:
                    self.db.commit()
                    logger.info(f"[MAINTENANCE] Archived: {c1} emotions, {c2} general")
                    
                    if user_id:
                        self._invalidate_cache(user_id)
                        
            except Exception as e:
                logger.error(f"[MAINTENANCE ERROR] {e}")

    def optimize_memories(self, user_id: str = None, target_count: int = 500):
        with self._lock:
            cursor = self.db.get_cursor()
            
            if user_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE user_id=? AND status='active'", 
                    (user_id,)
                )
                current_count = cursor.fetchone()[0]
                
                if current_count <= target_count:
                    return
                
                to_remove = current_count - target_count
                
                cursor.execute("""
                    UPDATE memories 
                    SET status = 'archived'
                    WHERE id IN (
                        SELECT id FROM memories 
                        WHERE user_id = ? AND status = 'active'
                        ORDER BY 
                            (priority * 0.3) + 
                            (use_count * 0.3) + 
                            (julianday('now') - julianday(last_used_at)) * -0.4
                        ASC
                        LIMIT ?
                    )
                """, (user_id, to_remove))
                
                archived = cursor.rowcount
                self.db.commit()
                
                self._invalidate_cache(user_id)
                
                logger.info(f"[OPTIMIZE] Archived {archived} low-priority memories for {user_id}")
            else:
                cursor.execute("""
                    SELECT user_id, COUNT(*) as cnt 
                    FROM memories 
                    WHERE status='active' 
                    GROUP BY user_id 
                    HAVING cnt > ?
                """, (target_count,))
                
                users = cursor.fetchall()
                total_archived = 0
                
                for user_id, count in users:
                    to_remove = count - target_count
                    
                    cursor.execute("""
                        UPDATE memories 
                        SET status = 'archived'
                        WHERE id IN (
                            SELECT id FROM memories 
                            WHERE user_id = ? AND status = 'active'
                            ORDER BY 
                                (priority * 0.3) + 
                                (use_count * 0.3) + 
                                (julianday('now') - julianday(last_used_at)) * -0.4
                            ASC
                            LIMIT ?
                        )
                    """, (user_id, to_remove))
                    
                    total_archived += cursor.rowcount
                
                if total_archived > 0:
                    self.db.commit()
                    logger.info(f"[OPTIMIZE] Global: Archived {total_archived} memories across {len(users)} users")

    def get_memory_stats(self, user_id: str) -> Dict:
        cache_key = f"stats_{user_id}"
        
        if cache_key in self._stats_cache:
            cached_data, cached_time = self._stats_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self.CACHE_TTL:
                return cached_data
        
        cursor = self.db.get_cursor()
        stats = {
            "total": 0, 
            "active": 0, 
            "archived": 0, 
            "types": {},
            "avg_priority": 0.0,
            "avg_use_count": 0.0,
            "total_retrievals": 0
        }
        
        try:
            cursor.execute(
                "SELECT status, count(*) FROM memories WHERE user_id=? GROUP BY status", 
                (user_id,)
            )
            for row in cursor.fetchall():
                stats[row[0]] = row[1]
                stats["total"] += row[1]
                
            cursor.execute(
                "SELECT memory_type, count(*) FROM memories WHERE user_id=? AND status='active' GROUP BY memory_type", 
                (user_id,)
            )
            for row in cursor.fetchall():
                stats["types"][row[0]] = row[1]
            
            cursor.execute(
                "SELECT AVG(priority), AVG(use_count), SUM(use_count) FROM memories WHERE user_id=? AND status='active'", 
                (user_id,)
            )
            avgs = cursor.fetchone()
            if avgs:
                stats["avg_priority"] = round(avgs[0] or 0.0, 2)
                stats["avg_use_count"] = round(avgs[1] or 0.0, 2)
                stats["total_retrievals"] = int(avgs[2] or 0)
            
            self._stats_cache[cache_key] = (stats, datetime.now())
            
            if len(self._stats_cache) > 100:
                oldest = min(self._stats_cache.items(), key=lambda x: x[1][1])
                del self._stats_cache[oldest[0]]
                
        except Exception as e:
            logger.error(f"[STATS ERROR] {e}")
            
        return stats

    def search_memories(self, user_id: str, keyword: str, limit: int = 10, 
                       status: str = 'active') -> List[Dict]:
        cursor = self.db.get_cursor()
        
        cursor.execute("""
            SELECT summary, memory_type, priority, use_count, last_used_at, created_at
            FROM memories 
            WHERE user_id = ? AND status = ? 
            AND lower(summary) LIKE ?
            ORDER BY priority DESC, use_count DESC, last_used_at DESC
            LIMIT ?
        """, (user_id, status, f"%{keyword.lower()}%", limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "summary": row[0],
                "type": row[1],
                "priority": row[2],
                "use_count": row[3],
                "last_used": row[4],
                "created_at": row[5]
            })
        
        return results

    def update_memory_priority(self, memory_id: str, new_priority: float, user_id: str = None) -> bool:
        new_priority = max(0.0, min(1.0, new_priority))
        
        with self._lock:
            cursor = self.db.get_cursor()
            
            try:
                if user_id:
                    cursor.execute(
                        "UPDATE memories SET priority = ? WHERE id = ? AND user_id = ?",
                        (new_priority, memory_id, user_id)
                    )
                else:
                    cursor.execute(
                        "UPDATE memories SET priority = ? WHERE id = ?",
                        (new_priority, memory_id)
                    )
                
                affected = cursor.rowcount
                
                if affected > 0:
                    self.db.commit()
                    
                    if user_id:
                        self._invalidate_cache(user_id)
                    
                    return True
                    
            except Exception as e:
                logger.error(f"[PRIORITY-UPDATE] Error: {e}")
                
        return False

    def get_memory_by_type(self, user_id: str, memory_type: str, limit: int = 20) -> List[Dict]:
        cursor = self.db.get_cursor()
        
        cursor.execute("""
            SELECT id, summary, priority, use_count, last_used_at
            FROM memories
            WHERE user_id = ? AND memory_type = ? AND status = 'active'
            ORDER BY priority DESC, last_used_at DESC
            LIMIT ?
        """, (user_id, memory_type, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "summary": row[1],
                "priority": row[2],
                "use_count": row[3],
                "last_used": row[4]
            })
        
        return results

    def _invalidate_cache(self, user_id: str):
        keys_to_remove = [k for k in self._stats_cache.keys() if user_id in k]
        for key in keys_to_remove:
            del self._stats_cache[key]

    def clear_cache(self):
        self._embedding_cache.clear()
        self._stats_cache.clear()
        logger.info("[MEMORY] All caches cleared")

    def get_global_stats(self) -> Dict:
        cursor = self.db.get_cursor()
        stats = {
            "total_memories": 0,
            "total_users": 0,
            "active_memories": 0,
            "archived_memories": 0,
            "memory_types": {}
        }
        
        try:
            cursor.execute("SELECT COUNT(DISTINCT user_id) FROM memories")
            stats["total_users"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT status, COUNT(*) FROM memories GROUP BY status")
            for row in cursor.fetchall():
                if row[0] == 'active':
                    stats["active_memories"] = row[1]
                elif row[0] == 'archived':
                    stats["archived_memories"] = row[1]
                stats["total_memories"] += row[1]
            
            cursor.execute("SELECT memory_type, COUNT(*) FROM memories WHERE status='active' GROUP BY memory_type")
            for row in cursor.fetchall():
                stats["memory_types"][row[0]] = row[1]
                
        except Exception as e:
            logger.error(f"[GLOBAL-STATS] Error: {e}")
        
        return stats