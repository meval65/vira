import uuid
import json
import logging
import numpy as np
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum

from src.services.handler.canonicalizer import MemoryCanonicalizer
from src.services.handler.memory_gatekeeper import MemoryGatekeeper

from src.database import DBConnection
from src.config import (
    MemoryType,
    MAX_RETRIEVED_MEMORIES,
    DECAY_DAYS_EMOTION,
    DECAY_DAYS_GENERAL
)

logger = logging.getLogger(__name__)


class ClusterStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    MERGED = "merged"


@dataclass
class MemoryItem:
    id: str
    summary: str
    priority: float
    embedding: Optional[np.ndarray]
    use_count: int
    last_used: datetime
    memory_type: str
    cluster_id: Optional[str] = None


@dataclass
class MemoryCluster:
    id: str
    centroid: np.ndarray
    member_ids: Set[str]
    dominant_type: str
    created_at: datetime
    last_updated: datetime


class EmbeddingCache:
    def __init__(self, max_size: int = 3000, ttl: int = 900):
        self._cache = {}
        self._timestamps = {}
        self._access_count = {}
        self.max_size = max_size
        self.ttl = ttl
        self._lock = threading.Lock()
        self._last_cleanup = time.time()
        self.cleanup_interval = 300

    def get(self, key: int) -> Optional[np.ndarray]:
        with self._lock:
            if key in self._cache:
                self._timestamps[key] = time.time()
                self._access_count[key] = self._access_count.get(key, 0) + 1
                return self._cache[key]
        return None

    def set(self, key: int, value: np.ndarray):
        with self._lock:
            if len(self._cache) >= self.max_size:
                self._evict_lfu()
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._access_count[key] = 1

    def _evict_lfu(self):
        now = time.time()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self.ttl]
        
        if expired:
            for k in expired[:max(1, len(expired) // 2)]:
                self._cache.pop(k, None)
                self._timestamps.pop(k, None)
                self._access_count.pop(k, None)
        else:
            if self._access_count:
                min_key = min(self._access_count, key=self._access_count.get)
                self._cache.pop(min_key, None)
                self._timestamps.pop(min_key, None)
                self._access_count.pop(min_key, None)

    def cleanup(self):
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return
        
        with self._lock:
            expired = [k for k, ts in self._timestamps.items() if now - ts > self.ttl]
            for k in expired:
                self._cache.pop(k, None)
                self._timestamps.pop(k, None)
                self._access_count.pop(k, None)
            self._last_cleanup = now

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
            self._access_count.clear()

class EmbeddingHandler:
    def __init__(self, cache: EmbeddingCache):
        self.cache = cache
        self.expected_dim = None

    def normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 1e-10 else vec

    def validate_dimension(self, vec: np.ndarray) -> bool:
        if vec is None or len(vec) == 0:
            return False
        if self.expected_dim is None:
            self.expected_dim = len(vec)
            return True
        return len(vec) == self.expected_dim

    def parse(self, data: any) -> Optional[np.ndarray]:
        if data is None:
            return None

        cache_key = None
        if isinstance(data, bytes):
            cache_key = hash(data)
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        try:
            vec = None
            if isinstance(data, bytes):
                vec = np.frombuffer(data, dtype=np.float32)
            elif isinstance(data, str):
                vec = np.array(json.loads(data), dtype=np.float32)
            elif isinstance(data, (list, np.ndarray)):
                vec = np.array(data, dtype=np.float32)

            if vec is not None and self.validate_dimension(vec):
                normalized = self.normalize(vec)
                if cache_key:
                    self.cache.set(cache_key, normalized)
                return normalized
        except Exception as e:
            logger.debug(f"[EMB-PARSE] Failed to parse embedding: {e}")
        return None

    def prepare(self, embedding: List[float]) -> Optional[Tuple[np.ndarray, bytes]]:
        try:
            vec = np.array(embedding, dtype=np.float32)
            if self.validate_dimension(vec):
                normalized = self.normalize(vec)
                return normalized, normalized.tobytes()
        except Exception as e:
            logger.error(f"[EMB-PREPARE] {e}")
        return None

    def compute_similarity_matrix(self, vectors: List[np.ndarray], query: np.ndarray) -> np.ndarray:
        if not vectors:
            return np.array([])
        try:
            matrix = np.array(vectors, dtype=np.float32)
            return np.dot(matrix, query)
        except Exception as e:
            logger.error(f"[EMB-SIMILARITY] {e}")
            return np.zeros(len(vectors), dtype=np.float32)


class MemoryScorer:
    def __init__(self):
        self.recency_halflife = 14.0
        self.weights = {
            'semantic': 0.65,
            'priority': 0.15,
            'recency': 0.15,
            'usage': 0.05
        }
        self.metadata_weights = {
            'priority': 0.5,
            'recency': 0.3,
            'usage': 0.2
        }
        self.no_emb_penalty = 0.4
        self.cluster_boost = 0.1

    def calculate(self, items: List[MemoryItem], query_emb: Optional[np.ndarray], emb_handler: EmbeddingHandler, cluster_map: Dict[str, str]) -> np.ndarray:
        if not items:
            return np.array([])
        
        if query_emb is not None and any(item.embedding is not None for item in items):
            return self._semantic_scoring(items, query_emb, emb_handler, cluster_map)
        return self._metadata_scoring(items)

    def _semantic_scoring(self, items: List[MemoryItem], query: np.ndarray,
                        handler: EmbeddingHandler, cluster_map: Dict[str, str]) -> np.ndarray:
        count = len(items)
        sim_scores = np.zeros(count, dtype=np.float32)
        
        vectors = []
        indices = []
        for i, item in enumerate(items):
            if item.embedding is not None:
                vectors.append(item.embedding)
                indices.append(i)
        
        if vectors:
            similarities = handler.compute_similarity_matrix(vectors, query)
            for idx, sim_idx in enumerate(indices):
                sim_scores[sim_idx] = similarities[idx]

        recency = self._compute_recency([item.last_used for item in items])
        priority = np.array([item.priority for item in items], dtype=np.float32)
        usage = np.log1p(np.array([item.use_count for item in items], dtype=np.float32)) / 5.0

        scores = (
            sim_scores * self.weights['semantic'] +
            priority * self.weights['priority'] +
            recency * self.weights['recency'] +
            usage * self.weights['usage']
        )

        for i, item in enumerate(items):
            if item.embedding is None:
                scores[i] *= self.no_emb_penalty
            if item.cluster_id and item.cluster_id in cluster_map:
                scores[i] *= (1 + self.cluster_boost)

        return scores

    def _metadata_scoring(self, items: List[MemoryItem]) -> np.ndarray:
        recency = self._compute_recency([item.last_used for item in items])
        priority = np.array([item.priority for item in items], dtype=np.float32)
        usage = np.log1p(np.array([item.use_count for item in items], dtype=np.float32)) / 5.0

        return (
            priority * self.metadata_weights['priority'] +
            recency * self.metadata_weights['recency'] +
            usage * self.metadata_weights['usage']
        )

    def _compute_recency(self, timestamps: List[datetime]) -> np.ndarray:
        if not timestamps:
            return np.array([], dtype=np.float32)

        try:
            ts_array = np.array(timestamps, dtype='datetime64[s]')
            now = np.datetime64(datetime.now(), 's')
            
            delta_seconds = (now - ts_array).astype('timedelta64[s]').astype(np.float32)
            delta_days = np.maximum(delta_seconds, 0) / 86400.0
            
            return np.exp(-delta_days / self.recency_halflife).astype(np.float32)
            
        except Exception as e:
            logger.debug(f"[RECENCY] Calculation error: {e}")
            return np.zeros(len(timestamps), dtype=np.float32)


class ClusterManager:
    def __init__(self, similarity_threshold: float = 0.85, min_cluster_size: int = 3):
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        self.clusters: Dict[str, MemoryCluster] = {}
        self._lock = threading.Lock()

    def auto_cluster(self, items: List[MemoryItem]) -> Dict[str, str]:
        with self._lock:
            valid_items = [item for item in items if item.embedding is not None]
            if len(valid_items) < self.min_cluster_size:
                return {}

            try:
                embeddings = np.array([item.embedding for item in valid_items], dtype=np.float32)
                similarity_matrix = np.dot(embeddings, embeddings.T)
                
                assigned = {}
                for i, item in enumerate(valid_items):
                    if item.id in assigned:
                        continue

                    similar_indices = np.where(similarity_matrix[i] > self.similarity_threshold)[0]
                    
                    if len(similar_indices) >= self.min_cluster_size:
                        cluster_items = [valid_items[j] for j in similar_indices]
                        cluster_id = self._create_cluster(cluster_items)
                        
                        for j in similar_indices:
                            assigned[valid_items[j].id] = cluster_id

                return assigned
            except Exception as e:
                logger.error(f"[CLUSTER] Auto-clustering failed: {e}")
                return {}

    def _create_cluster(self, items: List[MemoryItem]) -> str:
        cluster_id = str(uuid.uuid4())
        
        centroid = np.mean([item.embedding for item in items], axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 1e-10:
            centroid = centroid / norm
        
        type_counts = defaultdict(int)
        for item in items:
            type_counts[item.memory_type] += 1
        dominant_type = max(type_counts, key=type_counts.get)
        
        cluster = MemoryCluster(
            id=cluster_id,
            centroid=centroid,
            member_ids={item.id for item in items},
            dominant_type=dominant_type,
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        self.clusters[cluster_id] = cluster
        return cluster_id

    def get_cluster_summary(self, cluster_id: str, items: List[MemoryItem]) -> Optional[str]:
        if cluster_id not in self.clusters:
            return None
        
        cluster = self.clusters[cluster_id]
        cluster_items = [item for item in items if item.id in cluster.member_ids]
        
        if not cluster_items:
            return None
        
        sorted_items = sorted(cluster_items, key=lambda x: x.priority, reverse=True)
        top_summaries = [item.summary for item in sorted_items[:3]]
        
        return f"[Cluster: {cluster.dominant_type}] " + " | ".join(top_summaries)

class MemoryManager:
    def __init__(self, db: DBConnection, genai_client=None, tier_2_model=None):
        self.db = db
        self.similarity_threshold = 0.92
        self._lock = threading.RLock()
        
        self.cache = EmbeddingCache()
        self.emb_handler = EmbeddingHandler(self.cache)
        self.scorer = MemoryScorer()
        self.cluster_mgr = ClusterManager()
        
        if genai_client and tier_2_model:
            self.canonicalizer = MemoryCanonicalizer(genai_client, tier_2_model)
            self.gatekeeper = MemoryGatekeeper(db, self.emb_handler)
            self.use_canonicalization = True
        else:
            self.canonicalizer = None
            self.gatekeeper = None
            self.use_canonicalization = False
        
        self._stats_cache = {}
        self._batch_queue = defaultdict(list)
        self._batch_lock = threading.Lock()

    def add_memory(self, user_id: str, summary: str, m_type: str, priority: float = 0.5, embedding: List[float] = None):
        summary = summary.strip()
        if not summary or len(summary) < 3:
            logger.debug(f"[MEMORY-ADD] Invalid summary: too short")
            return

        priority = max(0.0, min(1.0, priority))
        
        with self._lock:
            vec_data = None
            embedding_blob = None
            
            if embedding:
                result = self.emb_handler.prepare(embedding)
                if result:
                    vec_data, embedding_blob = result
            
            if self.use_canonicalization and self.canonicalizer and self.gatekeeper:
                self._add_memory_with_canonicalization(
                    user_id, summary, m_type, priority, vec_data, embedding_blob
                )
            else:
                self._add_memory_legacy(
                    user_id, summary, m_type, priority, vec_data, embedding_blob
                )

    def _add_memory_with_canonicalization(self, user_id: str, summary: str, 
                                         m_type: str, priority: float,
                                         vec_data: Optional[np.ndarray],
                                         embedding_blob: Optional[bytes]):
        try:
            canonical_data = self.canonicalizer.canonicalize(summary, m_type, priority)
            
            gate_result = self.gatekeeper.validate_and_gate(user_id, canonical_data, vec_data)
            
            action = gate_result.get("action")
            
            if action == "reject":
                logger.info(f"[MEMORY-GATE] Rejected: {gate_result.get('reason')}")
                return
            
            elif action == "merge":
                existing = gate_result.get("existing")
                if existing:
                    self._merge_memories(existing, canonical_data, embedding_blob)
                return
            
            elif action == "resurrect":
                archived = gate_result.get("archived")
                if archived:
                    self.gatekeeper.resurrect_memory(archived["id"], canonical_data)
                return
            
            elif action == "create":
                self._create_canonical_memory(
                    user_id, canonical_data, embedding_blob
                )
        
        except Exception as e:
            logger.error(f"[MEMORY-CANONICAL] Failed: {e}, falling back to legacy")
            self._add_memory_legacy(user_id, summary, m_type, priority, vec_data, embedding_blob)

    def _create_canonical_memory(self, user_id: str, canonical_data: Dict, 
                                embedding_blob: Optional[bytes]):
        import json
        
        mem_id = str(uuid.uuid4())
        cursor = self.db.get_cursor()
        
        try:
            metadata_str = json.dumps(canonical_data)
            
            cursor.execute("""
                INSERT INTO memories (
                    id, user_id, summary, memory_type, priority,
                    last_used_at, use_count, status, embedding, 
                    created_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?)
            """, (
                mem_id, 
                user_id, 
                canonical_data.get("summary"),
                canonical_data.get("type", "general"),
                canonical_data.get("priority", 0.5),
                datetime.now(),
                embedding_blob,
                datetime.now(),
                metadata_str
            ))
            
            self.db.commit()
            self._invalidate_cache(user_id)
            logger.info(f"[MEMORY-CANONICAL] Created: {canonical_data.get('fingerprint')}")
        
        except Exception as e:
            logger.error(f"[MEMORY-CANONICAL] Insert failed: {e}")
            self.db.rollback()

    def _merge_memories(self, existing: Dict, new_canonical: Dict, 
                       embedding_blob: Optional[bytes]):
        import json
        
        try:
            merged = self.canonicalizer.merge_canonical_memories(
                existing.get("metadata", {}), 
                new_canonical
            )
            
            metadata_str = json.dumps(merged)
            cursor = self.db.get_cursor()
            
            update_embedding = embedding_blob if embedding_blob else existing.get("embedding")
            
            cursor.execute("""
                UPDATE memories 
                SET metadata=?, 
                    priority=?, 
                    summary=?,
                    embedding=?,
                    last_used_at=?,
                    use_count=use_count+1
                WHERE id=?
            """, (
                metadata_str,
                merged.get("priority", 0.5),
                merged.get("summary"),
                update_embedding,
                datetime.now(),
                existing["id"]
            ))
            
            self.db.commit()
            logger.info(f"[MEMORY-MERGE] Updated: {merged.get('fingerprint')}")
        
        except Exception as e:
            logger.error(f"[MEMORY-MERGE] Failed: {e}")
            self.db.rollback()

    def _add_memory_legacy(self, user_id: str, summary: str, m_type: str, 
                          priority: float, vec_data: Optional[np.ndarray],
                          embedding_blob: Optional[bytes]):
        if vec_data is not None:
            if self._check_duplicate(user_id, vec_data, summary):
                logger.info(f"[MEMORY-ADD] Duplicate detected, skipping")
                return
        
        mem_id = str(uuid.uuid4())
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute("""
                INSERT INTO memories (id, user_id, summary, memory_type, priority, 
                                    last_used_at, use_count, status, embedding, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 0, 'active', ?, ?)
            """, (mem_id, user_id, summary, m_type, priority, datetime.now(), embedding_blob, datetime.now()))
            self.db.commit()
            self._invalidate_cache(user_id)
            logger.info(f"[MEMORY-ADD] Added memory for user {user_id}: {summary[:50]}")
        except Exception as e:
            logger.error(f"[MEMORY-ADD] Failed to add memory: {e}")
            self.db.rollback()

    def _check_duplicate(self, user_id: str, vector: np.ndarray, summary: str) -> bool:
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute(
                "SELECT 1 FROM memories WHERE user_id=? AND lower(summary)=? AND status='active' LIMIT 1",
                (user_id, summary.lower())
            )
            if cursor.fetchone():
                return True
        except Exception as e:
            logger.error(f"[DUP-CHECK] Exact match check failed: {e}")

        try:
            cursor.execute("""
                SELECT id, embedding FROM memories 
                WHERE user_id=? AND status='active' AND embedding IS NOT NULL 
                ORDER BY last_used_at DESC LIMIT 150
            """, (user_id,))
            
            rows = cursor.fetchall()
            if not rows:
                return False

            embeddings = []
            ids = []
            
            for r_id, r_emb in rows:
                vec = self.emb_handler.parse(r_emb)
                if vec is not None:
                    embeddings.append(vec)
                    ids.append(r_id)

            if not embeddings:
                return False

            similarities = self.emb_handler.compute_similarity_matrix(embeddings, vector)
            max_sim = np.max(similarities)
            
            if max_sim > self.similarity_threshold:
                best_idx = np.argmax(similarities)
                self._mark_used([ids[best_idx]])
                logger.info(f"[DUP-CHECK] Similar memory found (sim={max_sim:.3f})")
                return True
        except Exception as e:
            logger.error(f"[DUP-CHECK] Similarity check failed: {e}")

        return False

    def get_relevant_memories(self, user_id: str, query_embedding: List[float] = None,
                            memory_type: str = None, min_priority: float = 0.0,
                            max_results: int = None, use_clusters: bool = True,
                            query_text: str = None, entities: List[str] = None) -> List[Dict]:
        max_results = max_results or MAX_RETRIEVED_MEMORIES
        self.cache.cleanup()
        
        results = []
        fingerprint_matches = []
        entity_matches = []
        
        if self.use_canonicalization and query_text:
            fingerprint_matches = self._search_by_fingerprint_extraction(
                user_id, query_text, max_results
            )
            
            if fingerprint_matches:
                logger.info(f"[MEMORY-SEARCH] Found {len(fingerprint_matches)} fingerprint matches")
                results.extend(fingerprint_matches)
        
        if entities and len(results) < max_results:
            entity_matches = self._search_by_entities(
                user_id, entities, max_results - len(results)
            )
            
            if entity_matches:
                logger.info(f"[MEMORY-SEARCH] Found {len(entity_matches)} entity matches")
                
                existing_ids = {r['id'] for r in results}
                for em in entity_matches:
                    if em['id'] not in existing_ids:
                        results.append(em)
        
        remaining_slots = max_results - len(results)
        if remaining_slots > 0:
            exclude_ids = [r['id'] for r in results]
            
            semantic_results = self._search_by_semantic(
                user_id, query_embedding, memory_type, 
                min_priority, remaining_slots, use_clusters,
                exclude_ids=exclude_ids
            )
            
            results.extend(semantic_results)
        
        return results[:max_results]

    def _search_by_entities(self, user_id: str, entities: List[str], limit: int) -> List[Dict]:
        """Search memories by multiple entities in a single query"""
        if not entities:
            return []
        
        cursor = self.db.get_cursor()
        results = []
        
        try:
            # Prepare placeholders for IN clause
            clean_entities = [e.lower() for e in entities]
            placeholders = ','.join(['?'] * len(clean_entities))
            
            query = f"""
                SELECT id, summary, memory_type, priority, embedding,
                    use_count, last_used_at, metadata, fingerprint,
                    entity, relation, value, confidence, source_count,
                    cluster_id
                FROM memories
                WHERE user_id=? AND entity IN ({placeholders}) AND status='active'
                ORDER BY confidence DESC, source_count DESC
                LIMIT ?
            """
            
            params = [user_id] + clean_entities + [limit]
            cursor.execute(query, params)
            
            ids_to_mark = []
            
            for row in cursor.fetchall():
                result = {
                    'id': row[0],
                    'summary': row[1],
                    'type': row[2],
                    'priority': row[3],
                    'use_count': row[5],
                    'last_used': row[6],
                    'score': row[12] or 0.7,
                    'cluster_id': row[14],
                    'match_type': 'entity_match',
                    'fingerprint': row[8],
                    'entity': row[9],
                    'relation': row[10],
                    'value': row[11],
                    'confidence': row[12],
                    'source_count': row[13]
                }
                results.append(result)
                ids_to_mark.append(row[0])
            
            if ids_to_mark:
                # Run sync (SQLite is fast) or use existing executor if available.
                # Avoid spawning raw threads for small updates.
                self._mark_used(ids_to_mark)
            
            return results
        
        except Exception as e:
            logger.error(f"[ENTITY-SEARCH] Failed: {e}")
            return []
    
    def _search_by_fingerprint_extraction(self, user_id: str, 
                                         query_text: str, 
                                         max_results: int) -> List[Dict]:
        if not self.canonicalizer:
            return []
        
        try:
            canonical_query = self.canonicalizer.canonicalize(
                query_text, "query", 0.5
            )
            
            fingerprint = canonical_query.get("fingerprint")
            entity = canonical_query.get("entity")
            relation = canonical_query.get("relation")
            
            if not fingerprint:
                return []
            
            exact_match = self._search_exact_fingerprint(user_id, fingerprint)
            if exact_match:
                return [exact_match]
            
            entity_matches = self._search_by_entity_relation(
                user_id, entity, relation, max_results
            )
            
            return entity_matches
            
        except Exception as e:
            logger.error(f"[FINGERPRINT-SEARCH] Failed: {e}")
            return []
    
    def _search_exact_fingerprint(self, user_id: str, fingerprint: str) -> Optional[Dict]:
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute("""
                SELECT id, summary, memory_type, priority, embedding,
                       use_count, last_used_at, metadata, fingerprint,
                       entity, relation, value, confidence, source_count,
                       cluster_id
                FROM memories
                WHERE user_id=? AND fingerprint=? AND status='active'
                LIMIT 1
            """, (user_id, fingerprint))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            embedding = self.emb_handler.parse(row[4])
            
            result = {
                'id': row[0],
                'summary': row[1],
                'type': row[2],
                'priority': row[3],
                'use_count': row[5],
                'last_used': row[6],
                'score': 1.0,
                'cluster_id': row[14],
                'match_type': 'fingerprint_exact',
                'fingerprint': row[8],
                'entity': row[9],
                'relation': row[10],
                'value': row[11],
                'confidence': row[12],
                'source_count': row[13]
            }
            
            threading.Thread(
                target=self._mark_used, 
                args=([row[0]],), 
                daemon=True
            ).start()
            
            return result
            
        except Exception as e:
            logger.error(f"[EXACT-FINGERPRINT] Search failed: {e}")
            return None
    
    def _search_by_entity_relation(self, user_id: str, entity: str, 
                                   relation: str, limit: int) -> List[Dict]:
        cursor = self.db.get_cursor()
        results = []
        
        try:
            if entity and relation:
                cursor.execute("""
                    SELECT id, summary, memory_type, priority, embedding,
                           use_count, last_used_at, metadata, fingerprint,
                           entity, relation, value, confidence, source_count,
                           cluster_id
                    FROM memories
                    WHERE user_id=? AND entity=? AND relation=? AND status='active'
                    ORDER BY confidence DESC, source_count DESC
                    LIMIT ?
                """, (user_id, entity, relation, limit))
            
            elif entity:
                cursor.execute("""
                    SELECT id, summary, memory_type, priority, embedding,
                           use_count, last_used_at, metadata, fingerprint,
                           entity, relation, value, confidence, source_count,
                           cluster_id
                    FROM memories
                    WHERE user_id=? AND entity=? AND status='active'
                    ORDER BY confidence DESC, source_count DESC
                    LIMIT ?
                """, (user_id, entity, limit))
            
            elif relation:
                cursor.execute("""
                    SELECT id, summary, memory_type, priority, embedding,
                           use_count, last_used_at, metadata, fingerprint,
                           entity, relation, value, confidence, source_count,
                           cluster_id
                    FROM memories
                    WHERE user_id=? AND relation=? AND status='active'
                    ORDER BY confidence DESC, source_count DESC
                    LIMIT ?
                """, (user_id, relation, limit))
            else:
                return []
            
            ids_to_mark = []
            
            for row in cursor.fetchall():
                result = {
                    'id': row[0],
                    'summary': row[1],
                    'type': row[2],
                    'priority': row[3],
                    'use_count': row[5],
                    'last_used': row[6],
                    'score': row[12],
                    'cluster_id': row[14],
                    'match_type': 'entity_relation',
                    'fingerprint': row[8],
                    'entity': row[9],
                    'relation': row[10],
                    'value': row[11],
                    'confidence': row[12],
                    'source_count': row[13]
                }
                results.append(result)
                ids_to_mark.append(row[0])
            
            if ids_to_mark:
                threading.Thread(
                    target=self._mark_used, 
                    args=(ids_to_mark,), 
                    daemon=True
                ).start()
            
            return results
            
        except Exception as e:
            logger.error(f"[ENTITY-RELATION] Search failed: {e}")
            return []
    
    def _search_by_semantic(self, user_id: str, query_embedding: List[float],
                           memory_type: str, min_priority: float,
                           max_results: int, use_clusters: bool,
                           exclude_ids: List[str] = None) -> List[Dict]:
        exclude_ids = exclude_ids or []
        
        items = self._fetch_memories(user_id, memory_type, min_priority)
        
        if exclude_ids:
            items = [item for item in items if item.id not in exclude_ids]
        
        if not items:
            return []
        
        cluster_map = {}
        if use_clusters and query_embedding:
            cluster_map = self.cluster_mgr.auto_cluster(items)
        
        query_vec = None
        if query_embedding:
            result = self.emb_handler.prepare(query_embedding)
            if result:
                query_vec, _ = result
        
        scores = self.scorer.calculate(items, query_vec, self.emb_handler, cluster_map)
        
        results = self._select_top_memories(
            items, scores, max_results, query_vec is not None, cluster_map
        )
        
        for result in results:
            result['match_type'] = 'semantic'
        
        return results

    def _fetch_memories(self, user_id: str, memory_type: Optional[str],
                        min_priority: float) -> List[MemoryItem]:
        query = """
            SELECT id, summary, priority, embedding, use_count, last_used_at, memory_type, cluster_id
            FROM memories WHERE user_id = ? AND status = 'active'
        """
        params = [user_id]
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        if min_priority > 0:
            query += " AND priority >= ?"
            params.append(min_priority)

        cursor = self.db.get_cursor()
        try:
            cursor.execute(query, params)
            
            items = []
            for row in cursor.fetchall():
                embedding = self.emb_handler.parse(row[3])
                items.append(MemoryItem(
                    id=row[0],
                    summary=row[1],
                    priority=row[2],
                    embedding=embedding,
                    use_count=row[4],
                    last_used=row[5],
                    memory_type=row[6],
                    cluster_id=row[7] if len(row) > 7 else None
                ))
            
            return items
        except Exception as e:
            logger.error(f"[FETCH-MEM] {e}")
            return []

    def _select_top_memories(self, items: List[MemoryItem], scores: np.ndarray,
                            max_results: int, is_semantic: bool,
                            cluster_map: Dict[str, str]) -> List[Dict]:
        if len(scores) == 0:
            return []
        
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        ids_to_update = []
        type_counts = defaultdict(int)
        cluster_counts = defaultdict(int)
        char_count = 0
        max_chars = 4000
        max_per_type = 5
        max_per_cluster = 3
        
        for idx in sorted_indices:
            if len(results) >= max_results:
                break
            
            score = float(scores[idx])
            if is_semantic and score < 0.3:
                continue

            item = items[idx]
            
            if char_count + len(item.summary) > max_chars:
                continue
            
            if type_counts[item.memory_type] >= max_per_type:
                continue
            
            if item.cluster_id and cluster_counts[item.cluster_id] >= max_per_cluster:
                continue

            results.append({
                'id': item.id,
                'summary': item.summary,
                'type': item.memory_type,
                'priority': item.priority,
                'use_count': item.use_count,
                'last_used': item.last_used,
                'score': score,
                'cluster_id': item.cluster_id
            })
            
            ids_to_update.append(item.id)
            type_counts[item.memory_type] += 1
            if item.cluster_id:
                cluster_counts[item.cluster_id] += 1
            char_count += len(item.summary)

        if ids_to_update:
            threading.Thread(target=self._mark_used, args=(ids_to_update,), daemon=True).start()

        return results

    def _mark_used(self, memory_ids: List[str]):
        """Batched update of memory usage stats"""
        if not memory_ids:
            return
        
        try:
            cursor = self.db.get_cursor()
            now = datetime.now()
            
            BATCH_SIZE = 100
            for i in range(0, len(memory_ids), BATCH_SIZE):
                chunk = memory_ids[i:i + BATCH_SIZE]
                placeholders = ','.join(['?'] * len(chunk))
                
                cursor.execute(
                    f"""UPDATE memories 
                        SET use_count = use_count + 1, 
                            last_used_at = ? 
                        WHERE id IN ({placeholders})""",
                    [now] + chunk
                )
            
            self.db.commit()
        except Exception as e:
            logger.error(f"[MARK-USED] Failed to update stats: {e}")

    def batch_add_memories(self, user_id: str, memories: List[Dict]):
        with self._batch_lock:
            cursor = self.db.get_cursor()
            now = datetime.now()
            
            try:
                added_count = 0
                for mem in memories:
                    mem_id = str(uuid.uuid4())
                    summary = mem.get('summary', '').strip()
                    
                    if not summary or len(summary) < 3:
                        continue
                    
                    priority = max(0.0, min(1.0, mem.get('priority', 0.5)))
                    m_type = mem.get('type', MemoryType.GENERAL.value)
                    embedding_blob = None
                    
                    if 'embedding' in mem:
                        result = self.emb_handler.prepare(mem['embedding'])
                        if result:
                            _, embedding_blob = result
                    
                    cursor.execute("""
                        INSERT INTO memories (id, user_id, summary, memory_type, priority,
                                            last_used_at, use_count, status, embedding, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, 0, 'active', ?, ?)
                    """, (mem_id, user_id, summary, m_type, priority, now, embedding_blob, now))
                    added_count += 1
                
                self.db.commit()
                self._invalidate_cache(user_id)
                logger.info(f"[BATCH-ADD] Added {added_count} memories for user {user_id}")
            except Exception as e:
                logger.error(f"[BATCH-ADD] {e}")
                self.db.rollback()

    def wipe_all_memories(self, user_id: str) -> int:
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                cursor.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
                count = cursor.rowcount
                self.db.commit()
                self._invalidate_cache(user_id)
                logger.info(f"[WIPE] Deleted {count} memories for user {user_id}")
                return count
            except Exception as e:
                logger.error(f"[WIPE] {e}")
                return 0

    def forget_memory(self, user_id: str, query: str, embedding: List[float] = None) -> Optional[str]:
        with self._lock:
            query_vec = None
            if embedding:
                res = self.emb_handler.prepare(embedding)
                if res:
                    query_vec, _ = res
            
            if query_vec is None:
                logger.warning(f"[MEMORY-FORGET] No embedding provided for query: {query}")
                return None

            cursor = self.db.get_cursor()
            try:
                cursor.execute("""
                    SELECT id, summary, embedding FROM memories 
                    WHERE user_id=? AND status='active' AND embedding IS NOT NULL
                """, (user_id,))
                
                rows = cursor.fetchall()
                if not rows:
                    return None

                mem_ids = []
                mem_vecs = []
                mem_summaries = []

                for r in rows:
                    vec = self.emb_handler.parse(r[2])
                    if vec is not None:
                        mem_ids.append(r[0])
                        mem_summaries.append(r[1])
                        mem_vecs.append(vec)

                if not mem_vecs:
                    return None

                similarity_threshold = 0.82
                
                similarities = self.emb_handler.compute_similarity_matrix(mem_vecs, query_vec)
                best_idx = np.argmax(similarities)
                max_sim = similarities[best_idx]

                if max_sim >= similarity_threshold:
                    target_id = mem_ids[best_idx]
                    target_summary = mem_summaries[best_idx]
                    
                    cursor.execute("""
                        UPDATE memories 
                        SET status='archived', priority=0, last_used_at=? 
                        WHERE id=?
                    """, (datetime.now(), target_id))
                    
                    self.db.commit()
                    self._invalidate_cache(user_id)
                    logger.info(f"[MEMORY-FORGET] User {user_id} forgot: {target_summary[:50]} (Sim: {max_sim:.2f})")
                    
                    return target_summary
                
                return None
            except Exception as e:
                logger.error(f"[MEMORY-FORGET] {e}")
                return None

    def deduplicate_existing_memories(self, user_id: str) -> int:
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                cursor.execute("""
                    SELECT id, embedding, summary FROM memories 
                    WHERE user_id=? AND status='active' AND embedding IS NOT NULL
                    ORDER BY created_at ASC
                """, (user_id,))
                
                rows = cursor.fetchall()
                if len(rows) < 2:
                    return 0
                
                to_remove = set()
                
                for i in range(len(rows)):
                    if rows[i][0] in to_remove:
                        continue
                        
                    vec_i = self.emb_handler.parse(rows[i][1])
                    if vec_i is None:
                        continue
                    
                    for j in range(i + 1, len(rows)):
                        if rows[j][0] in to_remove:
                            continue
                        
                        vec_j = self.emb_handler.parse(rows[j][1])
                        if vec_j is None:
                            continue
                        
                        similarity = np.dot(vec_i, vec_j)
                        if similarity > self.similarity_threshold:
                            to_remove.add(rows[j][0])
                
                if to_remove:
                    placeholders = ','.join(['?'] * len(to_remove))
                    cursor.execute(
                        f"UPDATE memories SET status='archived' WHERE id IN ({placeholders})",
                        list(to_remove)
                    )
                    self.db.commit()
                    self._invalidate_cache(user_id)
                    logger.info(f"[DEDUP] Removed {len(to_remove)} duplicates for user {user_id}")
                    return len(to_remove)
                
                return 0
            except Exception as e:
                logger.error(f"[DEDUP] {e}")
                return 0

    def apply_decay_rules(self, user_id: str = None):
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                now = datetime.now()
                
                cutoff_emotion = now - timedelta(days=DECAY_DAYS_EMOTION)
                cutoff_general = now - timedelta(days=DECAY_DAYS_GENERAL)

                query = """
                    UPDATE memories SET status='archived' 
                    WHERE status='active' AND (
                        (memory_type=? AND last_used_at < ?) OR
                        (memory_type!=? AND last_used_at < ?)
                    )
                """
                params = [MemoryType.EMOTION.value, cutoff_emotion, MemoryType.EMOTION.value, cutoff_general]
                
                if user_id:
                    query += " AND user_id=?"
                    params.append(user_id)
                
                cursor.execute(query, params)
                
                if cursor.rowcount > 0:
                    self.db.commit()
                    logger.info(f"[DECAY] Archived {cursor.rowcount} memories")
                    if user_id:
                        self._invalidate_cache(user_id)
            except Exception as e:
                logger.error(f"[DECAY] {e}")

    def optimize_memories(self, user_id: str = None, target_count: int = 500):
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                users = [user_id] if user_id else []
                
                if not user_id:
                    cursor.execute("""
                        SELECT user_id FROM memories 
                        WHERE status='active' 
                        GROUP BY user_id 
                        HAVING COUNT(*) > ?
                    """, (target_count,))
                    users = [r[0] for r in cursor.fetchall()]
                
                total_archived = 0
                for uid in users:
                    cursor.execute(
                        "SELECT COUNT(*) FROM memories WHERE user_id=? AND status='active'",
                        (uid,)
                    )
                    count = cursor.fetchone()[0]
                    
                    if count <= target_count:
                        continue
                    
                    excess = count - target_count
                    cursor.execute("""
                        UPDATE memories SET status='archived' WHERE id IN (
                            SELECT id FROM memories WHERE user_id=? AND status='active'
                            ORDER BY (priority * 0.3 + use_count * 0.1 + 
                                     (julianday('now') - julianday(last_used_at)) * -0.01) ASC
                            LIMIT ?
                        )
                    """, (uid, excess))
                    total_archived += cursor.rowcount
                
                if total_archived > 0:
                    self.db.commit()
                    if user_id:
                        self._invalidate_cache(user_id)
                        
                logger.info(f"[OPTIMIZE] Archived {total_archived} memories")
            except Exception as e:
                logger.error(f"[OPTIMIZE] {e}")

    def search_memories(self, user_id: str, keyword: str, limit: int = 10) -> List[Dict]:
        try:
            cursor = self.db.get_cursor()
            cursor.execute("""
                SELECT summary, memory_type, priority, use_count, last_used_at
                FROM memories 
                WHERE user_id=? AND status='active' AND lower(summary) LIKE ?
                ORDER BY priority DESC, use_count DESC
                LIMIT ?
            """, (user_id, f"%{keyword.lower()}%", limit))
            
            return [
                {
                    "summary": r[0],
                    "type": r[1],
                    "priority": r[2],
                    "use_count": r[3],
                    "last_used": r[4]
                }
                for r in cursor.fetchall()
            ]
        except Exception as e:
            logger.error(f"[SEARCH-MEM] {e}")
            return []

    def get_memory_stats(self, user_id: str) -> Dict:
        cache_key = f"stats_{user_id}"
        if cache_key in self._stats_cache:
            return self._stats_cache[cache_key]
        
        try:
            cursor = self.db.get_cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status='archived' THEN 1 ELSE 0 END) as archived,
                    AVG(CASE WHEN status='active' THEN priority ELSE NULL END) as avg_priority,
                    MAX(last_used_at) as last_activity
                FROM memories WHERE user_id=?
            """, (user_id,))
            
            row = cursor.fetchone()
            stats = {
                "total": row[0] or 0,
                "active": row[1] or 0,
                "archived": row[2] or 0,
                "avg_priority": round(row[3], 2) if row[3] else 0.0,
                "last_activity": row[4]
            }
            
            self._stats_cache[cache_key] = stats
            return stats
        except Exception as e:
            logger.error(f"[STATS] {e}")
            return {"total": 0, "active": 0, "archived": 0, "avg_priority": 0.0}

    def _invalidate_cache(self, user_id: str):
        self._stats_cache.pop(f"stats_{user_id}", None)