import logging
import numpy as np
from typing import Dict, Optional, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MemoryGatekeeper:
    SIMILARITY_THRESHOLD = 0.90
    ARCHIVE_CHECK_DAYS = 90
    MIN_CONFIDENCE = 0.3
    
    def __init__(self, db, emb_handler):
        self.db = db
        self.emb_handler = emb_handler
    
    def _check_fingerprint_collision(self, user_id: str, fingerprint: str) -> Optional[Dict]:
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute("""
                SELECT id, summary, metadata, status, embedding, priority, use_count, last_used_at
                FROM memories 
                WHERE user_id=? AND metadata LIKE ? AND status='active'
                LIMIT 1
            """, (user_id, f'%"fingerprint":"{fingerprint}"%'))
            
            row = cursor.fetchone()
            if row:
                import json
                metadata = json.loads(row[2]) if row[2] else {}
                
                return {
                    "id": row[0],
                    "summary": row[1],
                    "metadata": metadata,
                    "status": row[3],
                    "embedding": self.emb_handler.parse(row[4]),
                    "priority": row[5],
                    "use_count": row[6],
                    "last_used_at": row[7]
                }
        except Exception as e:
            logger.error(f"Fingerprint check failed: {e}")
        
        return None
    
    def _check_archived_memories(self, user_id: str, fingerprint: str) -> Optional[Dict]:
        cursor = self.db.get_cursor()
        cutoff_date = datetime.now() - timedelta(days=self.ARCHIVE_CHECK_DAYS)
        
        try:
            cursor.execute("""
                SELECT id, summary, metadata, last_used_at
                FROM memories 
                WHERE user_id=? AND metadata LIKE ? AND status='archived'
                AND last_used_at > ?
                ORDER BY last_used_at DESC LIMIT 1
            """, (user_id, f'%"fingerprint":"{fingerprint}"%', cutoff_date))
            
            row = cursor.fetchone()
            if row:
                import json
                return {
                    "id": row[0],
                    "summary": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "last_used_at": row[3]
                }
        except Exception as e:
            logger.error(f"Archive check failed: {e}")
        
        return None
    
    def _check_semantic_similarity(self, user_id: str, embedding: np.ndarray) -> Optional[Dict]:
        if embedding is None:
            return None
        
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute("""
                SELECT id, summary, metadata, embedding, priority
                FROM memories 
                WHERE user_id=? AND status='active' AND embedding IS NOT NULL
                ORDER BY last_used_at DESC LIMIT 100
            """, (user_id,))
            
            rows = cursor.fetchall()
            if not rows:
                return None
            
            valid_rows = []
            embedding_matrix = []
            
            for row in rows:
                vec = self.emb_handler.parse(row[3])
                if vec is not None:
                    embedding_matrix.append(vec)
                    valid_rows.append(row)
            
            if not embedding_matrix:
                return None
            
            embedding_matrix = np.array(embedding_matrix) # Shape: (N, Dim)
            similarities = np.dot(embedding_matrix, embedding) # Shape: (N,)
            
            max_idx = np.argmax(similarities)
            max_sim = similarities[max_idx]
            
            if max_sim > self.SIMILARITY_THRESHOLD:
                best_row = valid_rows[max_idx]
                import json
                return {
                    "similarity": float(max_sim),
                    "id": best_row[0],
                    "summary": best_row[1],
                    "metadata": json.loads(best_row[2]) if best_row[2] else {},
                    "priority": best_row[4]
                }
        
        except Exception as e:
            logger.error(f"Semantic similarity check failed: {e}")
        
        return None
    
    def validate_and_gate(self, user_id: str, canonical_data: Dict, 
                        embedding: Optional[np.ndarray]) -> Dict:
        fingerprint = canonical_data.get("fingerprint")
        confidence = canonical_data.get("confidence", 0.5)
        
        if confidence < self.MIN_CONFIDENCE:
            return {
                "action": "reject",
                "reason": "confidence_too_low",
                "confidence": confidence
            }
        
        fingerprint_match = self._check_fingerprint_collision(user_id, fingerprint)
        if fingerprint_match:
            return {
                "action": "merge",
                "reason": "fingerprint_collision",
                "existing": fingerprint_match
            }
        
        archived = self._check_archived_memories(user_id, fingerprint)
        if archived:
            return {
                "action": "resurrect",
                "reason": "found_in_archive",
                "archived": archived
            }
        
        if embedding is not None:
            semantic_match = self._check_semantic_similarity(user_id, embedding)
            if semantic_match:
                return {
                    "action": "merge",
                    "reason": "semantic_similarity",
                    "existing": semantic_match
                }
        
        return {
            "action": "create",
            "reason": "new_memory"
        }
    
    def resurrect_memory(self, memory_id: str, new_canonical: Dict) -> bool:
        try:
            cursor = self.db.get_cursor()
            import json
            
            metadata_str = json.dumps(new_canonical)
            
            cursor.execute("""
                UPDATE memories 
                SET status='active', 
                    priority=?, 
                    metadata=?,
                    last_used_at=?,
                    use_count=use_count+1
                WHERE id=?
            """, (
                new_canonical.get("priority", 0.5),
                metadata_str,
                datetime.now(),
                memory_id
            ))
            
            self.db.commit()
            logger.info(f"Memory resurrected: {memory_id}")
            return True
        
        except Exception as e:
            logger.error(f"Resurrection failed: {e}")
            return False