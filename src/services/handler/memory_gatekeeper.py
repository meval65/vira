import logging
import json
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class MemoryConflictResolver:
    """Smart conflict resolution with LLM-powered comparison."""
    
    def __init__(self, client=None, model: str = None):
        self._conflict_history: Dict[str, List[Dict]] = defaultdict(list)
        self.client = client
        self.model = model or "models/gemma-3-4b-it"  # Small fast model
        
    def record_conflict(self, fingerprint: str, existing: Dict, new: Dict, resolution: str):
        conflict_entry = {
            'timestamp': datetime.now().isoformat(),
            'existing_value': existing.get('value'),
            'new_value': new.get('value'),
            'existing_confidence': existing.get('confidence', 0.5),
            'new_confidence': new.get('confidence', 0.5),
            'resolution': resolution
        }
        
        self._conflict_history[fingerprint].append(conflict_entry)
        
        if len(self._conflict_history[fingerprint]) > 10:
            self._conflict_history[fingerprint].pop(0)
    
    def get_conflict_count(self, fingerprint: str, days: int = 30) -> int:
        conflicts = self._conflict_history.get(fingerprint, [])
        cutoff = datetime.now() - timedelta(days=days)
        
        return sum(1 for c in conflicts 
                  if datetime.fromisoformat(c['timestamp']) > cutoff)
    
    def is_highly_conflicted(self, fingerprint: str, threshold: int = 3) -> bool:
        return self.get_conflict_count(fingerprint, days=7) >= threshold
    
    async def _llm_compare_memories(self, existing: Dict, new: Dict) -> Tuple[str, str]:
        """Use small LLM to compare conflicting memories and decide resolution."""
        if not self.client:
            return "heuristic", "no_llm_client"
        
        try:
            existing_summary = existing.get('summary') or existing.get('value', '')
            new_summary = new.get('summary') or new.get('value', '')
            
            prompt = f"""Compare these two memory statements for the same entity:

EXISTING: {existing_summary}
NEW: {new_summary}

Determine the best action:
- "update": New info supersedes old (more recent, more specific, correction)
- "merge": Both are valid, combine them
- "flag_conflict": Contradictory info, needs user clarification
- "keep": Existing is more reliable, discard new

Response format (JSON only):
{{"action": "...", "reason": "brief explanation"}}"""

            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=100
                )
            )
            
            if response.text:
                text = response.text.strip()
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    data = json.loads(text[start:end])
                    return data.get("action", "keep"), data.get("reason", "LLM decision")
        
        except Exception as e:
            logger.warning(f"[CONFLICT] LLM comparison failed: {e}")
        
        return "heuristic", "llm_failed_fallback"
    
    async def resolve_conflict(self, existing: Dict, new: Dict, fingerprint: str) -> Tuple[str, Dict]:
        """Smart conflict resolution using LLM when heuristics insufficient."""
        existing_conf = existing.get('confidence', 0.5)
        new_conf = new.get('confidence', 0.5)
        
        existing_temporal = existing.get('temporal_context', 'unspecified')
        new_temporal = new.get('temporal_context', 'unspecified')
        
        conflict_count = self.get_conflict_count(fingerprint)
        
        # Clear temporal signals - use heuristic
        if new_temporal == 'current' and existing_temporal in ['past', 'unspecified']:
            self.record_conflict(fingerprint, existing, new, 'temporal_priority')
            return 'update', new
        
        if new_temporal == 'permanent':
            self.record_conflict(fingerprint, existing, new, 'permanent_override')
            return 'update', new
        
        # Clear confidence difference
        if new_conf > existing_conf + 0.2:
            self.record_conflict(fingerprint, existing, new, 'high_confidence')
            return 'update', new
        
        # Ambiguous case - use LLM
        if abs(new_conf - existing_conf) < 0.2 and existing_temporal == new_temporal:
            llm_action, llm_reason = await self._llm_compare_memories(existing, new)
            
            if llm_action == "update":
                self.record_conflict(fingerprint, existing, new, f'llm_update: {llm_reason}')
                return 'update', new
            elif llm_action == "merge":
                merged = existing.copy()
                merged['value'] = f"{existing.get('value')}; {new.get('value')}"
                merged['summary'] = f"{existing.get('summary', '')} + {new.get('summary', '')}"
                merged['confidence'] = (existing_conf + new_conf) / 2
                self.record_conflict(fingerprint, existing, new, f'llm_merge: {llm_reason}')
                return 'merge', merged
            elif llm_action == "flag_conflict":
                merged = existing.copy()
                merged['value'] = f"{existing.get('value')} (Note: Conflicting info exists)"
                merged['confidence'] *= 0.7
                merged['conflict_flag'] = True
                merged['conflict_reason'] = llm_reason
                self.record_conflict(fingerprint, existing, new, f'llm_flag: {llm_reason}')
                return 'flag_conflict', merged
        
        # High conflict history - flag
        if conflict_count > 3:
            merged = existing.copy()
            merged['value'] = f"{existing.get('value')} (Note: Conflicting info exists)"
            merged['confidence'] *= 0.7
            merged['conflict_flag'] = True
            self.record_conflict(fingerprint, existing, new, 'flagged_conflict')
            return 'flag_conflict', merged
        
        self.record_conflict(fingerprint, existing, new, 'keep_existing')
        return 'keep', existing
    
    def get_conflict_report(self, fingerprint: str) -> Dict:
        conflicts = self._conflict_history.get(fingerprint, [])
        
        if not conflicts:
            return {
                'total_conflicts': 0,
                'is_conflicted': False
            }
        
        resolution_counts = defaultdict(int)
        for c in conflicts:
            resolution_counts[c['resolution']] += 1
        
        return {
            'total_conflicts': len(conflicts),
            'recent_conflicts_7d': self.get_conflict_count(fingerprint, 7),
            'is_highly_conflicted': self.is_highly_conflicted(fingerprint),
            'resolution_distribution': dict(resolution_counts),
            'latest_conflict': conflicts[-1] if conflicts else None
        }


class MemoryGatekeeper:
    """Validates and gates memory operations with conflict resolution."""
    
    SIMILARITY_THRESHOLD = 0.90
    ARCHIVE_CHECK_DAYS = 90
    MIN_CONFIDENCE = 0.3
    SEMANTIC_BATCH_SIZE = 100
    
    def __init__(self, db, emb_handler, client=None):
        self.db = db
        self.emb_handler = emb_handler
        self.client = client
        self.conflict_resolver = MemoryConflictResolver(client=client)
        self._gatekeeper_stats = {
            'total_checks': 0,
            'fingerprint_matches': 0,
            'semantic_matches': 0,
            'resurrections': 0,
            'rejections': 0,
            'new_creates': 0
        }
    
    async def _check_fingerprint_collision(self, user_id: str, fingerprint: str) -> Optional[Dict]:
        try:
            row = await self.db.fetchone("""
                SELECT id, summary, metadata, status, embedding, priority, use_count, last_used_at
                FROM memories 
                WHERE user_id=? AND metadata LIKE ? AND status='active'
                LIMIT 1
            """, (user_id, f'%"fingerprint":"{fingerprint}"%'))
            
            if row:
                metadata = json.loads(row[2]) if row[2] else {}
                
                self._gatekeeper_stats['fingerprint_matches'] += 1
                
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
    
    async def _check_archived_memories(self, user_id: str, fingerprint: str) -> Optional[Dict]:
        cutoff_date = datetime.now() - timedelta(days=self.ARCHIVE_CHECK_DAYS)
        
        try:
            row = await self.db.fetchone("""
                SELECT id, summary, metadata, last_used_at, priority
                FROM memories 
                WHERE user_id=? AND metadata LIKE ? AND status='archived'
                AND last_used_at > ?
                ORDER BY last_used_at DESC LIMIT 1
            """, (user_id, f'%"fingerprint":"{fingerprint}"%', cutoff_date))
            
            if row:
                self._gatekeeper_stats['resurrections'] += 1
                
                return {
                    "id": row[0],
                    "summary": row[1],
                    "metadata": json.loads(row[2]) if row[2] else {},
                    "last_used_at": row[3],
                    "priority": row[4]
                }
        except Exception as e:
            logger.error(f"Archive check failed: {e}")
        
        return None
    
    async def _check_semantic_similarity(self, user_id: str, embedding: np.ndarray, 
                                  fingerprint: str) -> Optional[Dict]:
        if embedding is None:
            return None
        
        try:
            rows = await self.db.fetchall("""
                SELECT id, summary, metadata, embedding, priority, created_at
                FROM memories 
                WHERE user_id=? AND status='active' AND embedding IS NOT NULL
                ORDER BY last_used_at DESC LIMIT ?
            """, (user_id, self.SEMANTIC_BATCH_SIZE))
            
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
            
            embedding_matrix = np.array(embedding_matrix)
            similarities = np.dot(embedding_matrix, embedding)
            
            max_idx = np.argmax(similarities)
            max_sim = similarities[max_idx]
            
            if max_sim > self.SIMILARITY_THRESHOLD:
                best_row = valid_rows[max_idx]
                metadata = json.loads(best_row[2]) if best_row[2] else {}
                
                if metadata.get('fingerprint') == fingerprint:
                    return None
                
                self._gatekeeper_stats['semantic_matches'] += 1
                
                return {
                    "similarity": float(max_sim),
                    "id": best_row[0],
                    "summary": best_row[1],
                    "metadata": metadata,
                    "priority": best_row[4],
                    "created_at": best_row[5]
                }
        
        except Exception as e:
            logger.error(f"Semantic similarity check failed: {e}")
        
        return None
    
    def _check_quality_threshold(self, canonical_data: Dict) -> bool:
        confidence = canonical_data.get("confidence", 0.5)
        
        if confidence < self.MIN_CONFIDENCE:
            return False
        
        summary = canonical_data.get("summary", "")
        if len(summary.split()) < 2:
            return False
        
        entity = canonical_data.get("entity", "")
        if entity == "unknown" and confidence < 0.6:
            return False
        
        return True
    
    async def validate_and_gate(self, user_id: str, canonical_data: Dict, 
                        embedding: Optional[np.ndarray]) -> Dict:
        self._gatekeeper_stats['total_checks'] += 1
        
        fingerprint = canonical_data.get("fingerprint")
        
        if not self._check_quality_threshold(canonical_data):
            self._gatekeeper_stats['rejections'] += 1
            return {
                "action": "reject",
                "reason": "quality_threshold_not_met",
                "confidence": canonical_data.get("confidence", 0.5)
            }
        
        fingerprint_match = await self._check_fingerprint_collision(user_id, fingerprint)
        if fingerprint_match:
            resolution_action, resolved_data = await self.conflict_resolver.resolve_conflict(
                fingerprint_match.get('metadata', {}),
                canonical_data,
                fingerprint
            )
            
            if resolution_action == 'update':
                return {
                    "action": "merge",
                    "reason": "fingerprint_collision_resolved_update",
                    "existing": fingerprint_match,
                    "resolution": "update"
                }
            elif resolution_action == 'flag_conflict':
                return {
                    "action": "merge",
                    "reason": "fingerprint_collision_conflict_flagged",
                    "existing": fingerprint_match,
                    "resolution": "flag_conflict",
                    "resolved_data": resolved_data
                }
            else:
                return {
                    "action": "merge",
                    "reason": "fingerprint_collision_keep_existing",
                    "existing": fingerprint_match,
                    "resolution": "keep"
                }
        
        archived = await self._check_archived_memories(user_id, fingerprint)
        if archived:
            return {
                "action": "resurrect",
                "reason": "found_in_archive",
                "archived": archived
            }
        
        if embedding is not None:
            semantic_match = await self._check_semantic_similarity(user_id, embedding, fingerprint)
            if semantic_match:
                return {
                    "action": "merge",
                    "reason": "semantic_similarity",
                    "existing": semantic_match,
                    "similarity": semantic_match.get('similarity')
                }
        
        self._gatekeeper_stats['new_creates'] += 1
        return {
            "action": "create",
            "reason": "new_memory"
        }
    
    async def resurrect_memory(self, memory_id: str, new_canonical: Dict) -> bool:
        try:
            metadata_str = json.dumps(new_canonical)
            
            await self.db.execute("""
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
            
            logger.info(f"Memory resurrected: {memory_id}")
            return True
        
        except Exception as e:
            logger.error(f"Resurrection failed: {e}")
            return False
    
    async def get_fingerprint_health(self, user_id: str, fingerprint: str) -> Dict:
        conflict_report = self.conflict_resolver.get_conflict_report(fingerprint)
        
        try:
            row = await self.db.fetchone("""
                SELECT use_count, priority, last_used_at, metadata
                FROM memories
                WHERE user_id=? AND metadata LIKE ? AND status='active'
            """, (user_id, f'%"fingerprint":"{fingerprint}"%'))
            
            if row:
                metadata = json.loads(row[3]) if row[3] else {}
                
                return {
                    'exists': True,
                    'use_count': row[0],
                    'priority': row[1],
                    'last_used': row[2],
                    'stability_score': metadata.get('stability_score', 1.0),
                    'conflict_info': conflict_report
                }
        except Exception as e:
            logger.error(f"Fingerprint health check failed: {e}")
        
        return {
            'exists': False,
            'conflict_info': conflict_report
        }
    
    def get_gatekeeper_stats(self) -> Dict:
        stats = self._gatekeeper_stats.copy()
        
        if stats['total_checks'] > 0:
            stats['rejection_rate'] = round(stats['rejections'] / stats['total_checks'] * 100, 2)
            stats['match_rate'] = round(
                (stats['fingerprint_matches'] + stats['semantic_matches']) / stats['total_checks'] * 100, 
                2
            )
            stats['resurrection_rate'] = round(stats['resurrections'] / stats['total_checks'] * 100, 2)
        
        return stats
    
    def reset_stats(self):
        self._gatekeeper_stats = {
            'total_checks': 0,
            'fingerprint_matches': 0,
            'semantic_matches': 0,
            'resurrections': 0,
            'rejections': 0,
            'new_creates': 0
        }