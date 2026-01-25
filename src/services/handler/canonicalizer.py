import logging
import hashlib
import re
import json
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from google import genai
from google.genai import types
from src.config import CANONICALIZATION_INSTRUCTION

logger = logging.getLogger(__name__)


class MemoryEvolutionTracker:
    """Tracks evolution of memories with LLM-powered analysis."""
    
    UPDATE_TYPES = {
        "fact_update": "Old fact replaced by new information",
        "additional_info": "New detail added to existing fact", 
        "temporal_update": "Time-based change (past→present, status change)",
        "preference_change": "User preference evolved",
        "correction": "User corrected previous statement",
        "reinforcement": "Same fact confirmed/strengthened"
    }
    
    def __init__(self, client=None, tier2_model: str = None):
        self._evolution_history: Dict[str, List[Dict]] = defaultdict(list)
        self._lock = None
        self.client = client
        self.tier2_model = tier2_model or "models/gemma-3-12b-it"
        
    def record_evolution(self, fingerprint: str, old_value: Optional[str], 
                        new_value: str, reason: str, confidence: float,
                        update_type: str = "unknown"):
        """Record a memory evolution with classification."""
        evolution_entry = {
            "timestamp": datetime.now().isoformat(),
            "old_value": old_value,
            "new_value": new_value,
            "reason": reason,
            "confidence": confidence,
            "update_type": update_type
        }
        
        self._evolution_history[fingerprint].append(evolution_entry)
        
        if len(self._evolution_history[fingerprint]) > 20:
            self._evolution_history[fingerprint].pop(0)
        
        logger.debug(f"[EVOLUTION] {fingerprint}: {reason} ({update_type})")
    
    async def analyze_update_type(self, old_summary: str, new_summary: str, 
                                   temporal_context: str) -> Tuple[str, str]:
        """Use Tier-2 LLM to classify what type of update this is."""
        if not self.client:
            # Fallback to heuristic
            return self._heuristic_update_type(old_summary, new_summary, temporal_context)
        
        try:
            prompt = f"""Analyze these two memory statements and classify the relationship.

OLD: {old_summary}
NEW: {new_summary}
TEMPORAL_CONTEXT: {temporal_context}

Classify as ONE of:
- fact_update: Old fact replaced (e.g., "lives in Jakarta" → "lives in Bali")
- additional_info: New detail added (e.g., "likes coffee" + "prefers dark roast")
- temporal_update: Time-based change (e.g., "is student" → "is graduate")
- preference_change: Preference evolved (e.g., "hates spicy" → "now likes spicy")
- correction: User corrected previous statement
- reinforcement: Same fact confirmed

Response format (JSON only):
{{"type": "...", "reason": "brief explanation"}}"""

            response = self.client.models.generate_content(
                model=self.tier2_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=128
                )
            )
            
            if response.text:
                text = response.text.strip()
                # Extract JSON
                start = text.find('{')
                end = text.rfind('}') + 1
                if start != -1 and end > start:
                    data = json.loads(text[start:end])
                    return data.get("type", "unknown"), data.get("reason", "LLM analysis")
        
        except Exception as e:
            logger.warning(f"[EVOLUTION] LLM analysis failed: {e}")
        
        return self._heuristic_update_type(old_summary, new_summary, temporal_context)
    
    def _heuristic_update_type(self, old: str, new: str, temporal: str) -> Tuple[str, str]:
        """Fallback heuristic classification."""
        old_lower, new_lower = old.lower(), new.lower()
        
        if temporal in ["current", "future"] and any(w in old_lower for w in ["was", "used to", "before"]):
            return "temporal_update", "temporal context indicates time-based change"
        
        # Check for preference words
        pref_words = ["like", "love", "hate", "prefer", "enjoy", "dislike"]
        if any(w in old_lower or w in new_lower for w in pref_words):
            return "preference_change", "preference-related vocabulary detected"
        
        # Check similarity
        old_words = set(old_lower.split())
        new_words = set(new_lower.split())
        overlap = len(old_words & new_words) / max(len(old_words | new_words), 1)
        
        if overlap > 0.7:
            return "reinforcement", "high word overlap suggests reinforcement"
        elif overlap > 0.3:
            return "additional_info", "moderate overlap suggests additional info"
        else:
            return "fact_update", "low overlap suggests fact replacement"
    
    def get_evolution_history(self, fingerprint: str) -> List[Dict]:
        return self._evolution_history.get(fingerprint, [])
    
    def build_history_metadata(self, fingerprint: str) -> List[Dict]:
        """Build history array for DB metadata persistence."""
        history = self._evolution_history.get(fingerprint, [])
        # Return last 10 entries for DB storage
        return history[-10:]
    
    def get_change_frequency(self, fingerprint: str, days: int = 30) -> int:
        history = self._evolution_history.get(fingerprint, [])
        cutoff = datetime.now() - timedelta(days=days)
        
        return sum(1 for entry in history 
                  if datetime.fromisoformat(entry["timestamp"]) > cutoff)
    
    def is_volatile_memory(self, fingerprint: str, threshold: int = 3) -> bool:
        freq = self.get_change_frequency(fingerprint, days=7)
        return freq >= threshold
    
    def get_stability_score(self, fingerprint: str) -> float:
        history = self._evolution_history.get(fingerprint, [])
        if not history:
            return 1.0
        
        recent_changes = self.get_change_frequency(fingerprint, days=30)
        total_time_span = len(history)
        
        if total_time_span == 0:
            return 1.0
        
        stability = 1.0 - min(recent_changes / (total_time_span * 2), 1.0)
        return max(0.0, stability)


class MemoryCanonicalizer:
    TYPE_MAPPING = {
        "preference": ["like", "love", "enjoy", "prefer", "hate", "dislike", "favorite"],
        "fact": ["is", "has", "born", "lives", "works", "name", "age", "from"],
        "event": ["went", "visited", "attended", "completed", "started", "finished", "happened"],
        "skill": ["knows", "learning", "mastered", "studying", "practicing", "expert"],
        "context": ["background", "history", "experience", "situation", "condition"],
        "emotion": ["feels", "felt", "emotional", "mood", "happy", "sad", "angry"],
        "relationship": ["friend", "family", "partner", "colleague", "knows", "met"],
        "goal": ["want", "plan", "aim", "aspire", "dream", "wish", "hope"],
        "habit": ["usually", "always", "often", "regularly", "routine", "custom"],
        "opinion": ["think", "believe", "consider", "view", "perspective"]
    }
    
    TEMPORAL_KEYWORDS = {
        "current": ["now", "currently", "today", "this week", "recently"],
        "past": ["was", "used to", "before", "previously", "ago"],
        "future": ["will", "planning", "going to", "next", "soon"],
        "permanent": ["always", "never", "forever", "constantly"]
    }
    
    def __init__(self, client: genai.Client, model: str, tier2_model: str = None):
        self.client = client
        self.model = model
        self.tier2_model = tier2_model or "models/gemma-3-12b-it"
        self.evolution_tracker = MemoryEvolutionTracker(client=client, tier2_model=self.tier2_model)
        self._canonicalization_cache: Dict[str, Tuple[Dict, datetime]] = {}
        self._cache_ttl = 3600
        
    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        if not text:
            return None
            
        try:
            text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'```', '', text)
            text = text.strip()
            
            start_idx = text.find('{')
            end_idx = text.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                json_str = text[start_idx : end_idx + 1]
                return json.loads(json_str)
                
        except Exception as e:
            logger.error(f"JSON extraction failed: {e} | Input: {text[:100]}...")
        return None
    
    def _generate_fingerprint(self, mem_type: str, relation: str, entity: str) -> str:
        base = f"{mem_type}:{relation}:{entity}".lower()
        base = re.sub(r'[^a-z0-9:]', '', base)
        return base
    
    def _infer_type_from_summary(self, summary: str) -> str:
        summary_lower = summary.lower()
        
        type_scores = defaultdict(int)
        
        for mem_type, keywords in self.TYPE_MAPPING.items():
            for keyword in keywords:
                if keyword in summary_lower:
                    type_scores[mem_type] += 1
        
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]
        
        return "general"
    
    def _detect_temporal_context(self, summary: str) -> str:
        summary_lower = summary.lower()
        
        for temporal_type, keywords in self.TEMPORAL_KEYWORDS.items():
            if any(keyword in summary_lower for keyword in keywords):
                return temporal_type
        
        return "unspecified"
    
    def _calculate_priority_boost(self, summary: str, mem_type: str) -> float:
        boost = 0.0
        summary_lower = summary.lower()
        
        high_priority_words = ["important", "critical", "urgent", "must", "need", "penting", "harus"]
        if any(word in summary_lower for word in high_priority_words):
            boost += 0.2
        
        if mem_type in ["goal", "preference", "relationship"]:
            boost += 0.1
        
        if len(summary.split()) > 15:
            boost += 0.05
        
        return min(boost, 0.3)
    
    def _fallback_canonicalization(self, summary: str, mem_type: str) -> Dict:
        words = re.findall(r'\w+', summary.lower())
        entity = words[0] if words else "unknown"
        relation = "relates_to"
        
        if mem_type == "general":
            mem_type = self._infer_type_from_summary(summary)
        
        fingerprint = self._generate_fingerprint(mem_type, relation, entity)
        temporal = self._detect_temporal_context(summary)
        
        return {
            "fingerprint": fingerprint,
            "type": mem_type,
            "entity": entity,
            "relation": relation,
            "value": summary,
            "confidence": 0.5,
            "temporal_context": temporal,
            "extracted_entities": words[:5],
            "fallback": True
        }
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        if cache_key in self._canonicalization_cache:
            cached_data, cached_time = self._canonicalization_cache[cache_key]
            if (datetime.now() - cached_time).total_seconds() < self._cache_ttl:
                return cached_data
            del self._canonicalization_cache[cache_key]
        return None
    
    def _save_to_cache(self, cache_key: str, data: Dict):
        if len(self._canonicalization_cache) > 200:
            oldest_keys = sorted(
                self._canonicalization_cache.keys(),
                key=lambda k: self._canonicalization_cache[k][1]
            )[:50]
            for key in oldest_keys:
                del self._canonicalization_cache[key]
        
        self._canonicalization_cache[cache_key] = (data, datetime.now())
    
    def canonicalize(self, summary: str, mem_type: str, priority: float) -> Dict[str, Any]:
        cache_key = hashlib.md5(f"{summary}:{mem_type}".encode()).hexdigest()
        cached = self._get_from_cache(cache_key)
        if cached:
            logger.debug(f"Cache hit for canonicalization: {summary[:50]}")
            return cached
        
        try:
            prompt = f"{CANONICALIZATION_INSTRUCTION}\n\nInput: {summary}"
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=256
                )
            )
            
            if response.text:
                canonical_data = self._extract_json_from_response(response.text)
                
                if canonical_data and all(k in canonical_data for k in ["fingerprint", "type", "entity"]):
                    canonical_data["summary"] = summary
                    canonical_data["original_type"] = mem_type
                    
                    priority_boost = self._calculate_priority_boost(summary, canonical_data.get("type", mem_type))
                    canonical_data["priority"] = min(priority + priority_boost, 1.0)
                    
                    canonical_data["source_count"] = 1
                    canonical_data["created_at"] = datetime.now().isoformat()
                    canonical_data["updated_at"] = datetime.now().isoformat()
                    canonical_data["status"] = "active"
                    canonical_data["fallback"] = False
                    
                    if "confidence" not in canonical_data:
                        canonical_data["confidence"] = 0.7
                    
                    if "temporal_context" not in canonical_data:
                        canonical_data["temporal_context"] = self._detect_temporal_context(summary)
                    
                    self._save_to_cache(cache_key, canonical_data)
                    return canonical_data
        
        except Exception as e:
            logger.warning(f"Canonicalization failed, using fallback: {e}")
        
        fallback = self._fallback_canonicalization(summary, mem_type)
        fallback["summary"] = summary
        fallback["original_type"] = mem_type
        
        priority_boost = self._calculate_priority_boost(summary, fallback["type"])
        fallback["priority"] = min(priority + priority_boost, 1.0)
        
        fallback["source_count"] = 1
        fallback["created_at"] = datetime.now().isoformat()
        fallback["updated_at"] = datetime.now().isoformat()
        fallback["status"] = "active"
        
        self._save_to_cache(cache_key, fallback)
        return fallback
    
    async def merge_canonical_memories(self, existing: Dict, new: Dict) -> Dict:
        """Merge memories with LLM-powered evolution analysis."""
        merged = existing.copy()
        fingerprint = existing.get("fingerprint", "")
        
        merged["source_count"] = existing.get("source_count", 1) + 1
        merged["updated_at"] = datetime.now().isoformat()
        
        existing_confidence = existing.get("confidence", 0.5)
        new_confidence = new.get("confidence", 0.5)
        merged["confidence"] = min(1.0, (existing_confidence + new_confidence) / 2 + 0.1)
        
        old_value = existing.get("value")
        new_value = new.get("value")
        old_summary = existing.get("summary", "")
        new_summary = new.get("summary", "")
        
        if new_value != old_value:
            existing_temporal = existing.get("temporal_context", "unspecified")
            new_temporal = new.get("temporal_context", "unspecified")
            
            # Use LLM to analyze update type
            update_type, update_reason = await self.evolution_tracker.analyze_update_type(
                old_summary, new_summary, new_temporal
            )
            
            should_update = False
            
            if new_temporal in ["current", "future"] and existing_temporal in ["past", "unspecified"]:
                should_update = True
            elif new_confidence > existing_confidence + 0.15:
                should_update = True
            elif new_temporal == "permanent":
                should_update = True
            elif update_type in ["fact_update", "temporal_update", "correction"]:
                should_update = True
            
            if should_update:
                self.evolution_tracker.record_evolution(
                    fingerprint, old_value, new_value, update_reason, new_confidence,
                    update_type=update_type
                )
                merged["value"] = new_value
                merged["summary"] = new_summary
                merged["temporal_context"] = new_temporal
                merged["last_update_type"] = update_type
                merged["last_update_reason"] = update_reason
                
                if self.evolution_tracker.is_volatile_memory(fingerprint):
                    merged["volatility_flag"] = True
                    merged["confidence"] *= 0.9
            elif update_type == "additional_info":
                # Append additional info instead of replacing
                merged["value"] = f"{old_value}; {new_value}"
                merged["summary"] = f"{old_summary} Additionally: {new_summary}"
                self.evolution_tracker.record_evolution(
                    fingerprint, old_value, merged["value"], update_reason, new_confidence,
                    update_type="additional_info"
                )
        
        merged["priority"] = max(existing.get("priority", 0.5), new.get("priority", 0.5))
        
        stability_score = self.evolution_tracker.get_stability_score(fingerprint)
        merged["stability_score"] = stability_score
        
        # Persist history to metadata
        merged["history"] = self.evolution_tracker.build_history_metadata(fingerprint)
        
        return merged
    
    def analyze_memory_quality(self, canonical_data: Dict) -> Dict:
        quality_metrics = {
            "confidence": canonical_data.get("confidence", 0.5),
            "stability": canonical_data.get("stability_score", 1.0),
            "source_strength": min(canonical_data.get("source_count", 1) / 10, 1.0),
            "temporal_clarity": 1.0 if canonical_data.get("temporal_context") != "unspecified" else 0.6,
            "is_volatile": canonical_data.get("volatility_flag", False)
        }
        
        overall_quality = (
            quality_metrics["confidence"] * 0.4 +
            quality_metrics["stability"] * 0.3 +
            quality_metrics["source_strength"] * 0.2 +
            quality_metrics["temporal_clarity"] * 0.1
        )
        
        if quality_metrics["is_volatile"]:
            overall_quality *= 0.85
        
        quality_metrics["overall_score"] = min(overall_quality, 1.0)
        quality_metrics["grade"] = self._get_quality_grade(overall_quality)
        
        return quality_metrics
    
    def _get_quality_grade(self, score: float) -> str:
        if score >= 0.85:
            return "A"
        elif score >= 0.70:
            return "B"
        elif score >= 0.55:
            return "C"
        elif score >= 0.40:
            return "D"
        else:
            return "F"
    
    def get_evolution_insights(self, fingerprint: str) -> Dict:
        history = self.evolution_tracker.get_evolution_history(fingerprint)
        
        return {
            "total_changes": len(history),
            "recent_changes_7d": self.evolution_tracker.get_change_frequency(fingerprint, 7),
            "recent_changes_30d": self.evolution_tracker.get_change_frequency(fingerprint, 30),
            "is_volatile": self.evolution_tracker.is_volatile_memory(fingerprint),
            "stability_score": self.evolution_tracker.get_stability_score(fingerprint),
            "change_history": history[-5:]
        }
    
    def get_cache_stats(self) -> Dict:
        return {
            "cache_size": len(self._canonicalization_cache),
            "cache_ttl_seconds": self._cache_ttl,
            "tracked_memories": len(self.evolution_tracker._evolution_history)
        }
    
    def clear_cache(self):
        self._canonicalization_cache.clear()