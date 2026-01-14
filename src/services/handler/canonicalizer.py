import logging
import hashlib
import re
from typing import Dict, Optional, Any, List
from datetime import datetime
from google import genai
from google.genai import types
from src.config import CANONICALIZATION_INSTRUCTION

logger = logging.getLogger(__name__)


class MemoryCanonicalizer:
    TYPE_MAPPING = {
        "preference": ["like", "love", "enjoy", "prefer", "hate", "dislike"],
        "fact": ["is", "has", "born", "lives", "works", "name"],
        "event": ["went", "visited", "attended", "completed", "started"],
        "skill": ["knows", "learning", "mastered", "studying", "practicing"],
        "context": ["background", "history", "experience", "situation"],
        "emotion": ["feels", "felt", "emotional", "mood"]
    }
    
    def __init__(self, client: genai.Client, model: str):
        self.client = client
        self.model = model
        
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
                import json
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
        
        for mem_type, keywords in self.TYPE_MAPPING.items():
            if any(keyword in summary_lower for keyword in keywords):
                return mem_type
        
        return "general"
    
    def _fallback_canonicalization(self, summary: str, mem_type: str) -> Dict:
        words = re.findall(r'\w+', summary.lower())
        entity = words[0] if words else "unknown"
        relation = "relates_to"
        
        if mem_type == "general":
            mem_type = self._infer_type_from_summary(summary)
        
        fingerprint = self._generate_fingerprint(mem_type, relation, entity)
        
        return {
            "fingerprint": fingerprint,
            "type": mem_type,
            "entity": entity,
            "relation": relation,
            "value": summary,
            "confidence": 0.5
        }
    
    def canonicalize(self, summary: str, mem_type: str, priority: float) -> Dict[str, Any]:
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
                    canonical_data["priority"] = priority
                    canonical_data["source_count"] = 1
                    canonical_data["created_at"] = datetime.now().isoformat()
                    canonical_data["updated_at"] = datetime.now().isoformat()
                    canonical_data["status"] = "active"
                    
                    if "confidence" not in canonical_data:
                        canonical_data["confidence"] = 0.7
                    
                    return canonical_data
        
        except Exception as e:
            logger.warning(f"Canonicalization failed, using fallback: {e}")
        
        fallback = self._fallback_canonicalization(summary, mem_type)
        fallback["summary"] = summary
        fallback["original_type"] = mem_type
        fallback["priority"] = priority
        fallback["source_count"] = 1
        fallback["created_at"] = datetime.now().isoformat()
        fallback["updated_at"] = datetime.now().isoformat()
        fallback["status"] = "active"
        
        return fallback
    
    def merge_canonical_memories(self, existing: Dict, new: Dict) -> Dict:
        merged = existing.copy()
        
        merged["source_count"] = existing.get("source_count", 1) + 1
        merged["updated_at"] = datetime.now().isoformat()
        
        existing_confidence = existing.get("confidence", 0.5)
        new_confidence = new.get("confidence", 0.5)
        merged["confidence"] = min(1.0, (existing_confidence + new_confidence) / 2 + 0.1)
        
        if new.get("value") != existing.get("value"):
            if new_confidence > existing_confidence:
                merged["value"] = new.get("value")
                merged["summary"] = new.get("summary")
        
        merged["priority"] = max(existing.get("priority", 0.5), new.get("priority", 0.5))
        
        return merged