import json
import hashlib
from typing import Dict, Optional
from src.brain.constants import CANONICALIZATION_INSTRUCTION


class Canonicalizer:
    def __init__(self, openrouter=None):
        self._openrouter = openrouter

    async def canonicalize(self, summary: str, memory_type: str) -> Dict:
        if not self._openrouter:
            return self._fallback_canonicalize(summary, memory_type)
        
        try:
            prompt = f"{CANONICALIZATION_INSTRUCTION}\n\nInput: \"{summary}\""
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=256,
                temperature=0.1
            )
            result = self._extract_json(response)
            return result if result else self._fallback_canonicalize(summary, memory_type)
        except Exception:
            return self._fallback_canonicalize(summary, memory_type)

    def _fallback_canonicalize(self, summary: str, memory_type: str) -> Dict:
        words = summary.lower().split()
        entity = words[-1] if words else "unknown"
        relation = "related_to"
        
        relation_keywords = {
            "likes": ["likes", "loves", "enjoys", "prefers", "appreciates"],
            "dislikes": ["hates", "dislikes", "avoids", "despises"],
            "is": ["is", "am", "are", "was", "were"],
            "has": ["has", "have", "owns", "possesses"],
            "works_at": ["works", "employed", "job"],
            "lives_in": ["lives", "resides", "located"]
        }
        
        for rel, keywords in relation_keywords.items():
            if any(kw in words for kw in keywords):
                relation = rel
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

    def _extract_json(self, text: str) -> Optional[Dict]:
        import re
        text = re.sub(r'^```(?:json)?\s*', '', text.strip())
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
