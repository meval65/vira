import logging
import hashlib
from typing import List, Optional
from functools import lru_cache
from openai import OpenAI
from src.config import OLLAMA_BASE_URL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

class MemoryAnalyzer:
    def __init__(self):
        self.client = OpenAI(
            base_url=OLLAMA_BASE_URL,
            api_key="ollama",
        )
        self.embedding_cache = {}
        self.MAX_CACHE_SIZE = 500
        self.MAX_TEXT_LENGTH = 8000
        
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        if not text or not text.strip():
            logger.warning("[EMBED] Empty text provided")
            return []
        
        clean_text = self._preprocess_text(text)
        
        if not clean_text:
            return []
        
        if use_cache:
            cache_key = self._get_cache_key(clean_text)
            if cache_key in self.embedding_cache:
                logger.debug(f"[EMBED] Cache hit for text: {clean_text[:30]}...")
                return self.embedding_cache[cache_key]
        
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[clean_text]
            )
            
            embedding = response.data[0].embedding
            
            if not embedding or len(embedding) == 0:
                logger.error("[EMBED] Empty embedding returned")
                return []
            
            if use_cache and len(self.embedding_cache) < self.MAX_CACHE_SIZE:
                self.embedding_cache[cache_key] = embedding
            elif use_cache and len(self.embedding_cache) >= self.MAX_CACHE_SIZE:
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
                self.embedding_cache[cache_key] = embedding
            
            logger.debug(f"[EMBED] Generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"[EMBED-FAIL] Error: {e}", exc_info=True)
            return []
    
    def _preprocess_text(self, text: str) -> str:
        clean_text = text.replace("\n", " ").replace("\r", " ")
        clean_text = " ".join(clean_text.split())
        clean_text = clean_text.strip()
        
        if len(clean_text) > self.MAX_TEXT_LENGTH:
            logger.warning(f"[EMBED] Text truncated from {len(clean_text)} to {self.MAX_TEXT_LENGTH} chars")
            clean_text = clean_text[:self.MAX_TEXT_LENGTH]
        
        return clean_text
    
    def get_embeddings_batch(self, texts: List[str], use_cache: bool = True) -> List[List[float]]:
        if not texts:
            return []
        
        results = []
        uncached_texts = []
        uncached_indices = []
        
        for idx, text in enumerate(texts):
            if not text or not text.strip():
                results.append([])
                continue
            
            clean_text = self._preprocess_text(text)
            
            if use_cache:
                cache_key = self._get_cache_key(clean_text)
                if cache_key in self.embedding_cache:
                    results.append(self.embedding_cache[cache_key])
                    continue
            
            results.append(None)
            uncached_texts.append(clean_text)
            uncached_indices.append(idx)
        
        if not uncached_texts:
            return results
        
        try:
            response = self.client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=uncached_texts
            )
            
            for i, embedding_data in enumerate(response.data):
                embedding = embedding_data.embedding
                idx = uncached_indices[i]
                results[idx] = embedding
                
                if use_cache and len(self.embedding_cache) < self.MAX_CACHE_SIZE:
                    cache_key = self._get_cache_key(uncached_texts[i])
                    self.embedding_cache[cache_key] = embedding
            
            logger.info(f"[EMBED-BATCH] Generated {len(uncached_texts)} embeddings")
            
        except Exception as e:
            logger.error(f"[EMBED-BATCH-FAIL] Error: {e}", exc_info=True)
            for idx in uncached_indices:
                if results[idx] is None:
                    results[idx] = []
        
        return results
    
    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        if not embedding1 or not embedding2:
            return 0.0
        
        if len(embedding1) != len(embedding2):
            logger.warning(f"[SIMILARITY] Dimension mismatch: {len(embedding1)} vs {len(embedding2)}")
            return 0.0
        
        try:
            import numpy as np
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"[SIMILARITY-FAIL] Error: {e}")
            return 0.0
    
    def clear_cache(self):
        self.embedding_cache.clear()
        logger.info("[EMBED] Cache cleared")
    
    def get_cache_stats(self) -> dict:
        return {
            "cache_size": len(self.embedding_cache),
            "max_size": self.MAX_CACHE_SIZE,
            "utilization": len(self.embedding_cache) / self.MAX_CACHE_SIZE
        }
    
    def health_check(self) -> bool:
        try:
            test_embedding = self.get_embedding("test", use_cache=False)
            return len(test_embedding) > 0
        except Exception as e:
            logger.error(f"[EMBED-HEALTH] Health check failed: {e}")
            return False