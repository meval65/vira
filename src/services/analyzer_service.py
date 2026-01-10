import logging
import hashlib
import numpy as np
from typing import List, Optional, Dict
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
        self.embedding_cache: Dict[str, List[float]] = {}
        self.MAX_CACHE_SIZE = 500
        self.MAX_TEXT_LENGTH = 8000
        self.expected_dimension = None
        self._cache_lock = None
        
        try:
            import threading
            self._cache_lock = threading.Lock()
        except ImportError:
            pass
    
    def _get_cache_key(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get_embedding(self, text: str, use_cache: bool = True) -> List[float]:
        if not text or not text.strip():
            logger.warning("[EMBED] Empty text provided")
            return []
        
        clean_text = self._preprocess_text(text)
        
        if not clean_text:
            return []
        
        cache_key = self._get_cache_key(clean_text)
        
        if use_cache:
            if self._cache_lock:
                with self._cache_lock:
                    if cache_key in self.embedding_cache:
                        logger.debug(f"[EMBED] Cache hit for text: {clean_text[:30]}...")
                        return self.embedding_cache[cache_key]
            else:
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
            
            if self.expected_dimension is None:
                self.expected_dimension = len(embedding)
                logger.info(f"[EMBED] Set expected dimension to {self.expected_dimension}")
            elif len(embedding) != self.expected_dimension:
                logger.error(f"[EMBED] Dimension mismatch! Expected {self.expected_dimension}, got {len(embedding)}")
                return []
            
            if use_cache:
                self._add_to_cache(cache_key, embedding)
            
            logger.debug(f"[EMBED] Generated embedding of dimension {len(embedding)}")
            return embedding
            
        except Exception as e:
            logger.error(f"[EMBED-FAIL] Error: {e}", exc_info=True)
            return []
    
    def _add_to_cache(self, cache_key: str, embedding: List[float]):
        if self._cache_lock:
            with self._cache_lock:
                if len(self.embedding_cache) >= self.MAX_CACHE_SIZE:
                    oldest_key = next(iter(self.embedding_cache))
                    del self.embedding_cache[oldest_key]
                self.embedding_cache[cache_key] = embedding
        else:
            if len(self.embedding_cache) >= self.MAX_CACHE_SIZE:
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            self.embedding_cache[cache_key] = embedding
    
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
            cache_key = self._get_cache_key(clean_text)
            
            if use_cache:
                if self._cache_lock:
                    with self._cache_lock:
                        cached_emb = self.embedding_cache.get(cache_key)
                else:
                    cached_emb = self.embedding_cache.get(cache_key)
                
                if cached_emb:
                    results.append(cached_emb)
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
                
                if self.expected_dimension is None:
                    self.expected_dimension = len(embedding)
                elif len(embedding) != self.expected_dimension:
                    logger.error(f"[EMBED-BATCH] Dimension mismatch at index {i}")
                    results[idx] = []
                    continue
                
                results[idx] = embedding
                
                if use_cache:
                    cache_key = self._get_cache_key(uncached_texts[i])
                    self._add_to_cache(cache_key, embedding)
            
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
            vec1 = np.array(embedding1, dtype=np.float32)
            vec2 = np.array(embedding2, dtype=np.float32)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"[SIMILARITY-FAIL] Error: {e}")
            return 0.0
    
    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        if not embedding:
            return []
        
        try:
            vec = np.array(embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            
            if norm == 0:
                return embedding
            
            normalized = vec / norm
            return normalized.tolist()
            
        except Exception as e:
            logger.error(f"[NORMALIZE-FAIL] Error: {e}")
            return embedding
    
    def validate_embedding_dimension(self, embedding: List[float]) -> bool:
        if not embedding:
            return False
        
        if self.expected_dimension is None:
            return True
        
        return len(embedding) == self.expected_dimension
    
    def clear_cache(self):
        if self._cache_lock:
            with self._cache_lock:
                self.embedding_cache.clear()
        else:
            self.embedding_cache.clear()
        logger.info("[EMBED] Cache cleared")
    
    def get_cache_stats(self) -> dict:
        if self._cache_lock:
            with self._cache_lock:
                cache_size = len(self.embedding_cache)
        else:
            cache_size = len(self.embedding_cache)
        
        return {
            "cache_size": cache_size,
            "max_size": self.MAX_CACHE_SIZE,
            "utilization": cache_size / self.MAX_CACHE_SIZE,
            "expected_dimension": self.expected_dimension
        }
    
    def health_check(self) -> bool:
        try:
            test_embedding = self.get_embedding("test", use_cache=False)
            if not test_embedding or len(test_embedding) == 0:
                return False
            
            if self.expected_dimension and len(test_embedding) != self.expected_dimension:
                logger.error(f"[EMBED-HEALTH] Dimension mismatch in health check")
                return False
            
            return True
        except Exception as e:
            logger.error(f"[EMBED-HEALTH] Health check failed: {e}")
            return False