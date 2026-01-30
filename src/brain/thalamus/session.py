import os
import datetime
from typing import List, Optional, Any
import numpy as np

import PIL.Image
from google.genai import types

from src.brain.db.mongo_client import MongoDBClient
from .types import SessionMessage


class SessionManagerMixin:
    MAX_SHORT_TERM: int = 20
    SIMILARITY_THRESHOLD: float = 0.75

    async def get_session(self, limit: int = 40) -> List[SessionMessage]:
        cursor = self._mongo.chat_logs.find().sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        messages = []
        for doc in reversed(docs):
            messages.append(SessionMessage(
                role=doc.get("role", "user"),
                content=doc.get("content", ""),
                timestamp=doc.get("timestamp", datetime.datetime.now()),
                image_path=doc.get("image_path"),
                embedding=doc.get("embedding")
            ))
        return messages

    def get_history_for_model(self) -> List[types.Content]:
        return []

    async def get_history_for_model_async(self) -> List[types.Content]:
        messages = await self.get_session(limit=self.MAX_SHORT_TERM)
        history = []
        
        for msg in messages:
            parts = []
            if msg.content:
                parts.append(types.Part.from_text(text=msg.content))
            if msg.image_path and os.path.exists(msg.image_path):
                try:
                    img = PIL.Image.open(msg.image_path)
                    parts.append(types.Part.from_image(image=img))
                except Exception:
                    pass
            if parts:
                history.append(types.Content(role=msg.role, parts=parts))
        
        return history

    async def update_session(
        self,
        user_text: str,
        ai_response: str,
        image_path: Optional[str] = None,
        user_embedding: Optional[List[float]] = None,
        ai_embedding: Optional[List[float]] = None
    ) -> None:
        now = datetime.datetime.now()
        
        await self._mongo.chat_logs.insert_one({
            "role": "user",
            "content": user_text,
            "timestamp": now,
            "image_path": image_path,
            "embedding": user_embedding
        })
        
        await self._mongo.chat_logs.insert_one({
            "role": "model",
            "content": ai_response,
            "timestamp": now + datetime.timedelta(milliseconds=1),
            "embedding": ai_embedding
        })
        
        self._metadata["last_interaction"] = now.isoformat()
        await self._save_metadata()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        v1 = np.array(vec1, dtype=np.float32)
        v2 = np.array(vec2, dtype=np.float32)

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(v1, v2) / (norm1 * norm2))

    async def get_hybrid_context(
        self,
        query_embedding: Optional[List[float]],
        short_term_limit: int = 20,
        long_term_limit: int = 5
    ) -> dict:
        from src.brain.brainstem import NeuralEventBus
        await NeuralEventBus.set_activity("thalamus", "Building Hybrid Context")
        
        short_term_cursor = self._mongo.chat_logs.find().sort("timestamp", -1).limit(short_term_limit)
        short_term_docs = await short_term_cursor.to_list(length=short_term_limit)
        
        short_term = []
        oldest_short_term = None
        for doc in reversed(short_term_docs):
            short_term.append(SessionMessage(
                role=doc.get("role", "user"),
                content=doc.get("content", ""),
                timestamp=doc.get("timestamp", datetime.datetime.now()),
                image_path=doc.get("image_path"),
                embedding=doc.get("embedding")
            ))
            if oldest_short_term is None or doc.get("timestamp", datetime.datetime.now()) < oldest_short_term:
                oldest_short_term = doc.get("timestamp")
        
        long_term = []
        if query_embedding and oldest_short_term:
            long_term_cursor = self._mongo.chat_logs.find({
                "timestamp": {"$lt": oldest_short_term},
                "embedding": {"$exists": True, "$ne": None}
            }).limit(100)
            
            long_term_docs = await long_term_cursor.to_list(length=100)
            
            scored = []
            for doc in long_term_docs:
                emb = doc.get("embedding")
                if emb:
                    sim = self._cosine_similarity(query_embedding, emb)
                    if sim >= self.SIMILARITY_THRESHOLD:
                        scored.append((doc, sim))
            
            scored.sort(key=lambda x: x[1], reverse=True)
            
            for doc, _ in scored[:long_term_limit]:
                long_term.append(SessionMessage(
                    role=doc.get("role", "user"),
                    content=doc.get("content", ""),
                    timestamp=doc.get("timestamp", datetime.datetime.now()),
                    image_path=doc.get("image_path"),
                    embedding=doc.get("embedding")
                ))
        
        await NeuralEventBus.clear_activity("thalamus")
        
        return {
            "short_term": short_term,
            "long_term": long_term
        }

    def get_relevant_history(
        self,
        query_embedding: Optional[List[float]],
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[SessionMessage]:
        return []

    async def get_relevant_history_async(
        self,
        query_embedding: Optional[List[float]],
        top_k: int = 5,
        min_similarity: float = 0.5
    ) -> List[SessionMessage]:
        if not query_embedding:
            return []
        
        cursor = self._mongo.chat_logs.find({
            "embedding": {"$exists": True, "$ne": None}
        }).limit(200)
        
        docs = await cursor.to_list(length=200)
        
        scored = []
        for doc in docs:
            emb = doc.get("embedding")
            if emb:
                sim = self._cosine_similarity(query_embedding, emb)
                if sim >= min_similarity:
                    scored.append((doc, sim))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        for doc, _ in scored[:top_k]:
            result.append(SessionMessage(
                role=doc.get("role", "user"),
                content=doc.get("content", ""),
                timestamp=doc.get("timestamp", datetime.datetime.now()),
                image_path=doc.get("image_path"),
                embedding=doc.get("embedding")
            ))
        
        return result

    async def clear_session(self) -> None:
        await self._mongo.chat_logs.delete_many({})
        self._metadata = {
            "summary": "",
            "memory_summary": "",
            "schedule_summary": "",
            "last_analysis_count": 0,
            "last_interaction": None
        }
        await self._save_metadata()

    async def cleanup_session(self) -> None:
        await self._save_metadata()

    def get_summary(self) -> str:
        return self._metadata.get("summary", "")

    def get_memory_summary(self) -> str:
        return self._metadata.get("memory_summary", "")

    def update_summary(self, summary: str) -> None:
        self._metadata["summary"] = summary

    def update_memory_summary(self, summary: str) -> None:
        self._metadata["memory_summary"] = summary

    def update_schedule_summary(self, summary: str) -> None:
        self._metadata["schedule_summary"] = summary

    def should_run_memory_analysis(self, current_count: int) -> bool:
        last_count = self._metadata.get("last_analysis_count", 0)
        return current_count >= last_count + 5

    def mark_memory_analysis_done(self, count: int) -> None:
        self._metadata["last_analysis_count"] = count
