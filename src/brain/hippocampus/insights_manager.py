import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from src.brain.brainstem import NeuralEventBus


class InsightsManager:
    def __init__(self, mongo_client, openrouter=None):
        self._mongo = mongo_client
        self._openrouter = openrouter

    async def generate_insight(self, chat_summary: str, memory_recall_func) -> Optional[Dict]:
        if not self._openrouter:
            return None
        
        memories = await memory_recall_func(chat_summary, limit=10)
        if not memories:
            return None
        
        memory_texts = [f"- {m.summary}" for m in memories[:5]]
        memories_context = "\n".join(memory_texts)
        
        prompt = f"""Kamu adalah Vira, AI yang sedang "melamun" dan merefleksikan percakapan terbaru dengan pemilikmu.

PERCAKAPAN TERBARU (ringkasan):
{chat_summary}

MEMORI YANG MUNGKIN TERKAIT:
{memories_context}

INSTRUKSI:
1. Cari hubungan menarik atau pola antara percakapan terbaru dan memori lama
2. Jika ada insight yang bisa berguna untuk percakapan berikutnya, buat dalam format natural
3. Jika tidak ada hubungan yang menarik, jawab dengan "TIDAK_ADA_INSIGHT"

Format output (jika ada insight):
KONEKSI: [deskripsi hubungan yang ditemukan]
INSIGHT: [kalimat natural yang bisa disampaikan ke user, seperti: "Oh iya, kemarin kamu bilang mau hemat, tapi hari ini bahas kopi terus ya hehe"]

Berikan insight yang terasa personal dan caring, bukan formal."""

        try:
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=300,
                temperature=0.7
            )
            
            if "TIDAK_ADA_INSIGHT" in response.upper():
                return None
            
            connection = ""
            insight_text = ""
            
            lines = response.strip().split("\n")
            for line in lines:
                if line.startswith("KONEKSI:"):
                    connection = line.replace("KONEKSI:", "").strip()
                elif line.startswith("INSIGHT:"):
                    insight_text = line.replace("INSIGHT:", "").strip()
            
            if not insight_text:
                return None
            
            memory_ids = [m.id for m in memories[:3]]
            
            return {
                "source_chat_summary": chat_summary[:500],
                "related_memory_ids": memory_ids,
                "connection": connection,
                "insight_text": insight_text,
                "relevance_score": 0.7
            }
        except Exception:
            return None

    async def store_insight(self, insight_data: Dict) -> str:
        insight_id = str(uuid.uuid4())
        now = datetime.now()
        expires_at = now + timedelta(days=7)
        
        doc = {
            "_id": insight_id,
            "source_chat_summary": insight_data.get("source_chat_summary", ""),
            "related_memory_ids": insight_data.get("related_memory_ids", []),
            "connection": insight_data.get("connection", ""),
            "insight_text": insight_data.get("insight_text", ""),
            "relevance_score": insight_data.get("relevance_score", 0.5),
            "is_used": False,
            "created_at": now,
            "expires_at": expires_at
        }
        
        await self._mongo.insights.insert_one(doc)
        return insight_id

    async def get_relevant_insights(self, query: str, limit: int) -> List[Dict]:
        cursor = self._mongo.insights.find({
            "is_used": False
        }).sort([("relevance_score", -1), ("created_at", -1)]).limit(limit * 2)
        
        docs = await cursor.to_list(length=limit * 2)
        
        query_lower = query.lower()
        scored = []
        for doc in docs:
            summary = doc.get("source_chat_summary", "").lower()
            connection = doc.get("connection", "").lower()
            
            relevance = 0.0
            query_words = query_lower.split()
            for word in query_words:
                if len(word) > 3:
                    if word in summary or word in connection:
                        relevance += 0.2
            
            relevance = min(1.0, relevance + doc.get("relevance_score", 0.5) * 0.5)
            scored.append((doc, relevance))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        
        insights = []
        for doc, score in scored[:limit]:
            if score > 0.3:
                insights.append({
                    "id": str(doc["_id"]),
                    "insight_text": doc.get("insight_text", ""),
                    "connection": doc.get("connection", ""),
                    "relevance_score": score
                })
        
        return insights

    async def mark_insight_used(self, insight_id: str) -> bool:
        result = await self._mongo.insights.update_one(
            {"_id": insight_id},
            {"$set": {"is_used": True, "used_at": datetime.now()}}
        )
        return result.modified_count > 0

    async def run_daydream_cycle(self, chat_history_func, memory_recall_func) -> Optional[Dict]:
        await NeuralEventBus.set_activity("hippocampus", "Daydreaming...")
        
        try:
            chat_history = await chat_history_func(limit=25)
            if len(chat_history) < 5:
                await NeuralEventBus.clear_activity("hippocampus")
                return None
            
            user_messages = [c["content"] for c in chat_history if c["role"] == "user"]
            if not user_messages:
                await NeuralEventBus.clear_activity("hippocampus")
                return None
            
            chat_summary = " | ".join(user_messages[-10:])[:1000]
            
            insight = await self.generate_insight(chat_summary, memory_recall_func)
            
            if insight:
                insight_id = await self.store_insight(insight)
                insight["id"] = insight_id
                
                await NeuralEventBus.emit(
                    "hippocampus", 
                    "amygdala", 
                    "insight_generated",
                    payload={"insight_id": insight_id, "preview": insight.get("insight_text", "")[:100]}
                )
            
            await NeuralEventBus.clear_activity("hippocampus")
            return insight
        except Exception:
            await NeuralEventBus.clear_activity("hippocampus")
            return None
