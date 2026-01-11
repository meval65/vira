import json
import logging
import datetime
from typing import Dict, Optional, List
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

class AnalysisService:
    def __init__(self, client: genai.Client, tier_1_model: str, tier_2_model: str):
        self.client = client
        self.tier_1_model = tier_1_model
        self.tier_2_model = tier_2_model
    
    def run_chat_analysis(self, history: list, old_summary: str, instruction: str) -> str:
        recent_history = history[-10:] if len(history) > 10 else history
        
        prompt = (
            f"{instruction}\n\n"
            f"[OLD SUMMARY]: {old_summary}\n\n"
            f"[HISTORY CHAT]: {recent_history}\n\n"
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.tier_1_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=512
                )
            )
            
            result = response.text if response.text else ""
            logger.info(f"[MODEL-OUTPUT: {self.tier_1_model}] (Chat Analysis): {result[:100]}...")
            return result
        
        except Exception as e:
            logger.error(f"[CHAT-ANALYSIS-ERROR] {e}")
            return old_summary
    
    def run_memory_analysis(self, memories: List[Dict], old_memory_summary: str, 
                            instruction: str) -> str:
        if not memories:
            return old_memory_summary
        
        mem_lines = [
            f"{i+1}: {m.get('summary', '')}" 
            for i, m in enumerate(memories)
        ]
        
        prompt = (
            f"{instruction}\n\n"
            f"[Old_Memory_Summary]:\n{old_memory_summary}\n\n"
            f"[New_Memory_Data]:\n{chr(10).join(mem_lines)}\n"
        )
        
        try:
            response = self.client.models.generate_content(
                model=self.tier_1_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.4,
                    max_output_tokens=512
                )
            )
            
            result = response.text if response.text else ""
            logger.info(f"[MODEL-OUTPUT: {self.tier_1_model}] (Memory Analysis): {result[:100]}...")
            return result
        
        except Exception as e:
            logger.error(f"[MEMORY-ANALYSIS-ERROR] {e}")
            return old_memory_summary
    
    def run_interaction_analysis(self, user_text: str, ai_text: str, 
                                instruction: str) -> Optional[Dict]:
        if not user_text or not ai_text:
            return None
        
        now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        prompt = (
            f"{instruction}\n\n"
            f"[SYSTEM TIME]: {now_str}\n"
            f"[INTERACTION]\nUser: {user_text}\nAI: {ai_text}\n"
        )
        
        try:
            chat = self.client.chats.create(
                model=self.tier_2_model,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                    top_p=0.9
                )
            )
            
            response = chat.send_message(prompt)
            
            if not response.text:
                return None
            
            logger.info(f"[MODEL-OUTPUT: {self.tier_2_model}] (Interaction Analysis): {response.text[:100]}...")
            
            return self._safe_json_parse(response.text)
        
        except Exception as e:
            logger.error(f"[INTERACTION-ANALYSIS-ERROR] {e}")
            return None
    
    def _safe_json_parse(self, text: str) -> Optional[Dict]:
        try:
            clean = text.strip()
            
            if clean.startswith("```"):
                lines = clean.splitlines()
                
                if lines[0].startswith("```"):
                    lines = lines[1:]
                
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                
                clean = "\n".join(lines)
            
            return json.loads(clean.strip())
        
        except json.JSONDecodeError as e:
            logger.error(f"[JSON-PARSE-ERROR] {e}: {text[:200]}")
            return None
        
        except Exception as e:
            logger.error(f"[JSON-PARSE-UNEXPECTED-ERROR] {e}")
            return None