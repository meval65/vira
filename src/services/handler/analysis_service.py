import json
import logging
import re
import datetime
from typing import Dict, Optional, List, Any
from google import genai
from google.genai import types
from src.config import SCHEDULE_SUMMARY_INSTRUCTION

logger = logging.getLogger(__name__)

class AnalysisService:
    MAX_HISTORY_LENGTH = 15
    MAX_CONTENT_PREVIEW = 200
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.4
    MAX_RETRIES = 2
    
    def __init__(self, client: genai.Client, tier_1_model: str, tier_2_model: str):
        self.client = client
        self.tier_1_model = tier_1_model
        self.tier_2_model = tier_2_model
    
    def _extract_json(self, text: str) -> Optional[Dict]:
        if not text:
            return None
        
        try:
            text = text.strip()
            match = re.search(r'\{.*\}', text, re.DOTALL)
            json_str = match.group(0) if match else text
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {text[:100]}... Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in JSON extraction: {e}")
            return None
    
    def _generate_content(self, model: str, prompt: str, 
                        max_tokens: int, temperature: float) -> Optional[str]:
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            return response.text if response.text else None
        except Exception as e:
            logger.warning(f"Generation failed: {e}")
            return None

    def _generate_with_retry(self, model: str, prompt: str, 
                            max_tokens: int = DEFAULT_MAX_TOKENS, 
                            temp: float = DEFAULT_TEMPERATURE) -> Optional[str]:
        for attempt in range(self.MAX_RETRIES):
            result = self._generate_content(model, prompt, max_tokens, temp)
            if result:
                return result
            logger.warning(f"Retry {attempt + 1}/{self.MAX_RETRIES} failed")
        
        return None

    def _format_history_for_analysis(self, history: List[Dict]) -> str:
        formatted_messages = []
        for message in history:
            role = message.get('role', 'unknown')
            parts = message.get('parts', [])
            content = str(parts[0])[:self.MAX_CONTENT_PREVIEW] if parts else ''
            formatted_messages.append({'role': role, 'content': content})
        
        return json.dumps(formatted_messages, ensure_ascii=False)

    def run_chat_analysis(self, history: List[Dict], old_summary: str, 
                        instruction: str) -> str:
        if not history:
            return old_summary

        prompt = (
            f"{instruction}\n\n"
            f"PREVIOUS SUMMARY:\n{old_summary}\n\n"
        )
        
        result = self._generate_with_retry(self.tier_1_model, prompt)
        return result if result else old_summary
    
    def _format_memories_for_analysis(self, memories: List[Dict]) -> str:
        memory_lines = []
        for index, memory in enumerate(memories, 1):
            summary = memory.get('summary', '')
            if summary:
                memory_lines.append(f"{index}. {summary}")
        return "\n".join(memory_lines)

    def run_memory_analysis(self, memories: List[Dict], old_memory_summary: str, 
                            instruction: str) -> str:
        if not memories:
            return old_memory_summary
        
        memory_text = self._format_memories_for_analysis(memories)

        prompt = (
            f"{instruction}\n\n"
            f"CURRENT MEMORY SUMMARY:\n{old_memory_summary}\n\n"
            f"NEW MEMORY ENTRIES:\n{memory_text}\n"
        )
        
        result = self._generate_with_retry(self.tier_1_model, prompt)
        return result if result else old_memory_summary

    def run_interaction_analysis(self, user_text: str, ai_text: str, instruction: str) -> Optional[Dict]:
        if not user_text or not ai_text:
            return None
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        prompt = (
            f"{instruction}\n\n"
            f"TIMESTAMP: {timestamp}\n"
            f"USER: {user_text}\n"
            f"AI: {ai_text}\n\n"
        )
        
        raw_text = self._generate_with_retry(
            self.tier_2_model, 
            prompt, 
            max_tokens=self.DEFAULT_MAX_TOKENS,
            temp=0.1
        )
        
        if not raw_text:
            return None
            
        return self._extract_json(raw_text)

    def _format_schedule_for_summary(self, schedule: Dict) -> str:
        time_value = schedule.get('scheduled_at')
        
        if isinstance(time_value, datetime.datetime):
            time_str = time_value.strftime('%H:%M')
        else:
            time_str = str(time_value)
        
        context = schedule.get('context', 'Tanpa detail')
        return f"[{time_str}] - {context}"

    def generate_schedule_summary(self, schedules: List[Dict]) -> str:
        if not schedules:
            return "Tidak ada jadwal mendatang."
        
        schedule_lines = [
            self._format_schedule_for_summary(schedule) 
            for schedule in schedules
        ]
        schedule_text = "\n".join(schedule_lines)

        prompt = (
            f"{SCHEDULE_SUMMARY_INSTRUCTION}\n\n"
            f"RAW SCHEDULES:\n{schedule_text}\n"
        )
        
        summary = self._generate_with_retry(
            self.tier_2_model, 
            prompt, 
            max_tokens=150, 
            temp=0.5
        )
        
        return summary if summary else "Jadwal kamu sudah tersimpan."