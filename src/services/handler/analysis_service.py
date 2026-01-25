"""
Analysis Service for Vira AI.
Handles AI-powered analysis of chat history, memories, interactions, and schedules.
All methods support async operation with proper error handling.
"""

import json
import logging
import re
import datetime
import asyncio
from typing import Dict, Optional, List, Any

from google import genai
from google.genai import types

from src.config import SCHEDULE_SUMMARY_INSTRUCTION

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for AI-powered analysis operations."""
    
    MAX_HISTORY_LENGTH = 15
    MAX_CONTENT_PREVIEW = 200
    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.4
    MAX_RETRIES = 2
    RETRY_DELAY = 1.0  # seconds

    def __init__(self, client: genai.Client, tier_1_model: str, tier_2_model: str):
        """
        Initialize the analysis service.
        
        Args:
            client: Google GenAI client
            tier_1_model: Primary model for complex analysis
            tier_2_model: Secondary model for simpler tasks
        """
        self.client = client
        self.tier_1_model = tier_1_model
        self.tier_2_model = tier_2_model
        self._generation_lock = asyncio.Lock()

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text with robust parsing."""
        if not text:
            return None

        try:
            text = text.strip()
            
            # Try to find JSON object in text
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                json_str = text
            
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed: {text[:100]}... Error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in JSON extraction: {e}")
            return None

    async def _generate_content(
        self, 
        model: str, 
        prompt: str,
        max_tokens: int, 
        temperature: float
    ) -> Optional[str]:
        """Generate content using the AI model asynchronously."""
        try:
            # Use asyncio.to_thread for thread-safe generation
            def _sync_generate():
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                        max_output_tokens=max_tokens
                    )
                )
                return response.text if response.text else None
            
            return await asyncio.to_thread(_sync_generate)
        except Exception as e:
            logger.warning(f"Generation failed on {model}: {e}")
            return None

    async def _generate_with_retry(
        self, 
        model: str, 
        prompt: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temp: float = DEFAULT_TEMPERATURE
    ) -> Optional[str]:
        """Generate content with automatic retry on failure."""
        for attempt in range(self.MAX_RETRIES):
            result = await self._generate_content(model, prompt, max_tokens, temp)
            if result:
                return result
            
            logger.warning(f"Retry {attempt + 1}/{self.MAX_RETRIES} failed for {model}")
            
            if attempt < self.MAX_RETRIES - 1:
                await asyncio.sleep(self.RETRY_DELAY)

        return None

    def _format_history_for_analysis(self, history: List[Dict]) -> str:
        """Format chat history for analysis prompt."""
        formatted_messages = []
        for message in history[-self.MAX_HISTORY_LENGTH:]:
            role = message.get('role', 'unknown')
            parts = message.get('parts', [])
            content = str(parts[0])[:self.MAX_CONTENT_PREVIEW] if parts else ''
            formatted_messages.append({'role': role, 'content': content})

        return json.dumps(formatted_messages, ensure_ascii=False)

    async def run_chat_analysis(
        self, 
        history: List[Dict], 
        old_summary: str,
        instruction: str
    ) -> str:
        """
        Analyze chat history and update context summary.
        
        Args:
            history: List of chat messages
            old_summary: Previous context summary
            instruction: Analysis instruction prompt
            
        Returns:
            Updated context summary
        """
        if not history:
            return old_summary

        formatted_history = self._format_history_for_analysis(history)
        
        prompt = (
            f"{instruction}\n\n"
            f"PREVIOUS SUMMARY:\n{old_summary}\n\n"
            f"NEW CONVERSATION:\n{formatted_history}\n"
        )

        result = await self._generate_with_retry(self.tier_1_model, prompt)
        return result if result else old_summary

    def _format_memories_for_analysis(self, memories: List[Dict]) -> str:
        """Format memories for analysis prompt."""
        memory_lines = []
        for index, memory in enumerate(memories, 1):
            summary = memory.get('summary', '')
            mem_type = memory.get('memory_type', 'general')
            if summary:
                memory_lines.append(f"{index}. [{mem_type}] {summary}")
        return "\n".join(memory_lines)

    async def run_memory_analysis(
        self, 
        memories: List[Dict], 
        old_memory_summary: str,
        instruction: str
    ) -> str:
        """
        Analyze memories and update memory summary.
        
        Args:
            memories: List of memory entries
            old_memory_summary: Previous memory summary
            instruction: Analysis instruction prompt
            
        Returns:
            Updated memory summary
        """
        if not memories:
            return old_memory_summary

        memory_text = self._format_memories_for_analysis(memories)

        prompt = (
            f"{instruction}\n\n"
            f"CURRENT MEMORY SUMMARY:\n{old_memory_summary}\n\n"
            f"NEW MEMORY ENTRIES:\n{memory_text}\n"
        )

        result = await self._generate_with_retry(self.tier_1_model, prompt)
        return result if result else old_memory_summary

    async def run_interaction_analysis(
        self, 
        user_text: str, 
        ai_text: str, 
        instruction: str
    ) -> Optional[Dict]:
        """
        Analyze user-AI interaction for memory and schedule extraction.
        
        Args:
            user_text: User's message
            ai_text: AI's response
            instruction: Analysis instruction prompt
            
        Returns:
            Parsed analysis result as dictionary, or None on failure
        """
        if not user_text or not ai_text:
            return None

        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        prompt = (
            f"{instruction}\n\n"
            f"TIMESTAMP: {timestamp}\n"
            f"USER: {user_text}\n"
            f"AI: {ai_text}\n\n"
            "Respond with valid JSON only."
        )

        raw_text = await self._generate_with_retry(
            self.tier_2_model,
            prompt,
            max_tokens=self.DEFAULT_MAX_TOKENS,
            temp=0.1
        )

        if not raw_text:
            return None

        return self._extract_json(raw_text)

    def _format_schedule_for_summary(self, schedule: Dict) -> str:
        """Format a schedule entry for summary generation."""
        time_value = schedule.get('scheduled_at')

        if isinstance(time_value, datetime.datetime):
            time_str = time_value.strftime('%Y-%m-%d %H:%M')
        else:
            time_str = str(time_value)

        context = schedule.get('context', 'Tanpa detail')
        priority = schedule.get('priority', 0)
        
        return f"[{time_str}] (P{priority}) - {context}"

    async def generate_schedule_summary(self, schedules: List[Dict]) -> str:
        """
        Generate a human-readable summary of upcoming schedules.
        
        Args:
            schedules: List of schedule entries
            
        Returns:
            Formatted schedule summary
        """
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

        summary = await self._generate_with_retry(
            self.tier_2_model,
            prompt,
            max_tokens=150,
            temp=0.5
        )

        return summary if summary else "Jadwal kamu sudah tersimpan."
    
    async def analyze_intent(self, text: str, instruction: str) -> Optional[Dict]:
        """
        Analyze user intent from text.
        
        Args:
            text: User input text
            instruction: Intent extraction instruction
            
        Returns:
            Intent analysis as dictionary
        """
        if not text:
            return None
            
        prompt = f"{instruction}\n\nINPUT: {text}\n"
        
        raw_text = await self._generate_with_retry(
            self.tier_2_model,
            prompt,
            max_tokens=256,
            temp=0.1
        )
        
        if not raw_text:
            return None
            
        return self._extract_json(raw_text)