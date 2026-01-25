"""
Custom Instruction Manager for Vira AI

Manages user-defined behavioral rules and instructions.
Allows users to customize how Vira interacts, responds, and handles specific topics.

Features:
- Add, update, delete custom instructions
- Activate/deactivate specific rules
- Categorize instructions (style, boundary, preference)
- Inject active instructions into system prompt
"""

import logging
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field

from src.database import DBConnection

logger = logging.getLogger(__name__)

@dataclass
class CustomInstruction:
    id: Optional[int]
    user_id: str
    instruction: str
    category: str # style, boundary, preference, roleplay
    is_active: bool = True
    priority: int = 1 # Higher priority overrides lower
    created_at: datetime = field(default_factory=datetime.now)

class CustomInstructionManager:
    def __init__(self, db: DBConnection):
        self.db = db
        self._cache: Dict[str, List[CustomInstruction]] = {}
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize database table for custom instructions."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS custom_instructions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                instruction TEXT NOT NULL,
                category TEXT DEFAULT 'preference',
                is_active BOOLEAN DEFAULT 1,
                priority INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, ())
        
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_instructions_user ON custom_instructions(user_id, is_active)",
            ()
        )
        logger.info("[INSTRUCTION-MGR] Tables initialized")
        
    async def add_instruction(self, user_id: str, text: str, category: str = "preference", priority: int = 1) -> bool:
        """Add a new custom instruction."""
        async with self._lock:
            try:
                await self.db.execute("""
                    INSERT INTO custom_instructions (user_id, instruction, category, priority)
                    VALUES (?, ?, ?, ?)
                """, (user_id, text, category, priority))
                
                self._invalidate_cache(user_id)
                logger.info(f"[INSTRUCTION-MGR] Added for {user_id}: {text[:30]}...")
                return True
            except Exception as e:
                logger.error(f"[INSTRUCTION-MGR] Add failed: {e}")
                return False
                
    async def get_active_instructions(self, user_id: str) -> List[CustomInstruction]:
        """Get all active instructions for a user, sorted by priority."""
        if user_id in self._cache:
            return self._cache[user_id]
            
        async with self._lock:
            rows = await self.db.fetchall("""
                SELECT id, instruction, category, is_active, priority, created_at
                FROM custom_instructions
                WHERE user_id=? AND is_active=1
                ORDER BY priority DESC, created_at DESC
            """, (user_id,))
            
            instructions = [
                CustomInstruction(
                    id=r[0], user_id=user_id, instruction=r[1], category=r[2],
                    is_active=bool(r[3]), priority=r[4], created_at=r[5]
                )
                for r in rows
            ]
            
            self._cache[user_id] = instructions
            return instructions

    async def compile_system_prompt_addition(self, user_id: str) -> str:
        """Compile active instructions into a string for the system prompt."""
        instructions = await self.get_active_instructions(user_id)
        if not instructions:
            return ""
            
        grouped = {}
        for inst in instructions:
            if inst.category not in grouped:
                grouped[inst.category] = []
            grouped[inst.category].append(inst.instruction)
            
        lines = ["\n[USER CUSTOM INSTRUCTIONS]"]
        
        if "roleplay" in grouped:
            lines.append("Role/Persona:")
            for item in grouped["roleplay"]:
                lines.append(f"- {item}")
                
        if "boundary" in grouped:
            lines.append("\nBoundaries/Constraints:")
            for item in grouped["boundary"]:
                lines.append(f"- {item}")
                
        if "style" in grouped:
            lines.append("\nCommunication Style:")
            for item in grouped["style"]:
                lines.append(f"- {item}")
                
        if "preference" in grouped:
            lines.append("\nPreferences:")
            for item in grouped["preference"]:
                lines.append(f"- {item}")
                
        return "\n".join(lines)
        
    async def update_instruction_status(self, instruction_id: int, is_active: bool):
        """Toggle instruction active status."""
        async with self._lock:
            # Get user_id first to invalidate cache
            row = await self.db.fetchone("SELECT user_id FROM custom_instructions WHERE id=?", (instruction_id,))
            if not row:
                return
                
            user_id = row[0]
            
            await self.db.execute("""
                UPDATE custom_instructions SET is_active=? WHERE id=?
            """, (is_active, instruction_id))
            
            self._invalidate_cache(user_id)
            
    async def delete_instruction(self, instruction_id: int):
        """Delete an instruction."""
        async with self._lock:
            row = await self.db.fetchone("SELECT user_id FROM custom_instructions WHERE id=?", (instruction_id,))
            if not row:
                return
            
            user_id = row[0]
            await self.db.execute("DELETE FROM custom_instructions WHERE id=?", (instruction_id,))
            self._invalidate_cache(user_id)

    def _invalidate_cache(self, user_id: str):
        if user_id in self._cache:
            del self._cache[user_id]
