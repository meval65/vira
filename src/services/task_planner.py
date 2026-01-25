"""
Task Planner Service for Vira AI

Manages decomposition of complex user goals into executable multi-step plans.
Handles state tracking, step execution, and plan monitoring.

Features:
- LLM-based goal decomposition
- State management for plans and steps
- Execution tracking
- Recovery/Resumption of interrupted plans
"""

import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from src.database import DBConnection

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    id: Optional[int]
    plan_id: int
    order_index: int
    description: str
    action_type: str  # e.g., "search", "memory_store", "calculation", "ask_user"
    status: StepStatus = StepStatus.PENDING
    input_context: Dict = field(default_factory=dict)
    output_result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class TaskPlan:
    id: Optional[int]
    user_id: str
    goal: str
    original_request: str
    status: PlanStatus = PlanStatus.PENDING
    steps: List[TaskStep] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class TaskPlanner:
    def __init__(self, db: DBConnection, genai_client=None, model_name: str = None):
        self.db = db
        self.client = genai_client
        self.model_name = model_name
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize database tables for task planning."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS task_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                goal TEXT NOT NULL,
                original_request TEXT,
                status TEXT DEFAULT 'pending',
                context TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """, ())
        
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS task_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER NOT NULL,
                order_index INTEGER NOT NULL,
                description TEXT NOT NULL,
                action_type TEXT DEFAULT 'general',
                status TEXT DEFAULT 'pending',
                input_context TEXT,
                output_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                FOREIGN KEY(plan_id) REFERENCES task_plans(id)
            )
        """, ())
        
        # Indexes for fast retrieval
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_plans_user_status ON task_plans(user_id, status)", 
            ()
        )
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS idx_steps_plan_order ON task_steps(plan_id, order_index)", 
            ()
        )
        
        logger.info("[TASK-PLANNER] Tables initialized")

    async def create_plan(self, user_id: str, goal: str, context: Dict = None) -> Optional[TaskPlan]:
        """Create a new plan by decomposing the goal using LLM."""
        if not self.client:
            logger.warning("[TASK-PLANNER] No LLM client available for planning")
            return None
            
        steps_data = await self._decompose_goal(goal, context)
        if not steps_data:
            return None
            
        async with self._lock:
            try:
                # Insert plan
                context_json = json.dumps(context) if context else "{}"
                await self.db.execute("""
                    INSERT INTO task_plans (user_id, goal, original_request, status, context, created_at)
                    VALUES (?, ?, ?, 'pending', ?, ?)
                """, (user_id, goal, goal, context_json, datetime.now()))
                
                plan_row = await self.db.fetchone("SELECT last_insert_rowid()", ())
                plan_id = plan_row[0]
                
                # Insert steps
                steps = []
                for idx, step_info in enumerate(steps_data):
                    await self.db.execute("""
                        INSERT INTO task_steps (
                            plan_id, order_index, description, action_type, status, input_context
                        ) VALUES (?, ?, ?, ?, 'pending', ?)
                    """, (
                        plan_id, 
                        idx + 1, 
                        step_info['description'], 
                        step_info.get('type', 'general'),
                        json.dumps(step_info.get('context', {}))
                    ))
                    
                    steps.append(TaskStep(
                        id=None,  # Not needed for local object right now
                        plan_id=plan_id,
                        order_index=idx + 1,
                        description=step_info['description'],
                        action_type=step_info.get('type', 'general'),
                        input_context=step_info.get('context', {})
                    ))
                
                logger.info(f"[TASK-PLANNER] Created plan {plan_id} with {len(steps)} steps for user {user_id}")
                
                return TaskPlan(
                    id=plan_id,
                    user_id=user_id,
                    goal=goal,
                    original_request=goal,
                    status=PlanStatus.PENDING,
                    steps=steps,
                    context=context or {}
                )
                
            except Exception as e:
                logger.error(f"[TASK-PLANNER] Failed to create plan: {e}")
                await self.db.rollback()
                return None

    async def get_active_plan(self, user_id: str) -> Optional[TaskPlan]:
        """Retrieve the current active or pending plan for the user."""
        try:
            # Get latest plan that is pending or in_progress
            plan_row = await self.db.fetchone("""
                SELECT id, goal, original_request, status, context, created_at, updated_at
                FROM task_plans
                WHERE user_id=? AND status IN ('pending', 'in_progress')
                ORDER BY created_at DESC LIMIT 1
            """, (user_id,))
            
            if not plan_row:
                return None
                
            plan_id = plan_row[0]
            
            # Get steps
            step_rows = await self.db.fetchall("""
                SELECT id, order_index, description, action_type, status, input_context, output_result, created_at
                FROM task_steps
                WHERE plan_id=? ORDER BY order_index ASC
            """, (plan_id,))
            
            steps = []
            for sr in step_rows:
                steps.append(TaskStep(
                    id=sr[0],
                    plan_id=plan_id,
                    order_index=sr[1],
                    description=sr[2],
                    action_type=sr[3],
                    status=StepStatus(sr[4]),
                    input_context=json.loads(sr[5]) if sr[5] else {},
                    output_result=sr[6],
                    created_at=sr[7]
                ))
            
            return TaskPlan(
                id=plan_id,
                user_id=user_id,
                goal=plan_row[1],
                original_request=plan_row[2],
                status=PlanStatus(plan_row[3]),
                steps=steps,
                context=json.loads(plan_row[4]) if plan_row[4] else {},
                created_at=plan_row[5],
                updated_at=plan_row[6]
            )
            
        except Exception as e:
            logger.error(f"[TASK-PLANNER] Failed to get active plan: {e}")
            return None

    async def mark_step_complete(self, step_id: int, result: str):
        """Mark a specific step as complete."""
        async with self._lock:
            try:
                await self.db.execute("""
                    UPDATE task_steps 
                    SET status='completed', output_result=?, completed_at=?
                    WHERE id=?
                """, (result, datetime.now(), step_id))
                
                # Check if all steps in plan are done
                step = await self.db.fetchone("SELECT plan_id FROM task_steps WHERE id=?", (step_id,))
                if step:
                    plan_id = step[0]
                    pending_count = await self.db.fetchone("""
                        SELECT COUNT(*) FROM task_steps 
                        WHERE plan_id=? AND status NOT IN ('completed', 'skipped')
                    """, (plan_id,))
                    
                    if pending_count and pending_count[0] == 0:
                        await self.db.execute("""
                            UPDATE task_plans SET status='completed', updated_at=? WHERE id=?
                        """, (datetime.now(), plan_id))
                        logger.info(f"[TASK-PLANNER] Plan {plan_id} completed")
                        
            except Exception as e:
                logger.error(f"[TASK-PLANNER] Failed to complete step: {e}")

    async def cancel_plan(self, plan_id: int, reason: str = "user_cancelled"):
        """Cancel an active plan."""
        try:
            await self.db.execute("""
                UPDATE task_plans SET status='cancelled', updated_at=? WHERE id=?
            """, (datetime.now(), plan_id))
        except Exception as e:
            logger.error(f"[TASK-PLANNER] Failed to cancel plan: {e}")

    async def _decompose_goal(self, goal: str, context: Dict = None) -> List[Dict]:
        """Use LLM to break down goal into steps."""
        if not self.client:
            return [{"description": f"Process request: {goal}", "type": "general", "context": {}}]
            
        system_prompt = """
        You are an expert task planner. Breakdown the user's request into 3-7 logical, sequential steps.
        Each step should provide a clear description of what needs to done.
        
        Supported action types: 
        - 'search': Searching for information externally
        - 'memory': Retrieving from or storing to memory/database
        - 'reasoning': Logical deduction or calculation
        - 'ask': Asking the user for clarification
        
        Return ONLY valid JSON array of objects: 
        [{"description": "...", "type": "...", "context": {}}]
        """
        
        user_prompt = f"Goal: {goal}\nContext: {json.dumps(context) if context else '{}'}"
        
        try:
            from google.genai import types
            response = self.client.models.generate_content(
                model=self.model_name or "gemini-2.0-flash-exp",
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            
            if response.text:
                return json.loads(response.text)
        except Exception as e:
            logger.error(f"[TASK-PLANNER] Decomposition failed: {e}")
            
        # Fallback if LLM fails
        return [{"description": f"Process request: {goal}", "type": "general"}]
