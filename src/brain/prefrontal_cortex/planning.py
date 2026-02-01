from datetime import datetime
from typing import Optional, List, Dict, Any

from src.brain.prefrontal_cortex.types import TaskPlan, TaskStep, PlanStatus, StepStatus
from src.brain.amygdala import PlanProgressState
from src.brain.prefrontal_cortex.utils import extract_json


class PlanManager:
    COMPLEXITY_THRESHOLD: float = 0.7
    MAX_SUB_PLAN_DEPTH: int = 3

    def __init__(self, openrouter_client, amygdala=None):
        self._openrouter = openrouter_client
        self.amygdala = amygdala
        self._active_plan: Optional[TaskPlan] = None
        self._plan_history: List[TaskPlan] = []
        self._sub_plans: Dict[int, TaskPlan] = {}

    def set_amygdala(self, amygdala):
        self.amygdala = amygdala

    def get_active_plan(self) -> Optional[TaskPlan]:
        return self._active_plan

    def get_plan_history(self) -> List[TaskPlan]:
        return self._plan_history

    def get_plan_progress(self) -> Optional[Dict[str, Any]]:
        if not self._active_plan:
            return None
        
        total_steps = len(self._active_plan.steps)
        completed_steps = sum(1 for s in self._active_plan.steps if s.status == StepStatus.COMPLETED)
        
        return {
            "goal": self._active_plan.goal,
            "progress": f"{completed_steps}/{total_steps}",
            "percentage": int((completed_steps / total_steps) * 100) if total_steps > 0 else 0,
            "current_step": self.get_current_step().description if self.get_current_step() else None,
            "status": self._active_plan.status.value,
            "has_sub_plans": any(s.has_sub_plan() for s in self._active_plan.steps)
        }

    def get_current_step(self) -> Optional[TaskStep]:
        if not self._active_plan:
            return None

        for step in self._active_plan.steps:
            if step.status in [StepStatus.PENDING, StepStatus.IN_PROGRESS]:
                return step
        return None

    def should_create_plan(self, text: str) -> bool:
        if self._active_plan:
            return False

        text_lower = text.lower()
        plan_triggers = [
            "buatkan rencana", "buat jadwal", "susun rencana",
            "make a plan", "create schedule", "help me plan",
            "step by step", "langkah-langkah"
        ]

        for trigger in plan_triggers:
            if trigger in text_lower:
                return True

        return False

    async def analyze_step_complexity(self, step_description: str) -> float:
        try:
            prompt = f"""Analyze the complexity of this task step and rate it from 0.0 to 1.0.

Step: {step_description}

Consider these factors:
- Does it require multiple sub-steps? (higher complexity)
- Does it involve research or learning? (higher complexity)
- Is it a simple action? (lower complexity)
- Does it have dependencies on external resources? (higher complexity)

Output JSON:
{{"complexity_score": 0.0-1.0, "reason": "brief reason"}}"""

            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=128,
                temperature=0.1,
                tier="utility_model",
                json_mode=True
            )

            data = extract_json(response)
            if data and "complexity_score" in data:
                return max(0.0, min(1.0, float(data["complexity_score"])))
        except Exception:
            pass
        return 0.5

    async def auto_decompose_complex_step(self, step: TaskStep, parent_plan: TaskPlan) -> Optional[TaskPlan]:
        if not parent_plan.can_create_sub_plan():
            return None

        try:
            prompt = f"""Break down this complex step into 2-4 simpler sub-steps:

Parent Goal: {parent_plan.goal}
Complex Step: {step.description}

Output format (JSON):
{{
  "sub_goal": "what this step achieves",
  "sub_steps": [
    {{"order": 1, "description": "sub-step description", "action_type": "action|info|decision", "complexity": 0.0-1.0}}
  ]
}}"""

            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.1,
                tier="utility_model",
                json_mode=True
            )

            data = extract_json(response)
            if not data or "sub_steps" not in data:
                return None

            sub_plan = parent_plan.create_sub_plan(
                step=step,
                sub_goal=data.get("sub_goal", step.description),
                sub_steps=data.get("sub_steps", [])
            )

            if sub_plan:
                sub_plan.id = len(self._plan_history) + len(self._sub_plans) + 100
                step.sub_plan_id = sub_plan.id
                self._sub_plans[sub_plan.id] = sub_plan

            return sub_plan
        except Exception:
            pass
        return None

    async def create_plan(self, goal: str, priority: int = 5, deadline: Optional[datetime] = None) -> Optional[TaskPlan]:
        try:
            prompt = f"""Decompose this goal into 3-5 actionable steps:

Goal: {goal}

Output format (JSON):
{{
  "goal_summary": "brief summary",
  "steps": [
    {{"order": 1, "description": "step description", "action_type": "action|info|decision"}}
  ]
}}"""

            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.1
            )

            data = extract_json(response)
            if not data:
                return None

            plan = TaskPlan(
                id=len(self._plan_history) + 1,
                goal=data.get("goal_summary", goal),
                original_request=goal,
                status=PlanStatus.IN_PROGRESS,
                priority=priority,
                deadline=deadline
            )

            for step_data in data.get("steps", []):
                step = TaskStep(
                    id=step_data.get("order"),
                    plan_id=plan.id,
                    order_index=step_data.get("order", 0),
                    description=step_data.get("description", ""),
                    action_type=step_data.get("action_type", "action")
                )
                plan.steps.append(step)

            for step in plan.steps:
                complexity = await self.analyze_step_complexity(step.description)
                step.complexity_score = complexity
                step.requires_sub_plan = complexity > self.COMPLEXITY_THRESHOLD

                if step.requires_sub_plan and plan.can_create_sub_plan():
                    await self.auto_decompose_complex_step(step, plan)

            self._active_plan = plan
            return plan

        except Exception:
            return None

    async def _execute_sub_plan(self, sub_plan: TaskPlan) -> bool:
        for step in sub_plan.steps:
            if step.status == StepStatus.PENDING:
                step.status = StepStatus.IN_PROGRESS
                return False

        all_complete = all(s.status == StepStatus.COMPLETED for s in sub_plan.steps)
        if all_complete:
            sub_plan.status = PlanStatus.COMPLETED
        return all_complete

    async def handle_active_plan(self, user_input: str) -> Optional[str]:
        if not self._active_plan:
            return None

        text_lower = user_input.lower()

        if any(w in text_lower for w in ["batal", "cancel", "stop", "hentikan"]):
            self._active_plan.status = PlanStatus.CANCELLED
            if self.amygdala:
                self.amygdala.update_satisfaction(PlanProgressState.ABANDONED)
            self._plan_history.append(self._active_plan)
            old_plan = self._active_plan
            self._active_plan = None
            self._sub_plans.clear()
            return f"Oke, rencana '{old_plan.goal}' dibatalkan."

        if any(w in text_lower for w in ["pause", "jeda", "tunda"]):
            self._active_plan.status = PlanStatus.PAUSED
            return f"Rencana '{self._active_plan.goal}' dijeda sementara."

        if any(w in text_lower for w in ["lanjut", "continue", "resume"]):
            if self._active_plan.status == PlanStatus.PAUSED:
                self._active_plan.status = PlanStatus.IN_PROGRESS
                return f"Rencana '{self._active_plan.goal}' dilanjutkan."

        if any(w in text_lower for w in ["selesai", "done", "sudah", "completed"]):
            current_step = self.get_current_step()
            if current_step:
                if current_step.has_sub_plan():
                    sub_plan = self._sub_plans.get(current_step.sub_plan_id)
                    if sub_plan:
                        current_sub_step = None
                        for s in sub_plan.steps:
                            if s.status in [StepStatus.PENDING, StepStatus.IN_PROGRESS]:
                                current_sub_step = s
                                break
                        
                        if current_sub_step:
                            current_sub_step.status = StepStatus.COMPLETED
                            current_sub_step.completed_at = datetime.now()
                            
                            next_sub_step = None
                            for s in sub_plan.steps:
                                if s.status == StepStatus.PENDING:
                                    next_sub_step = s
                                    break
                            
                            if next_sub_step:
                                return f"Sub-langkah selesai! Lanjut: {next_sub_step.description}"
                            else:
                                sub_plan.status = PlanStatus.COMPLETED

                current_step.status = StepStatus.COMPLETED
                current_step.completed_at = datetime.now()
                if self.amygdala:
                    self.amygdala.update_satisfaction(PlanProgressState.ON_TRACK)

                next_step = self.get_current_step()
                if next_step:
                    if next_step.has_sub_plan():
                        sub_plan = self._sub_plans.get(next_step.sub_plan_id)
                        if sub_plan and sub_plan.steps:
                            first_sub = sub_plan.steps[0]
                            return f"Mantap! Langkah selanjutnya: {next_step.description}\n📋 Sub-langkah: {first_sub.description}"
                    return f"Mantap! Langkah selanjutnya: {next_step.description}"
                else:
                    self._active_plan.status = PlanStatus.COMPLETED
                    if self.amygdala:
                        self.amygdala.update_satisfaction(PlanProgressState.COMPLETED)
                    self._plan_history.append(self._active_plan)
                    goal = self._active_plan.goal
                    self._active_plan = None
                    self._sub_plans.clear()
                    return f"Selamat! Rencana '{goal}' selesai semua. Bangga sama lu!"

        if any(w in text_lower for w in ["skip", "lewati"]):
            current_step = self.get_current_step()
            if current_step:
                current_step.status = StepStatus.SKIPPED
                next_step = self.get_current_step()
                if next_step:
                    return f"Langkah dilewati. Selanjutnya: {next_step.description}"

        if any(w in text_lower for w in ["detail", "sub-step", "breakdown"]):
            current_step = self.get_current_step()
            if current_step and current_step.has_sub_plan():
                sub_plan = self._sub_plans.get(current_step.sub_plan_id)
                if sub_plan:
                    sub_steps_text = "\n".join([
                        f"  {i+1}. {'✅' if s.status == StepStatus.COMPLETED else '⬜'} {s.description}"
                        for i, s in enumerate(sub_plan.steps)
                    ])
                    return f"📋 Sub-langkah untuk '{current_step.description}':\n{sub_steps_text}"

        return None


