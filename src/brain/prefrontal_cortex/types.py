from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict

class PlanStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class StepStatus(str, Enum):
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
    action_type: str
    status: StepStatus = StepStatus.PENDING
    output_result: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    
    sub_plan_id: Optional[int] = None
    complexity_score: float = 0.0
    requires_sub_plan: bool = False
    
    def mark_complete(self, result: str = None) -> None:
        self.status = StepStatus.COMPLETED
        self.completed_at = datetime.now()
        if result:
            self.output_result = result
    
    def mark_failed(self, error: str = None) -> None:
        self.status = StepStatus.FAILED
        self.completed_at = datetime.now()
        if error:
            self.output_result = f"Error: {error}"
    
    def is_pending(self) -> bool:
        return self.status == StepStatus.PENDING
    
    def is_complete(self) -> bool:
        return self.status == StepStatus.COMPLETED
    
    def has_sub_plan(self) -> bool:
        return self.sub_plan_id is not None

@dataclass
class TaskPlan:
    id: Optional[int]
    goal: str
    original_request: str
    status: PlanStatus = PlanStatus.PENDING
    steps: List[TaskStep] = field(default_factory=list)
    context: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    priority: int = 5
    deadline: Optional[datetime] = None
    
    parent_plan_id: Optional[int] = None
    parent_step_id: Optional[int] = None
    depth: int = 0
    max_depth: int = 3
    sub_plan_ids: List[int] = field(default_factory=list)
    
    def is_sub_plan(self) -> bool:
        return self.parent_plan_id is not None
    
    def can_create_sub_plan(self) -> bool:
        return self.depth < self.max_depth
    
    def get_progress(self) -> float:
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == StepStatus.COMPLETED)
        return completed / len(self.steps)
    
    def get_current_step(self) -> Optional[TaskStep]:
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None
    
    def add_step(self, description: str, action_type: str, complexity: float = 0.0) -> TaskStep:
        step = TaskStep(
            id=None,
            plan_id=self.id,
            order_index=len(self.steps),
            description=description,
            action_type=action_type,
            complexity_score=complexity,
            requires_sub_plan=complexity > 0.7
        )
        self.steps.append(step)
        self.updated_at = datetime.now()
        return step
    
    def create_sub_plan(self, step: TaskStep, sub_goal: str, sub_steps: List[Dict]) -> Optional['TaskPlan']:
        if not self.can_create_sub_plan():
            return None
        
        sub_plan = TaskPlan(
            id=None,
            goal=sub_goal,
            original_request=step.description,
            parent_plan_id=self.id,
            parent_step_id=step.id,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            context=self.context.copy()
        )
        
        for step_def in sub_steps:
            sub_plan.add_step(
                description=step_def.get("description", ""),
                action_type=step_def.get("action_type", "tool_use"),
                complexity=step_def.get("complexity", 0.0)
            )
        
        return sub_plan
    
    def update_status(self) -> None:
        if not self.steps:
            return
        
        all_complete = all(s.status == StepStatus.COMPLETED for s in self.steps)
        any_failed = any(s.status == StepStatus.FAILED for s in self.steps)
        any_in_progress = any(s.status == StepStatus.IN_PROGRESS for s in self.steps)
        
        if all_complete:
            self.status = PlanStatus.COMPLETED
        elif any_failed:
            self.status = PlanStatus.FAILED
        elif any_in_progress:
            self.status = PlanStatus.IN_PROGRESS
        
        self.updated_at = datetime.now()

    def get_estimated_completion_time(self) -> datetime:
        from datetime import timedelta
        total_complexity = sum(s.complexity_score for s in self.steps)
        base_time_per_step = timedelta(minutes=10)
        return timedelta(minutes=len(self.steps) * 5) + (base_time_per_step * total_complexity)

    def prioritize_steps(self) -> None:
        self.steps.sort(key=lambda s: (s.complexity_score, s.order_index), reverse=True)

class IntentType(str, Enum):
    QUESTION = "question"
    STATEMENT = "statement"
    REQUEST = "request"
    GREETING = "greeting"
    COMMAND = "command"
    SMALL_TALK = "small_talk"
    CONFIRMATION = "confirmation"
    CORRECTION = "correction"

class RequestType(str, Enum):
    INFORMATION = "information"
    RECOMMENDATION = "recommendation"
    MEMORY_RECALL = "memory_recall"
    OPINION = "opinion"
    ACTION = "action"
    SCHEDULE = "schedule"
    GENERAL_CHAT = "general_chat"

@dataclass
class ProcessingMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    tool_executions: int = 0
    memory_retrievals: int = 0
    
    def record_request(self, success: bool, duration: float):
        self.total_requests += 1
        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + duration) / self.total_requests
        )

    def get_success_rate(self) -> float:
        return (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0.0
