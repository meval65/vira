from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class MemoryCreate(BaseModel):
    summary: str
    memory_type: str = "general"
    priority: float = 0.5
    embedding: Optional[List[float]] = None

class MemoryUpdate(BaseModel):
    summary: Optional[str] = None
    memory_type: Optional[str] = None
    priority: Optional[float] = None
    confidence: Optional[float] = None
    status: Optional[str] = None
    is_compressed: Optional[bool] = None

class TripleCreate(BaseModel):
    subject: str
    predicate: str
    object: str
    confidence: float = 0.8
    source_memory_id: Optional[str] = None

class TripleUpdate(BaseModel):
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    confidence: Optional[float] = None

class ScheduleCreate(BaseModel):
    context: str
    scheduled_at: str
    priority: int = 0
    status: str = "pending"

class ScheduleUpdate(BaseModel):
    context: Optional[str] = None
    scheduled_at: Optional[str] = None
    priority: Optional[int] = None
    status: Optional[str] = None

class EntityCreate(BaseModel):
    name: str
    entity_type: str = "unknown"
    metadata: Optional[Dict[str, Any]] = None

class EntityUpdate(BaseModel):
    name: Optional[str] = None
    entity_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class AdminProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    telegram_name: Optional[str] = None
    additional_info: Optional[str] = None

class CustomInstructionUpdate(BaseModel):
    instruction: str = Field(..., min_length=10, max_length=5000)
    name: Optional[str] = "Custom Override"

class PersonaCreate(BaseModel):
    name: str
    instruction: str
    temperature: float = 0.7
    description: Optional[str] = None

class PersonaUpdate(BaseModel):
    name: Optional[str] = None
    instruction: Optional[str] = None
    temperature: Optional[float] = None
    description: Optional[str] = None

class ChatLogEntry(BaseModel):
    role: str
    content: str
    timestamp: Optional[str] = None

class ChatLogCreate(BaseModel):
    session_id: str
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

class GlobalContextUpdate(BaseModel):
    context_text: str
    metadata: Optional[Dict[str, Any]] = None

class SystemConfigUpdate(BaseModel):
    chat_model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    proactive_check_interval: Optional[int] = None

class CompressionTriggerRequest(BaseModel):
    force: bool = False

class LogEntry(BaseModel):
    timestamp: str
    level: str
    message: str
    source: str = "system"


