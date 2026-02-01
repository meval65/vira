import json
from typing import Callable, Optional

from src.brain.parietal_lobe.types import Reflex


def get_crud_schemas_and_reflexes(
    create_memory_fn: Callable,
    get_memories_fn: Callable,
    update_memory_fn: Callable,
    delete_memory_fn: Callable,
    create_schedule_fn: Callable,
    get_schedules_fn: Callable,
    update_schedule_fn: Callable,
    delete_schedule_fn: Callable,
) -> list:
    return [
        Reflex(
            name="create_memory",
            description="Store a new memory. Use this when you learn something important about the user.",
            func=create_memory_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "create_memory",
                    "description": "Store a new memory about the user",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "The content of the memory to store"},
                            "memory_type": {"type": "string", "enum": ["preference", "fact", "event", "opinion", "relationship", "general"], "description": "Type of memory"},
                            "priority": {"type": "number", "description": "Priority from 0.0 to 1.0", "default": 0.5}
                        },
                        "required": ["summary", "memory_type"]
                    }
                }
            }
        ),
        Reflex(
            name="get_memories",
            description="Search and retrieve stored memories about the user.",
            func=get_memories_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "get_memories",
                    "description": "Search for stored memories",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "limit": {"type": "integer", "description": "Maximum number of memories to return", "default": 5}
                        },
                        "required": ["query"]
                    }
                }
            }
        ),
        Reflex(
            name="update_memory",
            description="Update an existing memory.",
            func=update_memory_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "update_memory",
                    "description": "Update an existing memory by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "The ID of the memory to update"},
                            "summary": {"type": "string", "description": "New content of the memory"},
                            "memory_type": {"type": "string", "enum": ["preference", "fact", "event", "opinion", "relationship", "general"]},
                            "priority": {"type": "number", "description": "New priority from 0.0 to 1.0"}
                        },
                        "required": ["memory_id"]
                    }
                }
            }
        ),
        Reflex(
            name="delete_memory",
            description="Delete a memory when it's no longer relevant or user asks to forget.",
            func=delete_memory_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "delete_memory",
                    "description": "Delete a memory by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "memory_id": {"type": "string", "description": "The ID of the memory to delete"}
                        },
                        "required": ["memory_id"]
                    }
                }
            }
        ),
        Reflex(
            name="create_schedule",
            description="Create a reminder or schedule for the user.",
            func=create_schedule_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "create_schedule",
                    "description": "Create a new schedule/reminder",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "scheduled_at": {"type": "string", "description": "ISO format datetime when to trigger (e.g. 2024-01-15T10:00:00)"},
                            "context": {"type": "string", "description": "What to remind the user about"}
                        },
                        "required": ["scheduled_at", "context"]
                    }
                }
            }
        ),
        Reflex(
            name="get_schedules",
            description="Get upcoming schedules and reminders.",
            func=get_schedules_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "get_schedules",
                    "description": "Get upcoming schedules",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hours_ahead": {"type": "integer", "description": "How many hours ahead to look", "default": 24},
                            "limit": {"type": "integer", "description": "Maximum number of schedules", "default": 10}
                        },
                        "required": []
                    }
                }
            }
        ),
        Reflex(
            name="update_schedule",
            description="Update an existing schedule.",
            func=update_schedule_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "update_schedule",
                    "description": "Update an existing schedule by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "schedule_id": {"type": "string", "description": "The ID of the schedule to update"},
                            "scheduled_at": {"type": "string", "description": "New ISO format datetime"},
                            "context": {"type": "string", "description": "New reminder content"},
                            "status": {"type": "string", "enum": ["pending", "executed", "cancelled"]}
                        },
                        "required": ["schedule_id"]
                    }
                }
            }
        ),
        Reflex(
            name="delete_schedule",
            description="Delete/cancel a schedule.",
            func=delete_schedule_fn,
            schema={
                "type": "function",
                "function": {
                    "name": "delete_schedule",
                    "description": "Delete a schedule by ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "schedule_id": {"type": "string", "description": "The ID of the schedule to delete"}
                        },
                        "required": ["schedule_id"]
                    }
                }
            }
        ),
    ]


async def create_memory_impl(
    hippocampus,
    summary: str,
    memory_type: str = "general",
    priority: float = 0.5
) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        mem_id = await hippocampus.store(summary, memory_type, priority)
        return json.dumps({"success": True, "id": str(mem_id), "message": f"Memory stored: {summary[:50]}..."})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def get_memories_impl(hippocampus, query: str, limit: int = 5) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        memories = await hippocampus.recall(query, limit=limit)
        results = [{"id": str(m.id), "summary": m.summary, "type": m.memory_type, "confidence": m.confidence} for m in memories]
        return json.dumps({"success": True, "count": len(results), "memories": results})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def update_memory_impl(
    hippocampus,
    memory_id: str,
    summary: Optional[str] = None,
    memory_type: Optional[str] = None,
    priority: Optional[float] = None
) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        update_data = {}
        if summary:
            update_data["summary"] = summary
        if memory_type:
            update_data["memory_type"] = memory_type
        if priority is not None:
            update_data["priority"] = priority
        success = await hippocampus.update_memory(memory_id, update_data)
        return json.dumps({"success": success, "message": "Memory updated" if success else "Memory not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def delete_memory_impl(hippocampus, memory_id: str) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        success = await hippocampus.delete_memory(memory_id)
        return json.dumps({"success": success, "message": "Memory deleted" if success else "Memory not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def create_schedule_impl(hippocampus, scheduled_at: str, context: str) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        from dateutil import parser as date_parser
        trigger_time = date_parser.parse(scheduled_at)
        schedule_id = await hippocampus.add_schedule(trigger_time, context)
        return json.dumps({"success": True, "id": str(schedule_id), "scheduled_at": trigger_time.isoformat(), "context": context})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def get_schedules_impl(hippocampus, hours_ahead: int = 24, limit: int = 10) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        schedules = await hippocampus.get_upcoming_schedules(hours_ahead=hours_ahead)
        results = []
        for s in schedules[:limit]:
            results.append({
                "id": str(s.get("id", s.get("_id", ""))),
                "scheduled_at": s.get("scheduled_at").isoformat() if hasattr(s.get("scheduled_at"), "isoformat") else str(s.get("scheduled_at")),
                "context": s.get("context", ""),
                "status": s.get("status", "pending")
            })
        return json.dumps({"success": True, "count": len(results), "schedules": results})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def update_schedule_impl(
    hippocampus,
    schedule_id: str,
    scheduled_at: Optional[str] = None,
    context: Optional[str] = None,
    status: Optional[str] = None
) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        update_data = {}
        if scheduled_at:
            from dateutil import parser as date_parser
            update_data["scheduled_at"] = date_parser.parse(scheduled_at)
        if context:
            update_data["context"] = context
        if status:
            update_data["status"] = status
        success = await hippocampus.update_schedule(schedule_id, update_data)
        return json.dumps({"success": success, "message": "Schedule updated" if success else "Schedule not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def delete_schedule_impl(hippocampus, schedule_id: str) -> str:
    if not hippocampus:
        return json.dumps({"error": "Hippocampus not connected"})
    try:
        success = await hippocampus.delete_schedule(schedule_id)
        return json.dumps({"success": success, "message": "Schedule deleted" if success else "Schedule not found"})
    except Exception as e:
        return json.dumps({"error": str(e)})


