import asyncio
import json
import math
import hashlib
import os
import subprocess
import tempfile
import textwrap
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import OrderedDict

import httpx
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Reflex:
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any]
    enabled: bool = True

class LRUCache:
    def __init__(self, max_size: int, ttl: int):
        self._cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def get(self, key: str) -> Optional[Any]:
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        if hashed_key in self._cache:
            value, timestamp = self._cache[hashed_key]
            if (datetime.now() - timestamp).total_seconds() < self._ttl:
                self._cache.move_to_end(hashed_key)
                return value
            del self._cache[hashed_key]
        return None

    def set(self, key: str, value: Any) -> None:
        hashed_key = hashlib.md5(key.encode()).hexdigest()
        if hashed_key in self._cache:
            self._cache.move_to_end(hashed_key)
        self._cache[hashed_key] = (value, datetime.now())
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

class ParietalLobe:
    CACHE_TTL_SECONDS = 3600
    MAX_CACHE_SIZE = 100
    DOCKER_TIMEOUT = 5
    MAX_OUTPUT_LENGTH = 2000
    MAX_ERROR_LENGTH = 500

    def __init__(self):
        self._brain = None
        self._reflexes: Dict[str, Reflex] = {}
        self._cache = LRUCache(self.MAX_CACHE_SIZE, self.CACHE_TTL_SECONDS)
        
        self._http_client: Optional[httpx.AsyncClient] = None
        self._safe_math_env = self._build_safe_math_env()
        self._register_default_reflexes()
        self._register_crud_tools()

    def bind_brain(self, brain) -> None:
        self._brain = brain

    @property
    def hippocampus(self):
        return self._brain.hippocampus if self._brain else None

    def _build_safe_math_env(self) -> Dict[str, Any]:
        return {k: v for k, v in math.__dict__.items() if not k.startswith("_")}

    async def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=10.0)
        return self._http_client

    async def close(self) -> None:
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def set_hippocampus(self, hippocampus) -> None:
        self._hippocampus = hippocampus
        self._register_crud_tools()

    def _register_default_reflexes(self) -> None:
        self.register(Reflex(
            name="get_current_time",
            description="Get the exact current local time info.",
            func=self._get_time,
            schema={
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current local time",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ))

        self.register(Reflex(
            name="calculate",
            description="Evaluate a mathematical expression (Supports complex scientific math).",
            func=self._calculate,
            schema={
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "Evaluate math expression",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression (e.g. 'sqrt(144) * sin(30)')"}
                        },
                        "required": ["expression"]
                    }
                }
            }
        ))

        if os.getenv("METEOSOURCE_API_KEY"):
            self.register(Reflex(
                name="get_weather",
                description="Get current weather for a specific location.",
                func=self._get_weather,
                schema={
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name or location"}
                            },
                            "required": ["location"]
                        }
                    }
                }
            ))

        self.register(Reflex(
            name="run_python_code",
            description="Execute Python code in a secure sandbox for complex calculations, data processing, or algorithmic tasks.",
            func=self._run_python_sandbox,
            schema={
                "type": "function",
                "function": {
                    "name": "run_python_code",
                    "description": "Execute Python code in sandbox",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Python code to execute. Must print() results."}
                        },
                        "required": ["code"]
                    }
                }
            }
        ))

    def register(self, reflex: Reflex) -> None:
        self._reflexes[reflex.name] = reflex
        print(f"  ✓ Reflex Registered (Parietal): {reflex.name}")

    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        if tool_name not in self._reflexes:
            return json.dumps({"error": f"Tool '{tool_name}' not found"})

        reflex = self._reflexes[tool_name]
        try:
            if asyncio.iscoroutinefunction(reflex.func):
                return await reflex.func(**args)
            return reflex.func(**args)
        except Exception as e:
            return json.dumps({"error": f"Error executing {tool_name}: {str(e)}"})

    def get_tools_schema(self) -> List[Dict]:
        return [r.schema for r in self._reflexes.values() if r.enabled]

    def get_tool_descriptions(self) -> str:
        return "\n".join([f"- {r.name}: {r.description}" for r in self._reflexes.values() if r.enabled])

    def _get_time(self) -> str:
        now = datetime.now()
        return json.dumps({
            "iso": now.isoformat(),
            "readable": now.strftime("%A, %d %B %Y - %H:%M:%S"),
            "timezone": str(now.astimezone().tzinfo),
            "unix_timestamp": int(now.timestamp())
        })

    def _calculate(self, expression: str) -> str:
        cached = self._cache.get(f"calc:{expression}")
        if cached:
            return cached

        if not expression or len(expression) > 500:
            return json.dumps({"error": "Expression too long or empty"})

        blocked: Set[str] = {';', '_', '__', 'import', 'exec', 'eval', 'compile', 'open', 'file'}
        expr_lower = expression.lower()
        if any(b in expr_lower for b in blocked):
            return json.dumps({"error": "Restricted characters or keywords in expression"})

        try:
            result = eval(expression, {"__builtins__": None}, self._safe_math_env)
            res_str = json.dumps({"result": str(result), "expression": expression})
            self._cache.set(f"calc:{expression}", res_str)
            return res_str
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _get_weather(self, location: str) -> str:
        cached = self._cache.get(f"weather:{location}")
        if cached:
            return cached

        client = await self._get_http_client()
        try:
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
            geo_resp = await client.get(geo_url)
            geo_data = geo_resp.json()

            if not geo_data.get("results"):
                return json.dumps({"error": f"Location '{location}' not found"})

            lat = geo_data["results"][0]["latitude"]
            lon = geo_data["results"][0]["longitude"]
            place_name = geo_data["results"][0]["name"]

            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
            w_resp = await client.get(weather_url)
            w_data = w_resp.json()

            current = w_data.get("current", {})
            result = json.dumps({
                "location": place_name,
                "temperature": f"{current.get('temperature_2m')} {w_data.get('current_units', {}).get('temperature_2m', '°C')}",
                "humidity": f"{current.get('relative_humidity_2m')}%",
                "condition_code": current.get("weather_code"),
                "wind_speed": f"{current.get('wind_speed_10m')} km/h",
                "time": current.get("time")
            })
            self._cache.set(f"weather:{location}", result)
            return result

        except Exception as e:
            return json.dumps({"error": f"Error fetching weather: {str(e)}"})

    def _run_python_sandbox(self, code: str) -> str:
        cached = self._cache.get(f"py:{code}")
        if cached:
            return cached

        BLOCKED_IMPORTS = {
            'os', 'sys', 'subprocess', 'shutil', 'pathlib',
            'socket', 'requests', 'urllib', 'http', 'ftplib',
            'pickle', 'shelve', 'marshal',
            'ctypes', 'multiprocessing', 'threading',
            'importlib', '__import__', 'exec', 'eval', 'compile',
            'open', 'file', 'input', 'raw_input',
            'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr',
        }

        code_lower = code.lower()
        for blocked in BLOCKED_IMPORTS:
            if blocked in code_lower:
                return json.dumps({"error": f"'{blocked}' is not allowed in sandbox for security reasons"})

        sandbox_code = textwrap.dedent(f'''
import math
import json
import statistics
from decimal import Decimal
from fractions import Fraction
from collections import Counter, defaultdict
from itertools import permutations, combinations
from functools import reduce

{code}
''')

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(sandbox_code)
                temp_path = f.name

            docker_command = [
                'docker', 'run',
                '--rm',
                '--network', 'none',
                '--memory', '128m',
                '--cpus', '0.5',
                '--pids-limit', '50',
                '--read-only',
                '--tmpfs', '/tmp:rw,noexec,nosuid,size=10m',
                '--security-opt', 'no-new-privileges',
                '--cap-drop', 'ALL',
                '-v', f'{temp_path}:/sandbox/code.py:ro',
                '-w', '/sandbox',
                '--user', '65534:65534',
                'python:3.11-alpine',
                'python', '/sandbox/code.py'
            ]

            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=self.DOCKER_TIMEOUT
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip()
                if len(error_msg) > self.MAX_ERROR_LENGTH:
                    error_msg = error_msg[:self.MAX_ERROR_LENGTH] + "..."
                return json.dumps({"error": error_msg})

            output = result.stdout.strip()
            if not output:
                return json.dumps({"message": "Code executed successfully but produced no output. Use print() to show results."})

            if len(output) > self.MAX_OUTPUT_LENGTH:
                output = output[:self.MAX_OUTPUT_LENGTH] + "\n... (output truncated)"

            response = json.dumps({"output": output})
            self._cache.set(f"py:{code}", response)
            return response

        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'kill', '--signal=KILL'], timeout=2, capture_output=True)
            return json.dumps({"error": f"Code execution timed out (max {self.DOCKER_TIMEOUT} seconds)"})
        except FileNotFoundError:
            return json.dumps({"error": "Docker is not installed or not in PATH"})
        except Exception as e:
            return json.dumps({"error": f"Error executing code: {str(e)}"})
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except:
                    pass

    def _register_crud_tools(self) -> None:
        self.register(Reflex(
            name="create_memory",
            description="Store a new memory. Use this when you learn something important about the user.",
            func=self._create_memory,
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
        ))

        self.register(Reflex(
            name="get_memories",
            description="Search and retrieve stored memories about the user.",
            func=self._get_memories,
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
        ))

        self.register(Reflex(
            name="update_memory",
            description="Update an existing memory.",
            func=self._update_memory,
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
        ))

        self.register(Reflex(
            name="delete_memory",
            description="Delete a memory when it's no longer relevant or user asks to forget.",
            func=self._delete_memory,
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
        ))

        self.register(Reflex(
            name="create_schedule",
            description="Create a reminder or schedule for the user.",
            func=self._create_schedule,
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
        ))

        self.register(Reflex(
            name="get_schedules",
            description="Get upcoming schedules and reminders.",
            func=self._get_schedules,
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
        ))

        self.register(Reflex(
            name="update_schedule",
            description="Update an existing schedule.",
            func=self._update_schedule,
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
        ))

        self.register(Reflex(
            name="delete_schedule",
            description="Delete/cancel a schedule.",
            func=self._delete_schedule,
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
        ))

    async def _create_memory(self, summary: str, memory_type: str = "general", priority: float = 0.5) -> str:
        if not self._hippocampus:
            return json.dumps({"error": "Hippocampus not connected"})
        try:
            mem_id = await self._hippocampus.store(summary, memory_type, priority)
            return json.dumps({"success": True, "id": str(mem_id), "message": f"Memory stored: {summary[:50]}..."})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _get_memories(self, query: str, limit: int = 5) -> str:
        if not self._hippocampus:
            return json.dumps({"error": "Hippocampus not connected"})
        try:
            memories = await self._hippocampus.recall(query, limit=limit)
            results = [{"id": str(m.id), "summary": m.summary, "type": m.memory_type, "confidence": m.confidence} for m in memories]
            return json.dumps({"success": True, "count": len(results), "memories": results})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _update_memory(self, memory_id: str, summary: str = None, memory_type: str = None, priority: float = None) -> str:
        if not self._hippocampus:
            return json.dumps({"error": "Hippocampus not connected"})
        try:
            update_data = {}
            if summary:
                update_data["summary"] = summary
            if memory_type:
                update_data["memory_type"] = memory_type
            if priority is not None:
                update_data["priority"] = priority
            success = await self._hippocampus.update_memory(memory_id, update_data)
            return json.dumps({"success": success, "message": "Memory updated" if success else "Memory not found"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _delete_memory(self, memory_id: str) -> str:
        if not self.hippocampus:
            return json.dumps({"error": "Hippocampus not connected"})
        try:
            success = await self.hippocampus.delete_memory(memory_id)
            return json.dumps({"success": success, "message": "Memory deleted" if success else "Memory not found"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _create_schedule(self, scheduled_at: str, context: str) -> str:
        if not self.hippocampus:
            return json.dumps({"error": "Hippocampus not connected"})
        try:
            from dateutil import parser as date_parser
            trigger_time = date_parser.parse(scheduled_at)
            schedule_id = await self.hippocampus.add_schedule(trigger_time, context)
            return json.dumps({"success": True, "id": str(schedule_id), "scheduled_at": trigger_time.isoformat(), "context": context})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _get_schedules(self, hours_ahead: int = 24, limit: int = 10) -> str:
        if not self.hippocampus:
            return json.dumps({"error": "Hippocampus not connected"})
        try:
            schedules = await self.hippocampus.get_upcoming_schedules(hours_ahead=hours_ahead)
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

    async def _update_schedule(self, schedule_id: str, scheduled_at: str = None, context: str = None, status: str = None) -> str:
        if not self.hippocampus:
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
            success = await self._hippocampus.update_schedule(schedule_id, update_data)
            return json.dumps({"success": success, "message": "Schedule updated" if success else "Schedule not found"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    async def _delete_schedule(self, schedule_id: str) -> str:
        if not self._hippocampus:
            return json.dumps({"error": "Hippocampus not connected"})
        try:
            success = await self._hippocampus.delete_schedule(schedule_id)
            return json.dumps({"success": success, "message": "Schedule deleted" if success else "Schedule not found"})
        except Exception as e:
            return json.dumps({"error": str(e)})