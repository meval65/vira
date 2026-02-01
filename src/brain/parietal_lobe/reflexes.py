import os
import json
import math
import subprocess
import tempfile
import textwrap
from datetime import datetime
from typing import Dict, Any, Set

import httpx

from src.brain.parietal_lobe.types import Reflex
from src.brain.parietal_lobe.cache import LRUCache


CACHE_TTL_SECONDS = 3600
MAX_CACHE_SIZE = 100
DOCKER_TIMEOUT = 5
MAX_OUTPUT_LENGTH = 2000
MAX_ERROR_LENGTH = 500


def build_safe_math_env() -> Dict[str, Any]:
    return {k: v for k, v in math.__dict__.items() if not k.startswith("_")}


def register_default_reflexes(
    register_fn,
    cache: LRUCache,
    safe_math_env: Dict[str, Any],
    get_http_client_fn,
    get_time_fn,
    calculate_fn,
    get_weather_fn,
    run_python_sandbox_fn,
) -> None:
    register_fn(Reflex(
        name="get_current_time",
        description="Get the exact current local time info.",
        func=get_time_fn,
        schema={
            "type": "function",
            "function": {
                "name": "get_current_time",
                "description": "Get current local time",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ))

    register_fn(Reflex(
        name="calculate",
        description="Evaluate a mathematical expression (Supports complex scientific math).",
        func=calculate_fn,
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
        register_fn(Reflex(
            name="get_weather",
            description="Get current weather for a specific location.",
            func=get_weather_fn,
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

    register_fn(Reflex(
        name="run_python_code",
        description="Execute Python code in a secure sandbox for complex calculations, data processing, or algorithmic tasks.",
        func=run_python_sandbox_fn,
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


def get_time_impl() -> str:
    now = datetime.now()
    return json.dumps({
        "iso": now.isoformat(),
        "readable": now.strftime("%A, %d %B %Y - %H:%M:%S"),
        "timezone": str(now.astimezone().tzinfo),
        "unix_timestamp": int(now.timestamp())
    })


def calculate_impl(expression: str, cache: LRUCache, safe_math_env: Dict[str, Any]) -> str:
    cached = cache.get(f"calc:{expression}")
    if cached:
        return cached

    if not expression or len(expression) > 500:
        return json.dumps({"error": "Expression too long or empty"})

    blocked: Set[str] = {';', '_', '__', 'import', 'exec', 'eval', 'compile', 'open', 'file'}
    expr_lower = expression.lower()
    if any(b in expr_lower for b in blocked):
        return json.dumps({"error": "Restricted characters or keywords in expression"})

    try:
        result = eval(expression, {"__builtins__": None}, safe_math_env)
        res_str = json.dumps({"result": str(result), "expression": expression})
        cache.set(f"calc:{expression}", res_str)
        return res_str
    except Exception as e:
        return json.dumps({"error": str(e)})


async def get_weather_impl(location: str, cache: LRUCache, get_http_client_fn) -> str:
    cached = cache.get(f"weather:{location}")
    if cached:
        return cached

    client = await get_http_client_fn()
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
        cache.set(f"weather:{location}", result)
        return result

    except Exception as e:
        return json.dumps({"error": f"Error fetching weather: {str(e)}"})


def run_python_sandbox_impl(code: str, cache: LRUCache) -> str:
    cached = cache.get(f"py:{code}")
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
            timeout=DOCKER_TIMEOUT
        )

        if result.returncode != 0:
            error_msg = result.stderr.strip()
            if len(error_msg) > MAX_ERROR_LENGTH:
                error_msg = error_msg[:MAX_ERROR_LENGTH] + "..."
            return json.dumps({"error": error_msg})

        output = result.stdout.strip()
        if not output:
            return json.dumps({"message": "Code executed successfully but produced no output. Use print() to show results."})

        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"

        response = json.dumps({"output": output})
        cache.set(f"py:{code}", response)
        return response

    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'kill', '--signal=KILL'], timeout=2, capture_output=True)
        return json.dumps({"error": f"Code execution timed out (max {DOCKER_TIMEOUT} seconds)"})
    except FileNotFoundError:
        return json.dumps({"error": "Docker is not installed or not in PATH"})
    except Exception as e:
        return json.dumps({"error": f"Error executing code: {str(e)}"})
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass