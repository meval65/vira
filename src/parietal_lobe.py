import asyncio
import json
import math
import os
import shutil
import subprocess
import tempfile
import textwrap
from datetime import datetime
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass, field

import httpx
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Reflex:
    """A tool that VIRA can use deterministically."""
    name: str
    description: str
    func: Callable
    schema: Dict[str, Any]
    enabled: bool = True

class ParietalLobe:
    """
    The Parietal Lobe handles sensory information integration, spatial awareness,
    and mathematical reasoning (Reflexes & Tools).
    """
    
    def __init__(self):
        self._reflexes: Dict[str, Reflex] = {}
        self._register_default_reflexes()
        
    def _register_default_reflexes(self) -> None:
        """Register the built-in reflex tools."""
        
        # 1. Clock (Time Check)
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
        
        # 2. Calculator (Enhanced)
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
        
        # 3. Weather
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
            description="Execute Python code in a secure sandbox for complex calculations, data processing, or algorithmic tasks. Use when math expressions are too complex for simple calculator.",
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
        """Register a new reflex tool."""
        self._reflexes[reflex.name] = reflex
        print(f"  ✓ Reflex Registered (Parietal): {reflex.name}")
        
    async def execute(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Execute a reflex tool."""
        if tool_name not in self._reflexes:
            return f"Error: Tool '{tool_name}' not found."
            
        reflex = self._reflexes[tool_name]
        try:
            if asyncio.iscoroutinefunction(reflex.func):
                return await reflex.func(**args)
            return reflex.func(**args)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"
            
    def get_tools_schema(self) -> List[Dict]:
        """Get schemas for all enabled tools."""
        return [r.schema for r in self._reflexes.values() if r.enabled]
        
    def get_tool_descriptions(self) -> str:
        """Get text description of available tools for prompting."""
        return "\n".join([f"- {r.name}: {r.description}" for r in self._reflexes.values() if r.enabled])

    # --- Tool Implementations ---

    def _get_time(self) -> str:
        now = datetime.now()
        return json.dumps({
            "iso": now.isoformat(),
            "readable": now.strftime("%A, %d %B %Y - %H:%M:%S"),
            "timezone": str(now.astimezone().tzinfo)
        })

    def _calculate(self, expression: str) -> str:
        # ALLOWED: All standard math signs, decimals, and alphanumeric (for functions like sin, sqrt, pi)
        # Still filtering aggressive system commands, but allowing more math freedom.
        blocked = set(";_") # Simple block for chaining or private access
        if any(c in blocked for c in expression):
             return "Error: Restricted characters in expression"

        # Safe math environment
        math_env = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
        
        try:
            # Using eval with limited environment
            result = eval(expression, {"__builtins__": None}, math_env)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    async def _get_weather(self, location: str) -> str:
        api_key = os.getenv("METEOSOURCE_API_KEY")
        if not api_key:
            return "Error: Weather API key not configured."
            
        url = "https://www.meteosource.com/api/v1/free/point"
        params = {
            "place_id": location,
            "sections": "current",
            "language": "en",
            "units": "metric",
            "key": api_key
        }
        
        async with httpx.AsyncClient() as client:
            try:
                # First try to find place_id (Meteosource specific complexity simplified here)
                # For this free tier request, we assume 'place_id' matches or autodetects city name often
                # But typically you need a 'find_places' call first.
                # Let's assume we use a simpler public weather API or fallback for robustness later.
                # For now let's use a simpler implementation using OpenMeteo (No key) if Meteosource is complex
                
                # Check OpenMeteo for easier geocoding + weather (Zero Config)
                geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1&language=en&format=json"
                geo_resp = await client.get(geo_url)
                geo_data = geo_resp.json()
                
                if not geo_data.get("results"):
                    return f"Error: Location '{location}' not found."
                    
                lat = geo_data["results"][0]["latitude"]
                lon = geo_data["results"][0]["longitude"]
                place_name = geo_data["results"][0]["name"]
                
                weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m&timezone=auto"
                w_resp = await client.get(weather_url)
                w_data = w_resp.json()
                
                current = w_data.get("current", {})
                
                return json.dumps({
                    "location": place_name,
                    "temperature": f"{current.get('temperature_2m')} {w_data.get('current_units', {}).get('temperature_2m')}",
                    "condition_code": current.get("weather_code"),
                    "wind": f"{current.get('wind_speed_10m')} km/h",
                    "time": current.get("time")
                })
                
            except Exception as e:
                return f"Error fetching weather: {str(e)}"

    def _run_python_sandbox(self, code: str) -> str:
        if shutil.which("docker") is None:
            return "Error: Docker runtime not available on host."

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

        temp_dir = tempfile.mkdtemp(prefix="vira_py_")
        script_name = "main.py"
        script_path = os.path.join(temp_dir, script_name)
        docker_image = os.getenv("VIRA_SANDBOX_IMAGE", "python:3.11-slim")

        try:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(sandbox_code)
            os.chmod(script_path, 0o644)

            docker_cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", os.getenv("VIRA_SANDBOX_MEMORY", "256m"),
                "--pids-limit", os.getenv("VIRA_SANDBOX_PIDS", "64"),
                "--cpus", os.getenv("VIRA_SANDBOX_CPUS", "1"),
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "--user", "65534:65534",
                "--read-only",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "-e", "PYTHONDONTWRITEBYTECODE=1",
                "-v", f"{temp_dir}:/workspace:ro",
                "-w", "/workspace",
                docker_image,
                "python", "-I", "-B", script_name
            ]

            result = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=float(os.getenv("VIRA_SANDBOX_TIMEOUT", "10"))
            )

            if result.returncode != 0:
                error_msg = result.stderr.strip() or "Sandbox execution failed"
                if len(error_msg) > 500:
                    error_msg = error_msg[:500] + "..."
                return f"Error: {error_msg}"

            output = result.stdout.strip()
            if not output:
                return "Code executed successfully but produced no output. Use print() to show results."

            if len(output) > 2000:
                output = output[:2000] + "\n... (output truncated)"

            return output

        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out."
        except FileNotFoundError:
            return "Error: Docker runtime not found."
        except Exception as e:
            return f"Error executing code: {str(e)}"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
