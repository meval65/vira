from typing import Optional, Tuple
from collections import deque
from src.brain.constants import DEFAULT_PERSONA_INSTRUCTION
from src.brain.occipital_lobe.connection_manager import ConnectionManager

_custom_instruction_override: Optional[str] = None
_custom_instruction_name: str = "Default Persona"
LOG_BUFFER: deque = deque(maxlen=200)

manager = ConnectionManager()

def get_active_instruction() -> Tuple[str, str]:
    global _custom_instruction_override, _custom_instruction_name
    if _custom_instruction_override:
        return (_custom_instruction_override, _custom_instruction_name)
    return (DEFAULT_PERSONA_INSTRUCTION, "Default Persona")

def set_active_instruction(instruction: str, name: str):
    global _custom_instruction_override, _custom_instruction_name
    _custom_instruction_override = instruction
    _custom_instruction_name = name

def reset_active_instruction():
    global _custom_instruction_override, _custom_instruction_name
    _custom_instruction_override = None
    _custom_instruction_name = "Default Persona"


