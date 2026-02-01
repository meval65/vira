"""
Tools API - Expose parietal lobe tools and usage statistics.
"""

from fastapi import APIRouter
from src.brain.brainstem import get_brain

router = APIRouter(prefix="/api/tools", tags=["tools"])


@router.get("")
async def get_tools():
    """Get all available tools with usage statistics."""
    brain = await get_brain()
    
    if not brain or not brain.parietal_lobe:
        return {"tools": [], "total": 0}
    
    tools_stats = brain.parietal_lobe.get_tools_stats()
    
    # Sort by usage count descending
    tools_stats.sort(key=lambda x: x["usage_count"], reverse=True)
    
    return {
        "tools": tools_stats,
        "total": len(tools_stats),
        "total_executions": sum(t["usage_count"] for t in tools_stats)
    }


@router.get("/{tool_name}")
async def get_tool(tool_name: str):
    """Get details of a specific tool."""
    brain = await get_brain()
    
    if not brain or not brain.parietal_lobe:
        return {"error": "Brain not initialized"}
    
    for tool in brain.parietal_lobe.get_tools_stats():
        if tool["name"] == tool_name:
            return tool
    
    return {"error": f"Tool '{tool_name}' not found"}


@router.post("/{tool_name}/reset")
async def reset_tool_usage(tool_name: str):
    """Reset usage count for a tool."""
    brain = await get_brain()
    
    if not brain or not brain.parietal_lobe:
        return {"error": "Brain not initialized"}
    
    if tool_name in brain.parietal_lobe._reflexes:
        brain.parietal_lobe._reflexes[tool_name].usage_count = 0
        brain.parietal_lobe._reflexes[tool_name].last_used = None
        return {"status": "reset", "tool": tool_name}
    
    return {"error": f"Tool '{tool_name}' not found"}
