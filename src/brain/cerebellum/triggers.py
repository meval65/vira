import logging
from typing import Any, Dict

from src.brain.brainstem import NeuralEventBus

logger = logging.getLogger(__name__)


async def trigger_memory_compression_manual(brain) -> Dict[str, Any]:
    if not brain or not brain.hippocampus:
        return {"status": "error", "message": "Brain not initialized"}
    try:
        await NeuralEventBus.set_activity("cerebellum", "Manual Compression")
        stats_before = await brain.hippocampus.get_compression_stats()
        compressed = await brain.hippocampus.check_and_compress_memories()
        stats_after = await brain.hippocampus.get_compression_stats()
        await NeuralEventBus.clear_activity("cerebellum")
        return {
            "status": "success" if compressed else "skipped",
            "compressed": compressed,
            "stats_before": stats_before,
            "stats_after": stats_after,
        }
    except Exception as e:
        await NeuralEventBus.clear_activity("cerebellum")
        return {"status": "error", "message": str(e)}


async def trigger_maintenance_manual(brain) -> Dict[str, Any]:
    if not brain or not brain.hippocampus:
        return {"status": "error", "message": "Brain not initialized"}
    try:
        await NeuralEventBus.set_activity("cerebellum", "Manual Maintenance")
        results = {
            "schedules_cleaned": 0,
            "memories_decayed": 0,
            "knowledge_graph_optimized": False,
        }
        results["schedules_cleaned"] = await brain.hippocampus.cleanup_old_schedules(
            days_old=30
        )
        results["memories_decayed"] = await brain.hippocampus.apply_memory_decay()
        await brain.hippocampus.optimize_knowledge_graph()
        results["knowledge_graph_optimized"] = True
        await NeuralEventBus.clear_activity("cerebellum")
        return {"status": "success", "results": results}
    except Exception as e:
        await NeuralEventBus.clear_activity("cerebellum")
        return {"status": "error", "message": str(e)}
