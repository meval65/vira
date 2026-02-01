import logging

from src.brain.brainstem import NeuralEventBus

from src.brain.cerebellum.context import get_brain_from_context
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


async def background_maintenance(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain:
        return
    try:
        await NeuralEventBus.set_activity("cerebellum", "Daily Maintenance")
        if brain.hippocampus:
            await brain.hippocampus.cleanup_old_schedules(days_old=30)
            await brain.hippocampus.apply_memory_decay()
            await brain.hippocampus.optimize_knowledge_graph()
        await NeuralEventBus.clear_activity("cerebellum")
    except Exception as e:
        logger.error("Maintenance failed: %s", e)
        await NeuralEventBus.clear_activity("cerebellum")


async def background_memory_optimization(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain or not brain.hippocampus:
        return
    try:
        await NeuralEventBus.set_activity("cerebellum", "Memory Optimization")
        await brain.hippocampus.consolidate_memories()
        await brain.hippocampus.optimize_knowledge_graph()
        await NeuralEventBus.clear_activity("cerebellum")
    except Exception as e:
        logger.error("Memory optimization failed: %s", e)
        await NeuralEventBus.clear_activity("cerebellum")


async def background_memory_compression(context: ContextTypes.DEFAULT_TYPE) -> None:
    brain = get_brain_from_context(context)
    if not brain or not brain.hippocampus:
        return
    try:
        await NeuralEventBus.set_activity("cerebellum", "Memory Compression Check")
        await NeuralEventBus.emit("cerebellum", "hippocampus", "compression_check")
        stats_before = await brain.hippocampus.get_compression_stats()
        compressed = await brain.hippocampus.check_and_compress_memories()
        if compressed:
            stats_after = await brain.hippocampus.get_compression_stats()
            await NeuralEventBus.emit(
                "cerebellum", "dashboard", "compression_complete",
                payload={
                    "status": "success",
                    "memories_before": stats_before.get("uncompressed_count", 0),
                    "memories_after": stats_after.get("uncompressed_count", 0),
                    "global_context_length": stats_after.get("global_context_length", 0),
                    "compression_version": stats_after.get("compression_version", 0),
                },
            )
        else:
            await NeuralEventBus.emit(
                "cerebellum", "dashboard", "compression_skipped",
                payload={
                    "reason": "threshold_not_met",
                    "uncompressed_count": stats_before.get("uncompressed_count", 0),
                    "threshold": brain.hippocampus.COMPRESSION_THRESHOLD,
                },
            )
        await NeuralEventBus.clear_activity("cerebellum")
    except Exception as e:
        logger.error("Memory compression failed: %s", e)
        await NeuralEventBus.emit(
            "cerebellum", "dashboard", "compression_failed",
            payload={"error": str(e)},
        )
        await NeuralEventBus.clear_activity("cerebellum")


