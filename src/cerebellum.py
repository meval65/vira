import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from telegram.ext import ContextTypes

from src.brainstem import NeuralEventBus

logger = logging.getLogger(__name__)

def get_brain_from_context(context: ContextTypes.DEFAULT_TYPE):
    """Get brain instance from context."""
    return context.bot_data.get('brain')

async def background_maintenance(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic maintenance tasks - runs daily."""
    brain = get_brain_from_context(context)
    if not brain:
        logger.warning("Brain not initialized, skipping maintenance")
        return

    try:
        await NeuralEventBus.set_activity("cerebellum", "Daily Maintenance")
        
        if brain.hippocampus:
            deleted = await brain.hippocampus.cleanup_old_schedules(days_old=30)
            logger.info(f"  ✓ Cleaned up {deleted} old schedules")
            
            decayed = await brain.hippocampus.apply_memory_decay()
            logger.info(f"  ✓ Applied decay to {decayed} memories")
            
            await brain.hippocampus.optimize_knowledge_graph()
            logger.info("  ✓ Optimized knowledge graph")

        await NeuralEventBus.clear_activity("cerebellum")
        logger.info("✅ Daily maintenance completed")
        
    except Exception as e:
        logger.error(f"❌ Maintenance failed: {e}")
        await NeuralEventBus.clear_activity("cerebellum")

async def background_memory_optimization(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Optimize and consolidate memories - runs every 2 hours."""
    brain = get_brain_from_context(context)
    if not brain or not brain.hippocampus:
        return

    try:
        await NeuralEventBus.set_activity("cerebellum", "Memory Optimization")
        
        consolidated = await brain.hippocampus.consolidate_memories()
        if consolidated > 0:
            logger.info(f"  ✓ Consolidated {consolidated} similar memories")
        
        await brain.hippocampus.optimize_knowledge_graph()

        await NeuralEventBus.clear_activity("cerebellum")
        logger.info("✅ Memory optimization completed")
        
    except Exception as e:
        logger.error(f"❌ Memory optimization failed: {e}")
        await NeuralEventBus.clear_activity("cerebellum")

async def background_memory_compression(context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Check and compress memories into Global Context - runs every 30 minutes.
    """
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
                    "compression_version": stats_after.get("compression_version", 0)
                }
            )
            logger.info(f"✅ Memory compression completed - Global Context updated")
        else:
            await NeuralEventBus.emit(
                "cerebellum", "dashboard", "compression_skipped",
                payload={
                    "reason": "threshold_not_met",
                    "uncompressed_count": stats_before.get("uncompressed_count", 0),
                    "threshold": brain.hippocampus.COMPRESSION_THRESHOLD
                }
            )

        await NeuralEventBus.clear_activity("cerebellum")
        
    except Exception as e:
        logger.error(f"❌ Memory compression failed: {e}")
        await NeuralEventBus.emit(
            "cerebellum", "dashboard", "compression_failed",
            payload={"error": str(e)}
        )
        await NeuralEventBus.clear_activity("cerebellum")

async def background_session_cleanup(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cleanup stale sessions - runs every 30 minutes."""
    brain = get_brain_from_context(context)
    if not brain or not brain.thalamus:
        return

    try:
        await NeuralEventBus.set_activity("cerebellum", "Session Cleanup")
        await brain.thalamus.cleanup_session()
        await NeuralEventBus.clear_activity("cerebellum")
        logger.info("✅ Session cleanup completed")
        
    except Exception as e:
        logger.error(f"❌ Session cleanup failed: {e}")
        await NeuralEventBus.clear_activity("cerebellum")

async def background_schedule_checker(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check and execute due schedules - runs every minute."""
    brain = get_brain_from_context(context)
    if not brain or not brain.hippocampus:
        return

    try:
        pending = await brain.hippocampus.get_pending_schedules(limit=10)

        for schedule in pending:
            schedule_id = schedule.get("id")
            schedule_context = schedule.get("context", "")

            from src.brainstem import ADMIN_ID
            if ADMIN_ID:
                try:
                    await NeuralEventBus.set_activity("cerebellum", f"Sending Reminder")
                    
                    if brain.prefrontal_cortex:
                        response = await brain.prefrontal_cortex.process(
                            message=f"[SCHEDULED REMINDER] {schedule_context}",
                            user_name="System"
                        )
                    else:
                        response = f"⏰ Pengingat: {schedule_context}"

                    await context.bot.send_message(
                        chat_id=int(ADMIN_ID),
                        text=response
                    )
                    await brain.hippocampus.mark_schedule_executed(
                        schedule_id, "delivered"
                    )
                    
                    await NeuralEventBus.emit(
                        "cerebellum", "motor_cortex", "schedule_delivered",
                        payload={"schedule_id": schedule_id}
                    )
                    
                    logger.info(f"📤 Schedule {schedule_id} delivered")

                except Exception as e:
                    logger.error(f"Failed to send schedule {schedule_id}: {e}")
                    await brain.hippocampus.mark_schedule_executed(
                        schedule_id, f"failed: {str(e)[:50]}"
                    )
                finally:
                    await NeuralEventBus.clear_activity("cerebellum")

    except Exception as e:
        logger.error(f"❌ Schedule checker failed: {e}")

async def background_emotional_decay(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gradually return emotional state to neutral - runs hourly."""
    brain = get_brain_from_context(context)
    if not brain or not brain.amygdala:
        return

    try:
        await NeuralEventBus.set_activity("cerebellum", "Emotional Decay")
        
        current = brain.amygdala.satisfaction
        if abs(current) > 0.1:
            decay_amount = 0.05 if current > 0 else -0.05
            brain.amygdala._state.satisfaction_level = max(-1, min(1, current - decay_amount))

        await brain.amygdala.save_state()
        
        await NeuralEventBus.emit(
            "cerebellum", "amygdala", "emotional_decay",
            payload={"new_satisfaction": brain.amygdala.satisfaction}
        )
        
        await NeuralEventBus.clear_activity("cerebellum")

    except Exception as e:
        logger.error(f"❌ Emotional decay failed: {e}")
        await NeuralEventBus.clear_activity("cerebellum")

async def background_proactive_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check for proactive engagement opportunities - runs every 30 minutes."""
    brain = get_brain_from_context(context)
    if not brain or not brain.thalamus:
        return

    try:
        await NeuralEventBus.set_activity("cerebellum", "Proactive Check")
        
        message = await brain.thalamus.check_proactive_triggers()
        
        if message:
            from src.brainstem import ADMIN_ID
            from telegram.constants import ParseMode
            
            if ADMIN_ID:
                try:
                    await context.bot.send_message(
                        chat_id=int(ADMIN_ID),
                        text=message,
                        parse_mode=ParseMode.MARKDOWN
                    )
                    
                    await NeuralEventBus.emit(
                        "cerebellum", "motor_cortex", "proactive_message_sent",
                        payload={"message_length": len(message)}
                    )
                    
                    logger.info("📤 Proactive message sent")
                except Exception as e:
                    logger.error(f"Failed to send proactive message: {e}")
        
        await NeuralEventBus.clear_activity("cerebellum")

    except Exception as e:
        logger.error(f"❌ Proactive check failed: {e}")
        await NeuralEventBus.clear_activity("cerebellum")

async def background_topic_analysis(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze conversation topics - runs every hour."""
    brain = get_brain_from_context(context)
    if not brain or not brain.hippocampus:
        return

    try:
        await NeuralEventBus.set_activity("cerebellum", "Topic Analysis")
        
        await NeuralEventBus.clear_activity("cerebellum")

    except Exception as e:
        logger.error(f"❌ Topic analysis failed: {e}")
        await NeuralEventBus.clear_activity("cerebellum")

async def background_health_check(context: ContextTypes.DEFAULT_TYPE) -> None:
    """System health monitoring - runs every 5 minutes."""
    brain = get_brain_from_context(context)
    if not brain:
        return

    try:
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "hippocampus": brain.hippocampus is not None,
            "prefrontal_cortex": brain.prefrontal_cortex is not None,
            "amygdala": brain.amygdala is not None,
            "thalamus": brain.thalamus is not None,
            "openrouter": brain.openrouter is not None
        }
        
        if brain.openrouter:
            api_status = brain.openrouter.get_status()
            health_status["api_configured"] = api_status.get("api_configured", False)
            health_status["failed_models"] = len(api_status.get("failed_models", []))
        
        await NeuralEventBus.emit(
            "cerebellum", "dashboard", "health_check",
            payload=health_status
        )

    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")

def register_background_jobs(app) -> None:
    """Register all background jobs with the application."""
    if not app.job_queue:
        logger.warning("Job queue not available, skipping background jobs")
        return

    app.job_queue.run_repeating(
        background_schedule_checker,
        interval=60,
        first=10,
        name="schedule_checker"
    )

    app.job_queue.run_repeating(
        background_session_cleanup,
        interval=1800,
        first=300,
        name="session_cleanup"
    )

    app.job_queue.run_repeating(
        background_memory_compression,
        interval=1800,
        first=600,
        name="memory_compression"
    )

    app.job_queue.run_repeating(
        background_memory_optimization,
        interval=7200,
        first=3600,
        name="memory_optimization"
    )

    app.job_queue.run_repeating(
        background_maintenance,
        interval=86400,
        first=60,
        name="daily_maintenance"
    )

    app.job_queue.run_repeating(
        background_emotional_decay,
        interval=3600,
        first=1800,
        name="emotional_decay"
    )

    app.job_queue.run_repeating(
        background_proactive_check,
        interval=1800,
        first=900,
        name="proactive_check"
    )

    app.job_queue.run_repeating(
        background_health_check,
        interval=300,
        first=30,
        name="health_check"
    )

    app.job_queue.run_repeating(
        background_topic_analysis,
        interval=3600,
        first=2400,
        name="topic_analysis"
    )

    logger.info("✅ Cerebellum: All background jobs registered")
    logger.info("   - Schedule checker: every 1 minute")
    logger.info("   - Session cleanup: every 30 minutes")
    logger.info("   - Memory compression: every 30 minutes")
    logger.info("   - Memory optimization: every 2 hours")
    logger.info("   - Daily maintenance: every 24 hours")
    logger.info("   - Emotional decay: every 1 hour")
    logger.info("   - Proactive check: every 30 minutes")
    logger.info("   - Health check: every 5 minutes")
    logger.info("   - Topic analysis: every 1 hour")

async def trigger_memory_compression_manual(brain) -> Dict[str, Any]:
    """Manually trigger memory compression."""
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
            "stats_after": stats_after
        }
        
    except Exception as e:
        await NeuralEventBus.clear_activity("cerebellum")
        return {"status": "error", "message": str(e)}

async def trigger_maintenance_manual(brain) -> Dict[str, Any]:
    """Manually trigger maintenance tasks."""
    if not brain or not brain.hippocampus:
        return {"status": "error", "message": "Brain not initialized"}
    
    try:
        await NeuralEventBus.set_activity("cerebellum", "Manual Maintenance")
        
        results = {
            "schedules_cleaned": 0,
            "memories_decayed": 0,
            "knowledge_graph_optimized": False
        }
        
        results["schedules_cleaned"] = await brain.hippocampus.cleanup_old_schedules(days_old=30)
        results["memories_decayed"] = await brain.hippocampus.apply_memory_decay()
        await brain.hippocampus.optimize_knowledge_graph()
        results["knowledge_graph_optimized"] = True
        
        await NeuralEventBus.clear_activity("cerebellum")
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        await NeuralEventBus.clear_activity("cerebellum")
        return {"status": "error", "message": str(e)}
