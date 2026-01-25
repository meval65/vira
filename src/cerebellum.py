"""
Cerebellum - Background Tasks & Automated Functions

The cerebellum handles automated, repetitive tasks that don't require
conscious attention - just like the real brain's cerebellum handles
motor coordination and timing.

Tasks:
- Schedule checking and execution
- Memory optimization and cleanup
- Session cleanup
- System maintenance
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from telegram.ext import ContextTypes

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
        if brain.hippocampus:
            # Cleanup old schedules
            await brain.hippocampus.cleanup_old_schedules(days_old=30)
            # Decay old memories
            await brain.hippocampus.apply_memory_decay()

        logger.info("✅ Daily maintenance completed")
    except Exception as e:
        logger.error(f"❌ Maintenance failed: {e}")


async def background_memory_optimization(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Optimize and consolidate memories - runs every 2 hours."""
    brain = get_brain_from_context(context)
    if not brain or not brain.hippocampus:
        return

    try:
        # Consolidate similar memories
        await brain.hippocampus.consolidate_memories()
        # Update knowledge graph access counts
        await brain.hippocampus.optimize_knowledge_graph()

        logger.info("✅ Memory optimization completed")
    except Exception as e:
        logger.error(f"❌ Memory optimization failed: {e}")


async def background_session_cleanup(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cleanup stale sessions - runs every 30 minutes."""
    brain = get_brain_from_context(context)
    if not brain or not brain.thalamus:
        return

    try:
        await brain.thalamus.cleanup_session()
        logger.info("✅ Session cleanup completed")
    except Exception as e:
        logger.error(f"❌ Session cleanup failed: {e}")


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
                    # Generate contextual response
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
                    logger.info(f"📤 Schedule {schedule_id} delivered")

                except Exception as e:
                    logger.error(f"Failed to send schedule {schedule_id}: {e}")
                    await brain.hippocampus.mark_schedule_executed(
                        schedule_id, f"failed: {str(e)[:50]}"
                    )

    except Exception as e:
        logger.error(f"❌ Schedule checker failed: {e}")


async def background_emotional_decay(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Gradually return emotional state to neutral - runs hourly."""
    brain = get_brain_from_context(context)
    if not brain or not brain.amygdala:
        return

    try:
        # Slowly decay satisfaction toward neutral
        current = brain.amygdala.satisfaction
        if abs(current) > 0.1:
            decay_amount = 0.05 if current > 0 else -0.05
            brain.amygdala._state.satisfaction_level = max(-1, min(1, current - decay_amount))

        await brain.amygdala.save_state()

    except Exception as e:
        logger.error(f"❌ Emotional decay failed: {e}")


def register_background_jobs(app) -> None:
    """Register all background jobs with the application."""
    if not app.job_queue:
        logger.warning("Job queue not available, skipping background jobs")
        return

    # Schedule checker - every minute
    app.job_queue.run_repeating(
        background_schedule_checker,
        interval=60,
        first=10,
        name="schedule_checker"
    )

    # Session cleanup - every 30 minutes
    app.job_queue.run_repeating(
        background_session_cleanup,
        interval=1800,
        first=300,
        name="session_cleanup"
    )

    # Memory optimization - every 2 hours
    app.job_queue.run_repeating(
        background_memory_optimization,
        interval=7200,
        first=3600,
        name="memory_optimization"
    )

    # Daily maintenance - every 24 hours
    app.job_queue.run_repeating(
        background_maintenance,
        interval=86400,
        first=60,
        name="daily_maintenance"
    )

    # Emotional decay - every hour
    app.job_queue.run_repeating(
        background_emotional_decay,
        interval=3600,
        first=1800,
        name="emotional_decay"
    )

    logger.info("✅ Cerebellum: All background jobs registered")
