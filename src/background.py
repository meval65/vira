"""
Background tasks for Vira Telegram Bot.
Contains scheduled maintenance and cleanup jobs.
"""

import asyncio
import logging
from datetime import datetime, timedelta

from telegram.ext import ContextTypes

from src.utils import USER_LOCKS, USER_LAST_ACTIVITY, RATE_LIMIT_TOKENS

logger = logging.getLogger(__name__)


async def background_maintenance(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Periodic maintenance tasks."""
    services = context.application.bot_data
    try:
        await services['scheduler'].cleanup_old_schedules(days_old=30)
        await services['mem_mgr'].optimize_memories(user_id=None)
        logger.info("Maintenance completed successfully")
    except Exception as e:
        logger.error(f"Maintenance failed: {e}")


async def background_memory_optimization(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Optimize memories for all active users."""
    services = context.application.bot_data
    try:
        rows = await services['db'].fetchall(
            "SELECT DISTINCT user_id FROM memories WHERE status='active'", ()
        )

        for row in rows:
            user_id = row[0]
            await asyncio.to_thread(
                services['mem_mgr'].deduplicate_existing_memories,
                user_id
            )
            await asyncio.to_thread(
                services['mem_mgr'].apply_decay_rules,
                user_id
            )

        logger.info("Memory optimization completed")
    except Exception as e:
        logger.error(f"Optimization failed: {e}")


async def background_cleanup(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Cleanup inactive user sessions."""
    now = datetime.now()
    timeout = timedelta(minutes=60)

    inactive_users = [
        uid for uid, last_time in USER_LAST_ACTIVITY.items()
        if now - last_time > timeout and uid in USER_LOCKS and not USER_LOCKS[uid].locked()
    ]

    for uid in inactive_users:
        USER_LOCKS.pop(uid, None)
        USER_LAST_ACTIVITY.pop(uid, None)
        RATE_LIMIT_TOKENS.pop(uid, None)

    if inactive_users:
        logger.info(f"Cleaned up {len(inactive_users)} inactive user sessions")


async def background_schedule_checker(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Check and execute due schedules."""
    services = context.application.bot_data
    try:
        due_items = await services['scheduler'].get_due_schedules_batch(batch_size=50)

        for item in due_items:
            user_id = str(item['user_id'])

            try:
                ai_resp = await services['chat_handler'].trigger_proactive_message(
                    user_id,
                    item['context']
                )

                if ai_resp:
                    try:
                        chat_id_int = int(user_id)
                        await context.bot.send_message(
                            chat_id=chat_id_int,
                            text=ai_resp
                        )
                        await services['scheduler'].mark_as_executed(
                            item['id'],
                            "Sent successfully"
                        )
                    except ValueError:
                        logger.warning(f"Skipping message for non-integer user_id: {user_id}")
                        await services['scheduler'].mark_as_executed(
                            item['id'],
                            "Skipped: Invalid Chat ID"
                        )
                else:
                    await services['scheduler'].mark_as_executed(
                        item['id'],
                        "Empty response"
                    )
            except Exception as e:
                logger.error(f"Schedule execution failed for {user_id}: {e}")
                await services['scheduler'].mark_as_executed(
                    item['id'],
                    f"Failed: {str(e)[:100]}"
                )
    except Exception as e:
        logger.error(f"Schedule checker failed: {e}")
