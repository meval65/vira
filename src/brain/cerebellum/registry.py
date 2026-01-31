import logging

from src.brain.cerebellum.schedule_checker import background_schedule_checker
from src.brain.cerebellum.session import background_session_cleanup
from src.brain.cerebellum.maintenance import (
    background_memory_compression,
    background_memory_optimization,
    background_maintenance,
)
from src.brain.cerebellum.emotional import background_emotional_decay
from src.brain.cerebellum.proactive import background_proactive_check
from src.brain.cerebellum.health import background_health_check
from src.brain.cerebellum.topic import background_topic_analysis

logger = logging.getLogger(__name__)


def register_background_jobs(app) -> None:
    if not app.job_queue:
        return
    app.job_queue.run_repeating(
        background_schedule_checker,
        interval=60,
        first=10,
        name="schedule_checker",
    )
    app.job_queue.run_repeating(
        background_session_cleanup,
        interval=1800,
        first=300,
        name="session_cleanup",
    )
    app.job_queue.run_repeating(
        background_memory_compression,
        interval=1800,
        first=600,
        name="memory_compression",
    )
    app.job_queue.run_repeating(
        background_memory_optimization,
        interval=7200,
        first=3600,
        name="memory_optimization",
    )
    app.job_queue.run_repeating(
        background_maintenance,
        interval=86400,
        first=60,
        name="daily_maintenance",
    )
    app.job_queue.run_repeating(
        background_emotional_decay,
        interval=3600,
        first=1800,
        name="emotional_decay",
    )
    app.job_queue.run_repeating(
        background_proactive_check,
        interval=1800,
        first=900,
        name="proactive_check",
    )
    app.job_queue.run_repeating(
        background_health_check,
        interval=300,
        first=30,
        name="health_check",
    )
    app.job_queue.run_repeating(
        background_topic_analysis,
        interval=3600,
        first=2400,
        name="topic_analysis",
    )
