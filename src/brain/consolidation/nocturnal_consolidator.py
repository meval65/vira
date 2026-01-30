"""
Nocturnal Memory Consolidation for Vira Personal Life OS.

This module processes daily episodic logs (chat history) into structured
semantic memory during off-peak hours (default: 3 AM).

The consolidation process:
1. Fetches today's chat logs from MongoDB
2. Extracts patterns, preferences, and facts using LLM
3. Stores them as high-priority semantic memories
4. Updates the Knowledge Graph with new entity relationships
5. Logs the consolidation for auditing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# LLM prompts for extraction
PREFERENCE_EXTRACTION_PROMPT = """# PREFERENCE EXTRACTION

Analyze these conversation logs and extract user preferences.

## Conversation Logs
{logs}

## Task
Identify explicit and implicit preferences about:
- Food, drinks, cuisine
- Entertainment (movies, music, games)
- Schedule preferences (morning/evening person, busy times)
- Communication style preferences
- Work/productivity habits
- Personal interests and hobbies

Return a JSON array of preferences:
```json
[
  {{
    "preference": "Description of the preference",
    "category": "food|entertainment|schedule|communication|productivity|interests|other",
    "confidence": 0.8,
    "evidence": "Quote or reference from conversation"
  }}
]
```

Only include preferences with confidence >= 0.6. Return empty array if none found.
"""

FACT_EXTRACTION_PROMPT = """# FACT EXTRACTION

Analyze these conversation logs and extract factual information about the user.

## Conversation Logs
{logs}

## Task
Identify factual information about:
- Personal details (job, location, family)
- Important dates (birthdays, anniversaries)
- Relationships (names of friends, family, colleagues)
- Skills and expertise
- Goals and aspirations
- Regular commitments

Return a JSON array of facts:
```json
[
  {{
    "fact": "The factual statement",
    "category": "personal|dates|relationships|skills|goals|commitments|other",
    "entity": "Person or thing this fact is about",
    "confidence": 0.9,
    "evidence": "Quote or reference from conversation"
  }}
]
```

Only include facts with confidence >= 0.7. Return empty array if none found.
"""

PATTERN_EXTRACTION_PROMPT = """# PATTERN EXTRACTION

Analyze these conversation logs and identify recurring patterns.

## Conversation Logs
{logs}

## Task
Identify patterns in:
- Conversation timing (when user is most active)
- Mood patterns (emotional trends)
- Topic recurrence (subjects user returns to)
- Request patterns (what user commonly asks for)
- Behavioral patterns

Return a JSON array of patterns:
```json
[
  {{
    "pattern": "Description of the pattern",
    "category": "timing|mood|topics|requests|behavior|other",
    "frequency": "daily|weekly|occasional",
    "confidence": 0.7,
    "examples": ["example1", "example2"]
  }}
]
```

Only include patterns observed at least twice. Return empty array if none found.
"""


@dataclass
class ExtractedItem:
    """Base class for extracted information."""
    content: str
    category: str
    confidence: float
    evidence: str
    extracted_at: datetime = field(default_factory=datetime.now)


@dataclass
class ExtractedPreference(ExtractedItem):
    """An extracted user preference."""
    pass


@dataclass
class ExtractedFact(ExtractedItem):
    """An extracted fact about the user."""
    entity: Optional[str] = None


@dataclass
class ExtractedPattern(ExtractedItem):
    """An extracted behavioral pattern."""
    frequency: str = "occasional"
    examples: List[str] = field(default_factory=list)


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""
    timestamp: datetime
    logs_processed: int
    preferences_extracted: int
    facts_extracted: int
    patterns_extracted: int
    memories_created: int
    kg_triples_created: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "logs_processed": self.logs_processed,
            "preferences_extracted": self.preferences_extracted,
            "facts_extracted": self.facts_extracted,
            "patterns_extracted": self.patterns_extracted,
            "memories_created": self.memories_created,
            "kg_triples_created": self.kg_triples_created,
            "duration_seconds": self.duration_seconds,
            "success": self.success,
            "error": self.error
        }


class NocturnalConsolidator:
    """
    Processes daily episodic logs into structured semantic memory.
    
    Designed to run during off-peak hours via cron/scheduler.
    """
    
    CONSOLIDATION_HOUR: int = 3  # 3 AM
    MIN_LOGS_TO_PROCESS: int = 5
    MAX_LOGS_PER_BATCH: int = 100
    PREFERENCE_CONFIDENCE_THRESHOLD: float = 0.6
    FACT_CONFIDENCE_THRESHOLD: float = 0.7
    
    def __init__(self, hippocampus, openrouter_client, mongo_client):
        """
        Initialize the consolidator.
        
        Args:
            hippocampus: Hippocampus instance for memory operations
            openrouter_client: OpenRouterClient for LLM calls
            mongo_client: MongoDB client for direct access
        """
        self._hippocampus = hippocampus
        self._openrouter = openrouter_client
        self._mongo = mongo_client
        self._last_consolidation: Optional[datetime] = None
        self._consolidation_history: List[ConsolidationResult] = []
    
    async def run_consolidation(
        self,
        force: bool = False,
        hours_back: int = 24
    ) -> ConsolidationResult:
        """
        Run the memory consolidation process.
        
        Args:
            force: Run even if outside normal hours or recently run
            hours_back: How many hours of logs to process
            
        Returns:
            ConsolidationResult with details of the run
        """
        start_time = datetime.now()
        
        # Check if we should run
        if not force and not self._should_run():
            return ConsolidationResult(
                timestamp=start_time,
                logs_processed=0,
                preferences_extracted=0,
                facts_extracted=0,
                patterns_extracted=0,
                memories_created=0,
                kg_triples_created=0,
                duration_seconds=0,
                success=False,
                error="Consolidation skipped (not scheduled time or recently run)"
            )
        
        try:
            logger.info("Starting nocturnal memory consolidation...")
            
            # 1. Fetch chat logs
            logs = await self._fetch_logs(hours_back)
            
            if len(logs) < self.MIN_LOGS_TO_PROCESS:
                logger.info(f"Only {len(logs)} logs found, skipping consolidation")
                return ConsolidationResult(
                    timestamp=start_time,
                    logs_processed=len(logs),
                    preferences_extracted=0,
                    facts_extracted=0,
                    patterns_extracted=0,
                    memories_created=0,
                    kg_triples_created=0,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                    success=True,
                    error=None
                )
            
            # Format logs for LLM
            formatted_logs = self._format_logs(logs)
            
            # 2. Extract preferences, facts, and patterns (parallel)
            preferences_task = self._extract_preferences(formatted_logs)
            facts_task = self._extract_facts(formatted_logs)
            patterns_task = self._extract_patterns(formatted_logs)
            
            preferences, facts, patterns = await asyncio.gather(
                preferences_task, facts_task, patterns_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(preferences, Exception):
                logger.error(f"Preference extraction failed: {preferences}")
                preferences = []
            if isinstance(facts, Exception):
                logger.error(f"Fact extraction failed: {facts}")
                facts = []
            if isinstance(patterns, Exception):
                logger.error(f"Pattern extraction failed: {patterns}")
                patterns = []
            
            # 3. Store as memories
            memories_created = 0
            kg_triples = 0
            
            # Store preferences
            for pref in preferences:
                if await self._store_preference(pref):
                    memories_created += 1
            
            # Store facts and update KG
            for fact in facts:
                if await self._store_fact(fact):
                    memories_created += 1
                    if fact.entity:
                        if await self._update_kg_from_fact(fact):
                            kg_triples += 1
            
            # Store patterns
            for pattern in patterns:
                if await self._store_pattern(pattern):
                    memories_created += 1
            
            # 4. Log the consolidation
            duration = (datetime.now() - start_time).total_seconds()
            result = ConsolidationResult(
                timestamp=start_time,
                logs_processed=len(logs),
                preferences_extracted=len(preferences) if isinstance(preferences, list) else 0,
                facts_extracted=len(facts) if isinstance(facts, list) else 0,
                patterns_extracted=len(patterns) if isinstance(patterns, list) else 0,
                memories_created=memories_created,
                kg_triples_created=kg_triples,
                duration_seconds=duration,
                success=True,
                error=None
            )
            
            await self._log_consolidation(result)
            self._last_consolidation = datetime.now()
            self._consolidation_history.append(result)
            
            logger.info(
                f"Consolidation complete: {memories_created} memories created, "
                f"{kg_triples} KG triples in {duration:.1f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            duration = (datetime.now() - start_time).total_seconds()
            return ConsolidationResult(
                timestamp=start_time,
                logs_processed=0,
                preferences_extracted=0,
                facts_extracted=0,
                patterns_extracted=0,
                memories_created=0,
                kg_triples_created=0,
                duration_seconds=duration,
                success=False,
                error=str(e)
            )
    
    def _should_run(self) -> bool:
        """Check if consolidation should run."""
        now = datetime.now()
        
        # Check if it's the right hour
        if now.hour != self.CONSOLIDATION_HOUR:
            return False
        
        # Check if we already ran today
        if self._last_consolidation:
            hours_since = (now - self._last_consolidation).total_seconds() / 3600
            if hours_since < 20:  # Don't run more than once per 20 hours
                return False
        
        return True
    
    async def _fetch_logs(self, hours_back: int) -> List[Dict]:
        """Fetch chat logs from the last N hours."""
        cutoff = datetime.now() - timedelta(hours=hours_back)
        
        cursor = self._mongo.chat_logs.find({
            "timestamp": {"$gte": cutoff}
        }).sort("timestamp", 1).limit(self.MAX_LOGS_PER_BATCH)
        
        logs = await cursor.to_list(length=self.MAX_LOGS_PER_BATCH)
        return logs
    
    def _format_logs(self, logs: List[Dict]) -> str:
        """Format logs for LLM processing."""
        formatted = []
        for log in logs:
            role = log.get("role", "unknown")
            content = log.get("content", "")[:500]  # Truncate long messages
            timestamp = log.get("timestamp", "")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.strftime("%H:%M")
            formatted.append(f"[{timestamp}] {role.upper()}: {content}")
        
        return "\n".join(formatted)
    
    async def _extract_preferences(self, logs: str) -> List[ExtractedPreference]:
        """Extract preferences from logs using LLM."""
        prompt = PREFERENCE_EXTRACTION_PROMPT.format(logs=logs)
        
        response = await self._openrouter.quick_completion(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
            tier="analysis_model",
            json_mode=True
        )
        
        return self._parse_preferences(response)
    
    async def _extract_facts(self, logs: str) -> List[ExtractedFact]:
        """Extract facts from logs using LLM."""
        prompt = FACT_EXTRACTION_PROMPT.format(logs=logs)
        
        response = await self._openrouter.quick_completion(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
            tier="analysis_model",
            json_mode=True
        )
        
        return self._parse_facts(response)
    
    async def _extract_patterns(self, logs: str) -> List[ExtractedPattern]:
        """Extract patterns from logs using LLM."""
        prompt = PATTERN_EXTRACTION_PROMPT.format(logs=logs)
        
        response = await self._openrouter.quick_completion(
            prompt=prompt,
            max_tokens=1024,
            temperature=0.1,
            tier="analysis_model",
            json_mode=True
        )
        
        return self._parse_patterns(response)
    
    def _parse_preferences(self, response: str) -> List[ExtractedPreference]:
        """Parse LLM response into preferences."""
        import json
        
        try:
            data = self._extract_json_array(response)
            preferences = []
            
            for item in data:
                if item.get("confidence", 0) >= self.PREFERENCE_CONFIDENCE_THRESHOLD:
                    preferences.append(ExtractedPreference(
                        content=item.get("preference", ""),
                        category=item.get("category", "other"),
                        confidence=item.get("confidence", 0),
                        evidence=item.get("evidence", "")
                    ))
            
            return preferences
            
        except Exception as e:
            logger.error(f"Failed to parse preferences: {e}")
            return []
    
    def _parse_facts(self, response: str) -> List[ExtractedFact]:
        """Parse LLM response into facts."""
        import json
        
        try:
            data = self._extract_json_array(response)
            facts = []
            
            for item in data:
                if item.get("confidence", 0) >= self.FACT_CONFIDENCE_THRESHOLD:
                    facts.append(ExtractedFact(
                        content=item.get("fact", ""),
                        category=item.get("category", "other"),
                        confidence=item.get("confidence", 0),
                        evidence=item.get("evidence", ""),
                        entity=item.get("entity")
                    ))
            
            return facts
            
        except Exception as e:
            logger.error(f"Failed to parse facts: {e}")
            return []
    
    def _parse_patterns(self, response: str) -> List[ExtractedPattern]:
        """Parse LLM response into patterns."""
        import json
        
        try:
            data = self._extract_json_array(response)
            patterns = []
            
            for item in data:
                patterns.append(ExtractedPattern(
                    content=item.get("pattern", ""),
                    category=item.get("category", "other"),
                    confidence=item.get("confidence", 0),
                    evidence="",
                    frequency=item.get("frequency", "occasional"),
                    examples=item.get("examples", [])
                ))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to parse patterns: {e}")
            return []
    
    def _extract_json_array(self, response: str) -> List[Dict]:
        """Extract JSON array from LLM response."""
        import json
        
        # Try to find JSON array in response
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "[" in response:
            start = response.find("[")
            end = response.rfind("]") + 1
            response = response[start:end]
        
        return json.loads(response)
    
    async def _store_preference(self, pref: ExtractedPreference) -> bool:
        """Store preference as a memory."""
        try:
            memory_content = f"User preference ({pref.category}): {pref.content}"
            
            await self._hippocampus.store(
                content=memory_content,
                memory_type="preference",
                priority=7,  # High priority for preferences
                metadata={
                    "source": "consolidation",
                    "category": pref.category,
                    "confidence": pref.confidence,
                    "evidence": pref.evidence,
                    "is_consolidated": True
                }
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to store preference: {e}")
            return False
    
    async def _store_fact(self, fact: ExtractedFact) -> bool:
        """Store fact as a memory."""
        try:
            memory_content = f"Fact ({fact.category}): {fact.content}"
            
            await self._hippocampus.store(
                content=memory_content,
                memory_type="fact",
                priority=8,  # High priority for facts
                entity=fact.entity,
                metadata={
                    "source": "consolidation",
                    "category": fact.category,
                    "confidence": fact.confidence,
                    "evidence": fact.evidence,
                    "is_consolidated": True
                }
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to store fact: {e}")
            return False
    
    async def _store_pattern(self, pattern: ExtractedPattern) -> bool:
        """Store pattern as a memory."""
        try:
            memory_content = (
                f"Behavioral pattern ({pattern.category}): {pattern.content} "
                f"[occurs {pattern.frequency}]"
            )
            
            await self._hippocampus.store(
                content=memory_content,
                memory_type="context",
                priority=6,
                metadata={
                    "source": "consolidation",
                    "category": pattern.category,
                    "frequency": pattern.frequency,
                    "examples": pattern.examples,
                    "confidence": pattern.confidence,
                    "is_consolidated": True
                }
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to store pattern: {e}")
            return False
    
    async def _update_kg_from_fact(self, fact: ExtractedFact) -> bool:
        """Update Knowledge Graph with extracted fact."""
        try:
            if not fact.entity:
                return False
            
            # Create a triple: (Admin, predicate, entity)
            # Determine predicate from category
            predicate_map = {
                "relationships": "knows",
                "skills": "has_skill",
                "goals": "wants_to",
                "commitments": "committed_to",
                "personal": "is",
                "dates": "has_date"
            }
            
            predicate = predicate_map.get(fact.category, "related_to")
            
            await self._hippocampus.add_kg_triple(
                subject="admin",
                predicate=predicate,
                obj=fact.entity.lower()
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to update KG: {e}")
            return False
    
    async def _log_consolidation(self, result: ConsolidationResult) -> None:
        """Log consolidation result to MongoDB."""
        try:
            await self._mongo.db["consolidation_log"].insert_one(result.to_dict())
        except Exception as e:
            logger.error(f"Failed to log consolidation: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics."""
        if not self._consolidation_history:
            return {"total_runs": 0}
        
        successful = sum(1 for r in self._consolidation_history if r.success)
        total_memories = sum(r.memories_created for r in self._consolidation_history)
        total_triples = sum(r.kg_triples_created for r in self._consolidation_history)
        
        return {
            "total_runs": len(self._consolidation_history),
            "successful_runs": successful,
            "total_memories_created": total_memories,
            "total_kg_triples": total_triples,
            "last_run": self._last_consolidation.isoformat() if self._last_consolidation else None
        }
