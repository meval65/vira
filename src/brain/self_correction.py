
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    ARGUMENT_ERROR = "argument_error"
    FORMAT_ERROR = "format_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT_ERROR = "timeout_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN_ERROR = "unknown_error"


class CorrectionStrategy(str, Enum):
    MODIFY_ARGS = "modify_args"
    USE_ALTERNATIVE = "use_alternative"
    SKIP = "skip"
    ABORT = "abort"
    RETRY_AS_IS = "retry_as_is"


@dataclass
class ToolExecutionResult:
    tool_name: str
    args: Dict[str, Any]
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    attempt_number: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "args": self.args,
            "success": self.success,
            "output": self.output[:500] if self.output else None,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "attempt_number": self.attempt_number,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class CritiqueResult:
    error_type: ErrorType
    explanation: str
    strategy: CorrectionStrategy
    modified_args: Optional[Dict[str, Any]] = None
    alternative_tool: Optional[str] = None
    confidence: float = 0.5
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.error_type.value,
            "explanation": self.explanation,
            "strategy": self.strategy.value,
            "modified_args": self.modified_args,
            "alternative_tool": self.alternative_tool,
            "confidence": self.confidence,
            "reasoning": self.reasoning
        }


@dataclass
class CorrectionAttempt:
    original_result: ToolExecutionResult
    critique: CritiqueResult
    corrected_result: Optional[ToolExecutionResult] = None
    successful_correction: bool = False


CRITIQUE_PROMPT_TEMPLATE = """# TOOL ERROR ANALYSIS

You are analyzing a failed tool execution to determine how to recover.

## Tool Execution Details
- **Tool Name**: {tool_name}
- **Arguments Used**: {args}
- **Error Output**: {error}

## Available Tools
{available_tools}

## Your Task
Analyze the error and determine the best recovery strategy.

Return a JSON object with:
```json
{{
  "error_type": "argument_error|format_error|resource_not_found|permission_error|timeout_error|api_error|validation_error|unknown_error",
  "explanation": "Brief explanation of what went wrong",
  "strategy": "modify_args|use_alternative|skip|abort|retry_as_is",
  "modified_args": {{"arg_name": "corrected_value"}},  // Only if strategy is modify_args
  "alternative_tool": "tool_name",  // Only if strategy is use_alternative
  "confidence": 0.8,  // 0.0 to 1.0
  "reasoning": "Why this strategy will work"
}}
```

### Guidelines:
- For "argument_error": Suggest corrected argument values
- For "resource_not_found": Check if entity name is misspelled
- For "format_error": Fix date/time/number formats
- For "timeout_error": Suggest retry_as_is
- For "api_error": Suggest skip or abort
- If error is unrecoverable, use "abort" strategy
"""


class SelfCorrectionLoop:
    
    MAX_CORRECTION_ATTEMPTS: int = 3
    CONFIDENCE_THRESHOLD: float = 0.6
    
    def __init__(self, openrouter_client, parietal_lobe, mongo_client=None):
        self._openrouter = openrouter_client
        self._parietal_lobe = parietal_lobe
        self._mongo = mongo_client
        self._correction_history: List[CorrectionAttempt] = []
        
    async def execute_with_correction(
        self,
        tool_name: str,
        args: Dict[str, Any],
        context: Optional[str] = None
    ) -> Tuple[Optional[str], bool, List[CorrectionAttempt]]:
        attempts: List[CorrectionAttempt] = []
        current_tool = tool_name
        current_args = args.copy()
        
        for attempt_num in range(1, self.MAX_CORRECTION_ATTEMPTS + 1):
            result = await self._execute_tool(current_tool, current_args, attempt_num)
            
            if result.success:
                if attempt_num > 1:
                    logger.info(
                        f"Self-correction successful on attempt {attempt_num} "
                        f"for {tool_name}"
                    )
                    await self._log_correction(attempts, success=True)
                return result.output, True, attempts
            
            if attempt_num >= self.MAX_CORRECTION_ATTEMPTS:
                logger.warning(
                    f"Max correction attempts ({self.MAX_CORRECTION_ATTEMPTS}) "
                    f"reached for {tool_name}"
                )
                await self._log_correction(attempts, success=False)
                return result.error, False, attempts
            
            critique = await self._critique_error(
                result, context
            )
            
            attempt = CorrectionAttempt(
                original_result=result,
                critique=critique
            )
            attempts.append(attempt)
            
            if critique.strategy == CorrectionStrategy.ABORT:
                logger.info(f"Aborting tool execution for {tool_name}: {critique.explanation}")
                await self._log_correction(attempts, success=False)
                return result.error, False, attempts
            
            if critique.strategy == CorrectionStrategy.SKIP:
                logger.info(f"Skipping tool {tool_name}: {critique.explanation}")
                return None, False, attempts
            
            if critique.confidence < self.CONFIDENCE_THRESHOLD:
                logger.warning(
                    f"Low confidence ({critique.confidence}) in correction, aborting"
                )
                await self._log_correction(attempts, success=False)
                return result.error, False, attempts
            
            if critique.strategy == CorrectionStrategy.MODIFY_ARGS:
                if critique.modified_args:
                    current_args = {**current_args, **critique.modified_args}
                    logger.info(f"Modified args for {tool_name}: {critique.modified_args}")
            
            elif critique.strategy == CorrectionStrategy.USE_ALTERNATIVE:
                if critique.alternative_tool:
                    current_tool = critique.alternative_tool
                    logger.info(f"Switching to alternative tool: {current_tool}")
            
            
            await asyncio.sleep(0.5)
        
        return None, False, attempts
    
    async def _execute_tool(
        self,
        tool_name: str,
        args: Dict[str, Any],
        attempt_number: int
    ) -> ToolExecutionResult:
        start_time = datetime.now()
        
        try:
            output = await self._parietal_lobe.execute(tool_name, args)
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            
            if output and self._is_error_output(output):
                return ToolExecutionResult(
                    tool_name=tool_name,
                    args=args,
                    success=False,
                    error=output,
                    duration_ms=duration_ms,
                    attempt_number=attempt_number
                )
            
            return ToolExecutionResult(
                tool_name=tool_name,
                args=args,
                success=True,
                output=output,
                duration_ms=duration_ms,
                attempt_number=attempt_number
            )
            
        except Exception as e:
            duration_ms = (datetime.now() - start_time).total_seconds() * 1000
            return ToolExecutionResult(
                tool_name=tool_name,
                args=args,
                success=False,
                error=str(e),
                duration_ms=duration_ms,
                attempt_number=attempt_number
            )
    
    def _is_error_output(self, output: str) -> bool:
        if not output:
            return False
        
        output_lower = output.lower()[:200]
        error_indicators = [
            "error:", "failed:", "exception:", "tidak ditemukan",
            "not found", "invalid", "gagal", "couldn't", "can't",
            "unable to", "no such", "does not exist"
        ]
        return any(indicator in output_lower for indicator in error_indicators)
    
    async def _critique_error(
        self,
        result: ToolExecutionResult,
        context: Optional[str] = None
    ) -> CritiqueResult:
        try:
            available_tools = ""
            if self._parietal_lobe:
                available_tools = self._parietal_lobe.get_tool_descriptions() or ""
            
            prompt = CRITIQUE_PROMPT_TEMPLATE.format(
                tool_name=result.tool_name,
                args=result.args,
                error=result.error or "Unknown error",
                available_tools=available_tools
            )
            
            if context:
                prompt += f"\n\n## Additional Context\n{context}"
            
            response = await self._openrouter.quick_completion(
                prompt=prompt,
                max_tokens=512,
                temperature=0.1,
                tier="analysis_model",
                json_mode=True
            )
            
            return self._parse_critique_response(response)
            
        except Exception as e:
            logger.error(f"Critique generation failed: {e}")
            return CritiqueResult(
                error_type=ErrorType.UNKNOWN_ERROR,
                explanation=str(e),
                strategy=CorrectionStrategy.ABORT,
                confidence=0.0,
                reasoning="Failed to generate critique"
            )
    
    def _parse_critique_response(self, response: str) -> CritiqueResult:
        import json
        
        try:
            json_match = None
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_match = response[start:end].strip()
            elif "{" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_match = response[start:end]
            
            if json_match:
                data = json.loads(json_match)
                
                error_type_str = data.get("error_type", "unknown_error")
                try:
                    error_type = ErrorType(error_type_str)
                except ValueError:
                    error_type = ErrorType.UNKNOWN_ERROR
                
                strategy_str = data.get("strategy", "abort")
                try:
                    strategy = CorrectionStrategy(strategy_str)
                except ValueError:
                    strategy = CorrectionStrategy.ABORT
                
                return CritiqueResult(
                    error_type=error_type,
                    explanation=data.get("explanation", ""),
                    strategy=strategy,
                    modified_args=data.get("modified_args"),
                    alternative_tool=data.get("alternative_tool"),
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", "")
                )
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Failed to parse critique: {e}")
        
        return CritiqueResult(
            error_type=ErrorType.UNKNOWN_ERROR,
            explanation="Failed to parse critique response",
            strategy=CorrectionStrategy.ABORT,
            confidence=0.0,
            reasoning="Parse error"
        )
    
    async def _log_correction(
        self,
        attempts: List[CorrectionAttempt],
        success: bool
    ) -> None:
        if not self._mongo or not attempts:
            return
        
        try:
            log_entry = {
                "timestamp": datetime.now(),
                "success": success,
                "total_attempts": len(attempts),
                "attempts": [
                    {
                        "original": a.original_result.to_dict(),
                        "critique": a.critique.to_dict(),
                        "corrected": a.corrected_result.to_dict() if a.corrected_result else None
                    }
                    for a in attempts
                ]
            }
            
            await self._mongo.db["correction_log"].insert_one(log_entry)
            
        except Exception as e:
            logger.error(f"Failed to log correction: {e}")
    
    def get_correction_stats(self) -> Dict[str, Any]:
        if not self._correction_history:
            return {"total": 0}
        
        successful = sum(1 for a in self._correction_history if a.successful_correction)
        
        error_types: Dict[str, int] = {}
        strategies: Dict[str, int] = {}
        
        for attempt in self._correction_history:
            error_type = attempt.critique.error_type.value
            strategy = attempt.critique.strategy.value
            
            error_types[error_type] = error_types.get(error_type, 0) + 1
            strategies[strategy] = strategies.get(strategy, 0) + 1
        
        return {
            "total": len(self._correction_history),
            "successful": successful,
            "success_rate": successful / len(self._correction_history),
            "error_types": error_types,
            "strategies": strategies
        }
