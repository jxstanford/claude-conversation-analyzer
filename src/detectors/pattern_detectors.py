"""Pattern detectors for analyzing conversation patterns."""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from ..conversation_scanner import Message

logger = logging.getLogger(__name__)


class InterventionType(Enum):
    """Types of user interventions."""
    STOP_ACTION = "stop_action"
    CORRECTION = "correction"
    CLARIFICATION = "clarification"
    SCOPE_REDIRECT = "scope_redirect"
    APPROACH_CHANGE = "approach_change"
    TOOL_REJECTION = "tool_rejection"
    TAKEOVER = "takeover"


class SuccessIndicator(Enum):
    """Types of success indicators."""
    TASK_COMPLETED = "task_completed"
    TESTS_PASSED = "tests_passed"
    USER_SATISFIED = "user_satisfied"
    PROBLEM_SOLVED = "problem_solved"
    BUILD_SUCCESS = "build_success"


class ErrorType(Enum):
    """Types of errors."""
    FILE_NOT_FOUND = "file_not_found"
    SYNTAX_ERROR = "syntax_error"
    TEST_FAILURE = "test_failure"
    BUILD_FAILURE = "build_failure"
    LOGIC_ERROR = "logic_error"
    MISUNDERSTANDING = "misunderstanding"
    TOOL_ERROR = "tool_error"


@dataclass
class Intervention:
    """Represents a user intervention."""
    type: InterventionType
    message_index: int
    user_message: str
    context_before: Optional[str] = None
    claude_action_interrupted: Optional[str] = None
    severity: str = "medium"  # low, medium, high
    extended_context: Optional[List[Dict[str, str]]] = None  # Additional context messages


@dataclass
class SuccessPattern:
    """Represents a successful interaction pattern."""
    indicator: SuccessIndicator
    message_index: int
    pattern_description: str
    tool_sequence: List[str] = field(default_factory=list)


@dataclass
class ErrorPattern:
    """Represents an error pattern."""
    type: ErrorType
    message_index: int
    error_message: str
    recovery_attempted: bool = False
    recovery_successful: bool = False


class InterventionDetector:
    """Detects user interventions in conversations."""
    
    def __init__(self):
        # System-generated message patterns to filter out
        self.system_message_patterns = [
            r"The user doesn't want to proceed with this tool use",
            r"Tool use was rejected.*STOP what you are doing",
            r"\[Request interrupted by user\]",  # System intervention message
            # Note: Removed line number pattern as it was too broad
            r"^\s*===+\s*test session starts\s*===+",  # Test output
            r"^\[!\].*Using tool",  # Tool usage notifications
            r"^<function_calls>",  # Function call markers
            r"^🎯.*Planning task:",  # Planning markers
            r"^System:",  # System messages
            r"^<command-message>.*</command-message>",  # Claude Code command messages
            r"^<command-name>.*</command-name>",  # Claude Code command names
        ]
        
        # Truncation indicators - be conservative to avoid false positives
        self.truncation_indicators = [
            r'\.\.\.$',  # Ends with ellipsis
            r'\.\.\.[\s\'"]*$',  # Ends with ellipsis and maybe quotes
            r'^\s*\.\.\.',  # Starts with ellipsis (continuation)
            r'<truncated>',  # Explicit truncation marker
            r'output truncated',  # Common truncation message
            r':\s*$',  # Ends with colon (often before missing content)
        ]
        
        self.intervention_patterns = {
            InterventionType.STOP_ACTION: [
                r'\b(stop|wait|hold on|pause)\b',
                r"(don't|do not)\s+(do|make|create|run)\s+that",
                r'\[Request interrupted by user\]',
            ],
            InterventionType.CORRECTION: [
                r'(no,?\s+)?(not|isn\'t|wasn\'t)\s+(that|correct|right)',
                r'(actually|instead)',
                r'(wrong|incorrect|mistake)',
            ],
            InterventionType.CLARIFICATION: [
                r'(let me|I\'ll)\s+explain',
                r'what I meant was',
                r'to clarify',
            ],
            InterventionType.SCOPE_REDIRECT: [
                r"(don't|no need to)\s+(refactor|change|modify)\s+that",
                r'(just|only)\s+do',
                r'stick to',
            ],
            InterventionType.APPROACH_CHANGE: [
                r'(try|use)\s+(\w+)\s+instead',
                r'different approach',
                r'let\'s go with',
            ],
            InterventionType.TOOL_REJECTION: [
                r'Tool use was rejected',
                r'user doesn\'t want to proceed',
            ],
            InterventionType.TAKEOVER: [
                r'(I\'ll|let me)\s+(handle|do|take care of)\s+(this|that|it)',
                r'I\'ll\s+\w+\s+it myself',
            ]
        }
    
    def detect_interventions(self, messages: List[Message]) -> List[Intervention]:
        """Detect interventions in a conversation."""
        interventions = []
        
        for i, message in enumerate(messages):
            if message.role != 'user':
                continue
            
            content = self._extract_text_content(message.content)
            if not content:
                # Log when we skip messages with no extractable content
                logger.debug(f"Skipping message {i} - no extractable user text content")
                continue
            
            # Check for intervention patterns (no filtering here - all filtering happens later)
            for intervention_type, patterns in self.intervention_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        logger.debug(f"Detected {intervention_type.value} at message {i} matching pattern '{pattern}': {content[:50]}...")
                        intervention = self._create_intervention(
                            intervention_type, i, content, messages
                        )
                        if intervention:
                            interventions.append(intervention)
                        break
        
        logger.info(f"Detected {len(interventions)} total interventions in conversation")
        return interventions
    
    def _create_intervention(self, 
                           intervention_type: InterventionType,
                           message_index: int,
                           user_message: str,
                           messages: List[Message]) -> Optional[Intervention]:
        """Create an intervention object with context."""
        # Get context before intervention
        context_before = None
        claude_action = None
        
        if message_index > 0:
            prev_message = messages[message_index - 1]
            if prev_message.role == 'assistant':
                context_before = self._extract_text_content(prev_message.content)
                # Check if Claude was about to use a tool
                if prev_message.tool_use:
                    claude_action = f"Tool: {prev_message.tool_use.get('name', 'unknown')}"
        
        # Determine severity
        severity = self._determine_severity(intervention_type, user_message)
        
        return Intervention(
            type=intervention_type,
            message_index=message_index,
            user_message=user_message[:500],  # Increased context capture
            context_before=context_before[:500] if context_before else None,
            claude_action_interrupted=claude_action,
            severity=severity
        )
    
    def _determine_severity(self, intervention_type: InterventionType, message: str) -> str:
        """Determine intervention severity."""
        message_lower = message.lower()
        
        # Critical severity indicators - potential data loss or security issues
        critical_indicators = [
            'delete', 'production', 'database', 'security', 'credentials',
            'private', 'secret', 'destroy', 'remove all', 'drop table'
        ]
        if any(indicator in message_lower for indicator in critical_indicators):
            return "high"
        
        # High severity indicators - wrong action or understanding
        high_severity_words = ['stop', 'wrong', 'don\'t do that', 'mistake', 
                              'no no no', 'absolutely not', 'incorrect']
        if any(word in message_lower for word in high_severity_words):
            return "high"
        
        # High severity intervention types
        if intervention_type in [InterventionType.STOP_ACTION, InterventionType.TAKEOVER]:
            return "high"
        
        # Medium severity types - need correction but not urgent
        if intervention_type in [InterventionType.CORRECTION, InterventionType.TOOL_REJECTION,
                                InterventionType.APPROACH_CHANGE]:
            return "medium"
        
        # Low severity types - minor adjustments
        if intervention_type in [InterventionType.CLARIFICATION, InterventionType.SCOPE_REDIRECT]:
            return "low"
        
        return "medium"
    
    
    def is_obviously_low_quality(self, intervention: Intervention) -> Tuple[bool, Optional[str]]:
        """Quick check for obviously low-quality interventions that we can skip without LLM.
        
        This is the single place where all filtering decisions are made.
        
        Returns:
            Tuple of (is_low_quality, reason_for_filtering)
        """
        content = intervention.user_message
        content_lower = content.lower()
        content_stripped = content.strip()
        
        # 1. Check for system-generated messages
        for pattern in self.system_message_patterns:
            if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                # Find which pattern matched for logging
                pattern_desc = pattern[:30] + "..." if len(pattern) > 30 else pattern
                reason = f"system_message: matched '{pattern_desc}'"
                logger.debug(f"Filtering intervention - {reason}: '{content[:50]}...'")
                return True, reason
        
        # 2. Skip very short messages (likely incomplete or just signals)
        if len(content_stripped) < 10:
            reason = f"too_short: {len(content_stripped)} chars"
            logger.debug(f"Filtering intervention - {reason}: '{content_stripped}'")
            return True, reason
        
        # 3. Check for truncation indicators (but be careful with code)
        # Only apply truncation check if message is short and looks incomplete
        if len(content) < 100:
            for pattern in self.truncation_indicators:
                if re.search(pattern, content_stripped):
                    # Exception for code blocks or commands
                    if not any(indicator in content for indicator in ['```', '$(', 'function', 'def', 'class']):
                        reason = f"truncated: matched pattern '{pattern}'"
                        logger.debug(f"Filtering intervention - {reason}: '{content[:50]}...'")
                        return True, reason
        
        # 4. Check intervention type-specific filters BEFORE generic short message filters
        
        # 4a. Pure tool cancellations without explanation
        if intervention.type == InterventionType.TOOL_REJECTION:
            # If the message is just about canceling a tool without teaching
            # Don't filter if it contains guidance words
            guidance_words = ['instead', 'should', 'rather', 'better', 'prefer', 'approach']
            has_guidance = any(word in content_lower for word in guidance_words)
            if len(content_stripped) < 30 and not has_guidance:
                reason = f"tool_rejection_without_guidance: {len(content_stripped)} chars"
                logger.debug(f"Filtering intervention - {reason}: '{content[:50]}...'")
                return True, reason
        
        # 5. Very short messages that are just stop signals
        if len(content_stripped) < 20:
            stop_words = ['stop', 'wait', 'no', 'hold on']
            for word in stop_words:
                if word in content_lower:
                    reason = f"short_stop_signal: {len(content_stripped)} chars with '{word}'"
                    logger.debug(f"Filtering intervention - {reason}: '{content_stripped}'")
                    return True, reason
        
        # 6. User explicitly taking over to run something themselves (without explanation)
        # Only filter short takeover messages without additional context
        if len(content_stripped) < 30:  # Short messages only
            takeover_phrases = [
                r'i\'ll (do|run|handle) (it|that) myself',
                r'let me (do|run) (it|that)$',  # Must end with "that" or "it"
                r'i want to see the output',
                r'run.*in.*terminal',
                r'i\'ll take it from here',
            ]
            for phrase in takeover_phrases:
                if re.search(phrase, content_lower):
                    reason = f"short_takeover: matched '{phrase[:30]}...'"
                    logger.debug(f"Filtering intervention - {reason}: '{content[:50]}...'")
                    return True, reason
        
        return False, None
    
    def _extract_text_content(self, content: Any) -> Optional[str]:
        """Extract text from various content formats.
        
        Only extracts actual user text, not tool results or system content.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    # Skip tool results - these are not user messages
                    if item.get('type') == 'tool_result':
                        continue
                    # Extract text from text-type items
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    # Skip other dict items that have 'content' - likely tool results
                elif isinstance(item, str):
                    text_parts.append(item)
            return ' '.join(text_parts) if text_parts else None
        elif isinstance(content, dict):
            # Only return text if it's explicitly a text item
            if content.get('type') == 'text':
                return content.get('text', '')
            # Skip tool results and other non-text content
            return None
        return None


class SuccessPatternDetector:
    """Detects successful interaction patterns."""
    
    def __init__(self):
        self.success_patterns = {
            SuccessIndicator.TASK_COMPLETED: [
                r'(done|completed|finished)',
                r'(works|working)\s+(now|perfectly|great)',
                r'task\s+completed',
            ],
            SuccessIndicator.TESTS_PASSED: [
                r'(tests?\s+pass)',
                r'all\s+tests?\s+passing',
                r'✓|✔|passed',
                r'\d+\s+passing',
            ],
            SuccessIndicator.USER_SATISFIED: [
                r'(perfect|excellent|great|thanks)',
                r'exactly\s+what\s+I\s+(wanted|needed)',
                r'(solved|fixed)\s+my\s+problem',
            ],
            SuccessIndicator.BUILD_SUCCESS: [
                r'build\s+success',
                r'compiled?\s+successfully',
                r'no\s+errors?',
            ]
        }
    
    def detect_success_patterns(self, messages: List[Message]) -> List[SuccessPattern]:
        """Detect success patterns in conversation."""
        patterns = []
        tool_sequence = []
        
        for i, message in enumerate(messages):
            # Track tool usage
            if message.role == 'assistant' and message.tool_use:
                tool_name = message.tool_use.get('name', 'unknown')
                tool_sequence.append(tool_name)
            
            # Check for success indicators
            content = self._extract_text_content(message.content)
            if not content:
                continue
            
            for indicator, pattern_list in self.success_patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, content, re.IGNORECASE):
                        success = SuccessPattern(
                            indicator=indicator,
                            message_index=i,
                            pattern_description=f"{indicator.value}: {pattern}",
                            tool_sequence=tool_sequence.copy()
                        )
                        patterns.append(success)
                        break
        
        return patterns
    
    def _extract_text_content(self, content: Any) -> Optional[str]:
        """Extract text from various content formats.
        
        Only extracts actual user text, not tool results or system content.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    # Skip tool results - these are not user messages
                    if item.get('type') == 'tool_result':
                        continue
                    # Extract text from text-type items
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            return ' '.join(text_parts) if text_parts else None
        elif isinstance(content, dict):
            # Only return text if it's explicitly a text item
            if content.get('type') == 'text':
                return content.get('text', '')
            return None
        return None


class ErrorPatternDetector:
    """Detects error patterns in conversations."""
    
    def __init__(self):
        self.error_patterns = {
            ErrorType.FILE_NOT_FOUND: [
                r'(file|path)\s+not\s+found',
                r'no\s+such\s+file',
                r'FileNotFoundError',
                r'ENOENT',
            ],
            ErrorType.SYNTAX_ERROR: [
                r'SyntaxError',
                r'unexpected\s+token',
                r'invalid\s+syntax',
                r'parsing\s+error',
            ],
            ErrorType.TEST_FAILURE: [
                r'test\s+fail',
                r'\d+\s+failing',
                r'✗|✘|failed',
                r'AssertionError',
            ],
            ErrorType.BUILD_FAILURE: [
                r'build\s+fail',
                r'compilation\s+error',
                r'npm\s+ERR!',
                r'error\s+TS\d+',
            ],
            ErrorType.TOOL_ERROR: [
                r'Tool\s+ran\s+with\s+errors?',
                r'command\s+failed',
                r'exit\s+code\s+[1-9]',
            ]
        }
    
    def detect_error_patterns(self, messages: List[Message]) -> List[ErrorPattern]:
        """Detect error patterns in conversation."""
        errors = []
        
        for i, message in enumerate(messages):
            content = self._extract_text_content(message.content)
            if not content:
                continue
            
            for error_type, pattern_list in self.error_patterns.items():
                for pattern in pattern_list:
                    if re.search(pattern, content, re.IGNORECASE):
                        # Check for recovery
                        recovery_attempted, recovery_successful = self._check_recovery(messages, i)
                        
                        error = ErrorPattern(
                            type=error_type,
                            message_index=i,
                            error_message=content[:200],
                            recovery_attempted=recovery_attempted,
                            recovery_successful=recovery_successful
                        )
                        errors.append(error)
                        break
        
        return errors
    
    def _check_recovery(self, messages: List[Message], error_index: int) -> Tuple[bool, bool]:
        """Check if error was recovered from."""
        recovery_attempted = False
        recovery_successful = False
        
        # Look at subsequent messages
        for i in range(error_index + 1, min(error_index + 10, len(messages))):
            message = messages[i]
            content = self._extract_text_content(message.content)
            
            if not content:
                continue
            
            # Check for recovery attempts
            if message.role == 'assistant':
                recovery_keywords = ['fix', 'resolve', 'correct', 'retry', 'try again']
                if any(keyword in content.lower() for keyword in recovery_keywords):
                    recovery_attempted = True
            
            # Check for success after error
            success_keywords = ['fixed', 'resolved', 'works now', 'success', 'passed']
            if any(keyword in content.lower() for keyword in success_keywords):
                recovery_successful = True
                break
        
        return recovery_attempted, recovery_successful
    
    def _extract_text_content(self, content: Any) -> Optional[str]:
        """Extract text from various content formats.
        
        Only extracts actual user text, not tool results or system content.
        """
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    # Skip tool results - these are not user messages
                    if item.get('type') == 'tool_result':
                        continue
                    # Extract text from text-type items
                    if item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                elif isinstance(item, str):
                    text_parts.append(item)
            return ' '.join(text_parts) if text_parts else None
        elif isinstance(content, dict):
            # Only return text if it's explicitly a text item
            if content.get('type') == 'text':
                return content.get('text', '')
            return None
        return None