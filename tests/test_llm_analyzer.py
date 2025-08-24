"""Tests for LLM Analyzer module."""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import List

from src.analyzers.llm_analyzer import LLMAnalyzer, ConversationClassification, InterventionAnalysis
from src.conversation_scanner import Message, ConversationFile
from src.detectors import Intervention, InterventionType


class TestLLMAnalyzer:
    """Test suite for LLMAnalyzer."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock LLM analyzer."""
        with patch.dict('os.environ', {'ANTHROPIC_API_KEY': 'test-key'}):
            analyzer = LLMAnalyzer(api_key='test-key')
            # Mock the async client
            analyzer.async_client = AsyncMock()
            return analyzer
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            Message(role="user", content="Help me build a web app", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="I'll help you build a web app", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="stop, let me do it differently", timestamp="2024-01-01T00:00:02"),
            Message(role="assistant", content="Of course, I'll follow your lead", timestamp="2024-01-01T00:00:03"),
        ]
    
    @pytest.fixture
    def sample_classification(self):
        """Create a sample conversation classification."""
        return ConversationClassification(
            conversation_id="test-123",
            user_intent="build_feature",
            task_type="development",
            complexity="medium",
            has_interventions=True,
            intervention_count=1,
            task_completed=False,
            success_level="partial",
            conversation_tone="bumpy",
            notable_features=["user correction"]
        )
    
    @pytest.fixture
    def sample_intervention(self):
        """Create a sample intervention."""
        return Intervention(
            type=InterventionType.CORRECTION,
            message_index=2,
            user_message="stop, let me do it differently",
            context_before="Building web app",
            severity="medium"
        )
    
    @pytest.mark.asyncio
    async def test_analyze_problematic_conversation_empty_response(self, mock_analyzer, sample_messages, sample_classification):
        """Test handling of empty API response in _analyze_problematic_conversation."""
        # Mock empty response
        mock_response = Mock()
        mock_response.content = []
        mock_analyzer.async_client.messages.create.return_value = mock_response
        
        conv_file = ConversationFile(
            conversation_id="test-123",
            project_path="proj-456",
            file_path=Path("/test/path"),
            message_count=4
        )
        
        result = await mock_analyzer._analyze_problematic_conversation(
            conv_file, sample_messages, sample_classification
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_problematic_conversation_no_text_content(self, mock_analyzer, sample_messages, sample_classification):
        """Test handling of response with no text content."""
        # Mock response with content but no text
        mock_content = Mock()
        mock_content.text = None
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_analyzer.async_client.messages.create.return_value = mock_response
        
        conv_file = ConversationFile(
            conversation_id="test-123",
            project_path="proj-456",
            file_path=Path("/test/path"),
            message_count=4
        )
        
        result = await mock_analyzer._analyze_problematic_conversation(
            conv_file, sample_messages, sample_classification
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_problematic_conversation_invalid_json(self, mock_analyzer, sample_messages, sample_classification):
        """Test handling of invalid JSON in response."""
        # Mock response with invalid JSON
        mock_content = Mock()
        mock_content.text = "This is not valid JSON"
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_analyzer.async_client.messages.create.return_value = mock_response
        
        conv_file = ConversationFile(
            conversation_id="test-123",
            project_path="proj-456",
            file_path=Path("/test/path"),
            message_count=4
        )
        
        result = await mock_analyzer._analyze_problematic_conversation(
            conv_file, sample_messages, sample_classification
        )
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_analyze_problematic_conversation_valid_response(self, mock_analyzer, sample_messages, sample_classification):
        """Test successful parsing of valid response."""
        # Mock valid response
        valid_json = {
            "root_cause": "unclear requirements",
            "what_went_wrong": "user had different expectations",
            "claude_assumption": "assumed web framework",
            "user_expectation": "wanted simple HTML",
            "prevention_rule": "always clarify tech stack first",
            "severity_assessment": "medium",
            "pattern_category": "assumption_mismatch"
        }
        
        mock_content = Mock()
        mock_content.text = f"Here's the analysis: {json.dumps(valid_json)}"
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_analyzer.async_client.messages.create.return_value = mock_response
        
        conv_file = ConversationFile(
            conversation_id="test-123",
            project_path="proj-456",
            file_path=Path("/test/path"),
            message_count=4
        )
        
        result = await mock_analyzer._analyze_problematic_conversation(
            conv_file, sample_messages, sample_classification
        )
        
        assert result is not None
        assert isinstance(result, InterventionAnalysis)
        assert result.root_cause == "unclear requirements"
        assert result.what_went_wrong == "user had different expectations"
    
    @pytest.mark.asyncio
    async def test_analyze_multiple_interventions_filtering(self, mock_analyzer, sample_messages, sample_classification):
        """Test that low-quality interventions are filtered."""
        # Create interventions with varying quality
        interventions = [
            Intervention(
                type=InterventionType.STOP_ACTION,
                message_index=1,
                user_message="stop",  # Very short, should be filtered
                context_before="context",
                severity="low"
            ),
            Intervention(
                type=InterventionType.CORRECTION,
                message_index=2,
                user_message="Actually, let's use a different approach because the current one won't scale",
                context_before="context",
                severity="medium"
            ),
            Intervention(
                type=InterventionType.TOOL_REJECTION,
                message_index=3,
                user_message="[!] Using tool Read with input {...}",  # System message, should be filtered
                context_before="context",
                severity="low"
            ),
        ]
        
        # Mock the API response
        mock_content = Mock()
        mock_content.text = json.dumps({
            "interventions": [
                {
                    "root_cause": "approach issue",
                    "what_went_wrong": "scaling concerns",
                    "claude_assumption": "simple solution",
                    "user_expectation": "scalable solution",
                    "prevention_rule": "consider scale early",
                    "severity_assessment": "high",
                    "pattern_category": "requirements_mismatch"
                }
            ]
        })
        mock_response = Mock()
        mock_response.content = [mock_content]
        mock_analyzer.async_client.messages.create.return_value = mock_response
        
        result = await mock_analyzer._analyze_multiple_interventions(
            interventions, sample_messages, sample_classification
        )
        
        # Should have filtered out the low-quality interventions
        assert len(result) <= 1  # Only the meaningful correction should remain
    
    
    def test_truncate_message(self, mock_analyzer):
        """Test message truncation utility."""
        long_message = "a" * 1000
        truncated = mock_analyzer._truncate_message(long_message, max_length=100)
        
        assert len(truncated) <= 103  # 100 + "..."
        assert truncated.endswith("...")
        
        short_message = "short"
        not_truncated = mock_analyzer._truncate_message(short_message, max_length=100)
        assert not_truncated == short_message
    


class TestInterventionFiltering:
    """Test intervention filtering logic."""
    
    def test_is_obviously_low_quality_short_stop(self):
        """Test that short stop messages are filtered."""
        from src.detectors import InterventionDetector
        
        detector = InterventionDetector()
        
        # Very short stop message
        intervention = Intervention(
            type=InterventionType.STOP_ACTION,
            message_index=1,
            user_message="stop",
            context_before="",
            severity="low"
        )
        
        is_low_quality, reason = detector.is_obviously_low_quality(intervention)
        assert is_low_quality == True
    
    def test_is_obviously_low_quality_system_message(self):
        """Test that system messages are filtered."""
        from src.detectors import InterventionDetector
        
        detector = InterventionDetector()
        
        # System message
        intervention = Intervention(
            type=InterventionType.CORRECTION,
            message_index=1,
            user_message="[!] Using tool Read with input {file: 'test.py'}",
            context_before="",
            severity="low"
        )
        
        is_low_quality, reason = detector.is_obviously_low_quality(intervention)
        assert is_low_quality == True
    
    def test_is_obviously_low_quality_meaningful_correction(self):
        """Test that meaningful corrections are NOT filtered."""
        from src.detectors import InterventionDetector
        
        detector = InterventionDetector()
        
        # Meaningful correction
        intervention = Intervention(
            type=InterventionType.CORRECTION,
            message_index=1,
            user_message="Actually, we should use async/await here instead of callbacks for better readability",
            context_before="",
            severity="medium"
        )
        
        is_low_quality, reason = detector.is_obviously_low_quality(intervention)
        assert is_low_quality == False
    
    def test_is_obviously_low_quality_takeover(self):
        """Test that user takeovers are filtered."""
        from src.detectors import InterventionDetector
        
        detector = InterventionDetector()
        
        # User takeover
        intervention = Intervention(
            type=InterventionType.TAKEOVER,
            message_index=1,
            user_message="I'll do it myself",
            context_before="",
            severity="low"
        )
        
        is_low_quality, reason = detector.is_obviously_low_quality(intervention)
        assert is_low_quality == True
    
    def test_is_obviously_low_quality_tool_rejection_with_guidance(self):
        """Test that tool rejections with guidance are NOT filtered."""
        from src.detectors import InterventionDetector
        
        detector = InterventionDetector()
        
        # Tool rejection with guidance
        intervention = Intervention(
            type=InterventionType.TOOL_REJECTION,
            message_index=1,
            user_message="Don't use that tool, instead we should approach this differently by first understanding the requirements",
            context_before="",
            severity="medium"
        )
        
        is_low_quality, reason = detector.is_obviously_low_quality(intervention)
        assert is_low_quality == False