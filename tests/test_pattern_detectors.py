"""Tests for Pattern Detectors module."""

import pytest
from src.detectors.pattern_detectors import InterventionDetector, Intervention, InterventionType
from src.conversation_scanner import Message


class TestInterventionDetector:
    """Test suite for InterventionDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create an intervention detector instance."""
        return InterventionDetector()
    
    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            Message(role="user", content="Build me a web app", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="I'll help you build a web app", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="Actually, stop. Let me think about this differently", timestamp="2024-01-01T00:00:02"),
            Message(role="assistant", content="Of course, I'll wait", timestamp="2024-01-01T00:00:03"),
        ]
    
    def test_detect_stop_intervention(self, detector):
        """Test detection of stop interventions."""
        messages = [
            Message(role="user", content="Build something", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="Building...", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="Stop! Wait a moment", timestamp="2024-01-01T00:00:02"),
        ]
        
        interventions = detector.detect_interventions(messages)
        assert len(interventions) > 0
        assert any(i.type == InterventionType.STOP_ACTION for i in interventions)
    
    def test_detect_correction_intervention(self, detector):
        """Test detection of correction interventions."""
        messages = [
            Message(role="user", content="Create a function", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="def my_function():", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="No, that's wrong. It should be async", timestamp="2024-01-01T00:00:02"),
        ]
        
        interventions = detector.detect_interventions(messages)
        assert len(interventions) > 0
        assert any(i.type == InterventionType.CORRECTION for i in interventions)
    
    def test_system_messages_filtered(self, detector):
        """Test that system messages are detected but marked as low quality."""
        messages = [
            Message(role="user", content="Do something", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="Doing...", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="[Request interrupted by user]", timestamp="2024-01-01T00:00:02"),
            Message(role="user", content="Actually, wait, let me explain", timestamp="2024-01-01T00:00:03"),
        ]
        
        interventions = detector.detect_interventions(messages)
        
        # System messages might be detected if they match patterns
        # But they should be marked as low quality
        for intervention in interventions:
            if "[Request interrupted" in intervention.user_message:
                assert detector.is_obviously_low_quality(intervention) == True
    
    def test_truncated_messages_filtered(self, detector):
        """Test that truncated messages are detected but marked as low quality."""
        messages = [
            Message(role="user", content="Build something", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="Building...", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="continue...", timestamp="2024-01-01T00:00:02"),  # Ends with ellipsis
        ]
        
        interventions = detector.detect_interventions(messages)
        
        # Truncated messages should be filtered as low quality if detected
        for intervention in interventions:
            if intervention.user_message.endswith("..."):
                assert detector.is_obviously_low_quality(intervention) == True
    
    def test_is_obviously_low_quality_short_stops(self, detector):
        """Test that very short stop messages are flagged as low quality."""
        test_cases = [
            ("stop", True),
            ("wait", True),
            ("no", True),
            ("hold on", True),
            ("stop, I need to explain something important", False),
            ("wait, let's use a different approach", False),
        ]
        
        for message, expected_low_quality in test_cases:
            intervention = Intervention(
                type=InterventionType.STOP_ACTION,
                message_index=1,
                user_message=message,
                context_before="",
                severity="low"
            )
            
            result = detector.is_obviously_low_quality(intervention)
            assert result == expected_low_quality, f"Failed for message: '{message}'"
    
    def test_is_obviously_low_quality_takeovers(self, detector):
        """Test that user takeovers are flagged as low quality."""
        test_cases = [
            ("I'll do it myself", True),
            ("let me do that", True),
            ("I want to see the output", True),
            ("run it in terminal", True),
            ("I'll take it from here", True),
            ("Let me handle the authentication logic myself", False),  # Longer message, not obviously low quality
            ("I'll implement this differently because we need better error handling", False),
        ]
        
        for message, expected_low_quality in test_cases:
            intervention = Intervention(
                type=InterventionType.TAKEOVER,
                message_index=1,
                user_message=message,
                context_before="",
                severity="low"
            )
            
            result = detector.is_obviously_low_quality(intervention)
            assert result == expected_low_quality, f"Failed for message: '{message}'"
    
    def test_is_obviously_low_quality_system_messages(self, detector):
        """Test that system messages are always flagged as low quality."""
        system_messages = [
            "[!] Using tool Read with input {file: 'test.py'}",
            "🎯 Planning task: Build feature",
            "System: Connection interrupted",
            "[Request interrupted by user]",
            "<function_calls>",
        ]
        
        for message in system_messages:
            intervention = Intervention(
                type=InterventionType.CORRECTION,
                message_index=1,
                user_message=message,
                context_before="",
                severity="medium"
            )
            
            result = detector.is_obviously_low_quality(intervention)
            assert result == True, f"System message not filtered: '{message}'"
    
    def test_is_obviously_low_quality_tool_rejections(self, detector):
        """Test tool rejection quality detection."""
        test_cases = [
            # Short rejections without guidance - low quality
            ("cancel", InterventionType.TOOL_REJECTION, True),
            ("don't do that", InterventionType.TOOL_REJECTION, True),
            ("stop that tool", InterventionType.TOOL_REJECTION, True),
            
            # Rejections with guidance - high quality
            ("Don't use that tool, instead we should use a different approach", 
             InterventionType.TOOL_REJECTION, False),
            ("Cancel that, we should handle errors differently", 
             InterventionType.TOOL_REJECTION, False),
        ]
        
        for message, intervention_type, expected_low_quality in test_cases:
            intervention = Intervention(
                type=intervention_type,
                message_index=1,
                user_message=message,
                context_before="",
                severity="medium"
            )
            
            result = detector.is_obviously_low_quality(intervention)
            assert result == expected_low_quality, f"Failed for message: '{message}'"
    
    def test_detect_interventions_comprehensive(self, detector):
        """Test comprehensive intervention detection scenario."""
        messages = [
            Message(role="user", content="Build a REST API", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="I'll create a REST API using Express", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="No wait, use FastAPI instead", timestamp="2024-01-01T00:00:02"),
            Message(role="assistant", content="Switching to FastAPI", timestamp="2024-01-01T00:00:03"),
            Message(role="user", content="[Request interrupted by user]", timestamp="2024-01-01T00:00:04"),  # System
            Message(role="user", content="Actually, let me clarify the requirements first", timestamp="2024-01-01T00:00:05"),
            Message(role="assistant", content="Please go ahead", timestamp="2024-01-01T00:00:06"),
            Message(role="user", content="stop", timestamp="2024-01-01T00:00:07"),  # Short stop
        ]
        
        interventions = detector.detect_interventions(messages)
        
        # Should detect interventions including short stops
        assert len(interventions) >= 1
        
        # The "No wait, use FastAPI instead" should be detected and NOT low quality
        fastapi_intervention = next((i for i in interventions 
                                    if "FastAPI" in i.user_message), None)
        assert fastapi_intervention is not None
        assert detector.is_obviously_low_quality(fastapi_intervention) == False
        
        # Short "stop" should be detected but marked as low quality
        stop_intervention = next((i for i in interventions 
                                 if i.user_message == "stop"), None)
        if stop_intervention:
            assert detector.is_obviously_low_quality(stop_intervention) == True
        
        # System messages should be marked as low quality if detected
        for intervention in interventions:
            if "[Request interrupted" in intervention.user_message:
                assert detector.is_obviously_low_quality(intervention) == True
    
    def test_empty_messages(self, detector):
        """Test handling of empty message list."""
        interventions = detector.detect_interventions([])
        assert interventions == []
    
    def test_no_interventions(self, detector):
        """Test conversation with no interventions."""
        messages = [
            Message(role="user", content="What's the weather?", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="I can't check weather", timestamp="2024-01-01T00:00:01"),
            Message(role="user", content="Thanks for letting me know", timestamp="2024-01-01T00:00:02"),
        ]
        
        interventions = detector.detect_interventions(messages)
        assert len(interventions) == 0