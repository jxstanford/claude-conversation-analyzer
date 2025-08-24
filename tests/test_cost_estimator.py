"""Tests for Cost Estimator module."""

import pytest
from unittest.mock import patch, Mock
from src.cost_estimator import CostEstimator, ModelPricing, CLAUDE_PRICING
from src.conversation_scanner import Message, ConversationFile


class TestCostEstimator:
    """Test suite for CostEstimator."""
    
    @pytest.fixture
    def estimator(self):
        """Create a cost estimator instance."""
        return CostEstimator()
    
    @pytest.fixture
    def sample_conversations(self):
        """Create sample conversations for testing."""
        messages = [
            Message(role="user", content="Help me build something", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="I'll help you build it", timestamp="2024-01-01T00:00:01"),
        ]
        
        conv_file = ConversationFile(
            conversation_id="test-123",
            project_id="proj-456",
            file_path="/test/path",
            last_modified="2024-01-01",
            message_count=2
        )
        
        # Create 10 conversations
        return [(conv_file, messages) for _ in range(10)]
    
    def test_token_counting_with_tiktoken(self, estimator):
        """Test token counting with tiktoken encoder."""
        # Should use tiktoken if available
        if estimator.encoder:
            text = "Hello, world! This is a test."
            tokens = estimator.count_tokens(text)
            assert tokens > 0
            assert tokens < len(text)  # Tokens should be less than character count
    
    def test_token_counting_fallback(self):
        """Test token counting fallback when tiktoken is not available."""
        with patch('src.cost_estimator.tiktoken.get_encoding', side_effect=Exception("No tiktoken")):
            estimator = CostEstimator()
            text = "Hello, world! This is a test."
            tokens = estimator.count_tokens(text)
            # Fallback uses 4 chars per token approximation
            assert tokens == len(text) // 4
    
    def test_calculate_cost(self, estimator):
        """Test cost calculation for different models."""
        # Test Haiku pricing
        haiku_pricing = CLAUDE_PRICING["claude-3-5-haiku-latest"]
        cost = estimator.calculate_cost(1000000, 500000, haiku_pricing)
        
        # 1M input tokens at $1/M + 500K output tokens at $5/M
        expected_cost = 1.0 + 2.5
        assert abs(cost - expected_cost) < 0.01
        
        # Test Opus pricing
        opus_pricing = CLAUDE_PRICING["claude-3-opus-latest"]
        cost = estimator.calculate_cost(100000, 50000, opus_pricing)
        
        # 100K input tokens at $15/M + 50K output tokens at $75/M
        expected_cost = 1.5 + 3.75
        assert abs(cost - expected_cost) < 0.01
    
    def test_estimate_conversation_tokens(self, estimator):
        """Test token estimation for a conversation."""
        messages = [
            Message(role="user", content="Hello " * 100, timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="Hi " * 100, timestamp="2024-01-01T00:00:01"),
        ]
        
        tokens = estimator.estimate_conversation_tokens(messages)
        assert tokens > 0
        # Should include overhead for role/metadata
        assert tokens > estimator.count_tokens("Hello " * 100 + "Hi " * 100)
    
    def test_estimate_phase_costs(self, estimator, sample_conversations):
        """Test phase-by-phase cost estimation."""
        estimates = estimator.estimate_phase_costs(sample_conversations)
        
        # Check all phases are present
        assert 'classification' in estimates
        assert 'intervention_analysis' in estimates
        assert 'quality_classification' in estimates
        assert 'deep_analysis' in estimates
        assert 'total' in estimates
        
        # Check total calculation
        total = estimates['total']
        assert total['conversations_analyzed'] == 10
        assert total['cost_usd'] > 0
        assert total['input_tokens'] > 0
        assert total['output_tokens'] > 0
        
        # Check phase calculations
        classification = estimates['classification']
        assert classification['conversations'] == 10
        assert classification['cost_usd'] > 0
        
        # Intervention analysis should be ~30% of conversations
        intervention = estimates['intervention_analysis']
        assert intervention['conversations'] == 3  # 30% of 10
        
        # Deep analysis should be ~10% of total
        deep = estimates['deep_analysis']
        assert deep['conversations'] >= 1
    
    def test_estimate_phase_costs_with_claude_md(self, estimator, sample_conversations):
        """Test cost estimation when CLAUDE.md is provided."""
        claude_md_content = "# CLAUDE.md\n" + "Test content " * 1000
        
        estimates = estimator.estimate_phase_costs(
            sample_conversations,
            claude_md_content=claude_md_content
        )
        
        # Should include synthesis phase
        assert 'claude_md_synthesis' in estimates
        synthesis = estimates['claude_md_synthesis']
        assert synthesis['claude_md_tokens'] > 0
        assert synthesis['cost_usd'] > 0
    
    def test_custom_models(self):
        """Test using custom model configuration."""
        custom_models = {
            'classifier': 'claude-3-5-haiku-latest',
            'analyzer': 'claude-3-5-sonnet-latest',
            'synthesizer': 'claude-3-opus-latest'
        }
        
        estimator = CostEstimator(models=custom_models)
        assert estimator.models == custom_models
    
    def test_cost_breakdown_string(self, estimator, sample_conversations):
        """Test generating cost breakdown as string."""
        estimates = estimator.estimate_phase_costs(sample_conversations)
        breakdown = estimator.get_cost_breakdown_string(estimates)
        
        assert "COST ESTIMATE" in breakdown
        assert "Total Estimated Cost" in breakdown
        assert "Classification" in breakdown
        assert "Model Configuration" in breakdown
        assert "$" in breakdown  # Should include cost values
    
    def test_display_cost_estimate(self, estimator, sample_conversations, capsys):
        """Test console display of cost estimates."""
        estimates = estimator.estimate_phase_costs(sample_conversations)
        
        # Mock rich console to test output
        with patch('src.cost_estimator.console') as mock_console:
            estimator.display_cost_estimate(estimates, show_details=True)
            
            # Should have called print for summary and table
            assert mock_console.print.called
            call_count = mock_console.print.call_count
            assert call_count >= 2  # At least summary and table
    
    def test_empty_conversations(self, estimator):
        """Test handling of empty conversation list."""
        estimates = estimator.estimate_phase_costs([])
        
        assert estimates['total']['conversations_analyzed'] == 0
        assert estimates['total']['cost_usd'] == 0
        assert estimates['total']['input_tokens'] == 0
    
    def test_large_conversation_set(self, estimator):
        """Test estimation for large number of conversations."""
        # Create 1000 conversations
        messages = [
            Message(role="user", content="Test", timestamp="2024-01-01T00:00:00"),
            Message(role="assistant", content="Response", timestamp="2024-01-01T00:00:01"),
        ]
        
        conv_file = ConversationFile(
            conversation_id="test",
            project_id="proj",
            file_path="/test",
            last_modified="2024-01-01",
            message_count=2
        )
        
        large_conversations = [(conv_file, messages) for _ in range(1000)]
        estimates = estimator.estimate_phase_costs(large_conversations)
        
        assert estimates['total']['conversations_analyzed'] == 1000
        assert estimates['intervention_analysis']['conversations'] == 300  # 30% of 1000
        assert estimates['deep_analysis']['conversations'] >= 50  # At least 5% of 1000
        
        # Cost should scale with conversation count
        assert estimates['total']['cost_usd'] > 10  # Should be substantial for 1000 conversations