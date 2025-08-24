"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import os
from pathlib import Path


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    # Set test environment variables
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-api-key")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    
    # Set test model configurations
    monkeypatch.setenv("CLAUDE_CLASSIFIER_MODEL", "claude-3-5-haiku-latest")
    monkeypatch.setenv("CLAUDE_ANALYZER_MODEL", "claude-3-5-sonnet-latest")
    monkeypatch.setenv("CLAUDE_SYNTHESIZER_MODEL", "claude-3-opus-latest")


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent.parent / "test_data"