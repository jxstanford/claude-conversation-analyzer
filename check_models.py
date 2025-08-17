#!/usr/bin/env python3
"""Check available Anthropic models."""

import os
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize client
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

# Test common model names
test_models = [
    # Haiku
    "claude-3-haiku-20240307",
    "claude-3-5-haiku-20241022",
    "claude-4-haiku-20241022",
    "claude-haiku-4-20241022",
    "claude-haiku-4",
    
    # Sonnet  
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-4-sonnet-20241022",
    "claude-sonnet-4-20241022",
    "claude-sonnet-4",
    "claude-sonnet-4-20250514",
    "claude-4-sonnet-20250514",
    
    # Opus
    "claude-3-opus-20240229",
    "claude-opus-4-20250514",
    "claude-opus-4-1-20250805",
    "claude-opus-4-1",
]

print("Testing model availability by making minimal API calls...\n")

available_models = []

for model in test_models:
    try:
        # Make a minimal API call
        response = client.messages.create(
            model=model,
            max_tokens=1,
            messages=[{"role": "user", "content": "Hi"}]
        )
        print(f"✓ {model} - AVAILABLE")
        available_models.append(model)
    except Exception as e:
        error_msg = str(e)
        if "model" in error_msg.lower():
            print(f"✗ {model} - NOT FOUND")
        else:
            print(f"? {model} - ERROR: {error_msg[:50]}...")

print(f"\nFound {len(available_models)} available models:")
for model in available_models:
    print(f"  - {model}")