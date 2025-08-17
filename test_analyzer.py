#!/usr/bin/env python3
"""Quick test of the analyzer with your actual conversations."""

import os
import sys
from pathlib import Path

# Test basic imports
try:
    from src.conversation_scanner import ConversationScanner
    from src.claude_md_parser import ClaudeMdParser
    from src.detectors import InterventionDetector
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_conversation_scanner():
    """Test scanning your actual conversations."""
    conversations_path = "/Users/jxstanford/claude_projects_folder_copy/projects"
    
    print(f"\nTesting ConversationScanner on: {conversations_path}")
    
    scanner = ConversationScanner()
    
    # Test scanning a smaller subset first
    test_dirs = [
        "-Users-jxstanford-devel-kz-local-install-kamiwaza",
        "-Users-jxstanford-devel-gh-repos-presentations"
    ]
    
    total_conversations = 0
    total_with_interventions = 0
    
    for test_dir in test_dirs:
        dir_path = Path(conversations_path) / test_dir
        if dir_path.exists():
            print(f"\nScanning {test_dir}...")
            try:
                conv_files = scanner.scan_directory(str(dir_path))
                print(f"  Found {len(conv_files)} conversations")
                
                # Check for interventions
                with_interventions = sum(1 for cf in conv_files if cf.has_interventions)
                print(f"  With interventions: {with_interventions}")
                
                total_conversations += len(conv_files)
                total_with_interventions += with_interventions
                
                # Load and check one conversation
                if conv_files:
                    sample = conv_files[0]
                    print(f"  Sample: {sample.conversation_id}")
                    print(f"    Messages: {sample.message_count}")
                    print(f"    Project: {sample.project_path}")
                    
            except Exception as e:
                print(f"  Error: {e}")
    
    print(f"\nTotal conversations found: {total_conversations}")
    print(f"Total with interventions: {total_with_interventions}")
    
    return total_conversations > 0

def test_claude_md_parser():
    """Test parsing CLAUDE.md."""
    claude_md_path = "/Users/jxstanford/.claude/CLAUDE.md"
    
    print(f"\nTesting ClaudeMdParser on: {claude_md_path}")
    
    parser = ClaudeMdParser()
    
    if Path(claude_md_path).exists():
        structure = parser.parse(claude_md_path)
        if structure:
            print(f"✓ Successfully parsed CLAUDE.md")
            print(f"  Rules found: {len(structure.rules)}")
            print(f"  Principles: {len(structure.principles)}")
            print(f"  Size limits: {len(structure.size_limits)}")
            
            # Show sample rule
            if structure.rules:
                sample_rule = structure.rules[0]
                print(f"\n  Sample rule:")
                print(f"    Category: {sample_rule.category.value}")
                print(f"    Title: {sample_rule.title}")
            
            return True
    else:
        print("✗ CLAUDE.md not found at expected location")
        print("\nGenerating template...")
        template = parser.generate_template()
        print(f"✓ Template generated ({len(template)} chars)")
        return True
    
    return False

def test_intervention_detector():
    """Test intervention detection on sample messages."""
    from src.conversation_scanner import Message
    
    print("\nTesting InterventionDetector...")
    
    detector = InterventionDetector()
    
    # Create sample messages with interventions
    sample_messages = [
        Message(role="user", content="Create a new auth system"),
        Message(role="assistant", content="I'll create a comprehensive authentication system..."),
        Message(role="user", content="Wait stop, don't create a new file, we already have auth"),
        Message(role="assistant", content="I apologize for the confusion..."),
    ]
    
    interventions = detector.detect_interventions(sample_messages)
    print(f"✓ Detected {len(interventions)} interventions in sample")
    
    if interventions:
        intervention = interventions[0]
        print(f"  Type: {intervention.type.value}")
        print(f"  Message: {intervention.user_message}")
        print(f"  Severity: {intervention.severity}")
    
    return True

def main():
    """Run all tests."""
    print("Claude Conversation Analyzer - Test Suite")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if api_key:
        print("✓ API key found")
    else:
        print("✗ API key not found (set ANTHROPIC_API_KEY)")
    
    # Run tests
    tests_passed = 0
    total_tests = 3
    
    if test_conversation_scanner():
        tests_passed += 1
    
    if test_claude_md_parser():
        tests_passed += 1
    
    if test_intervention_detector():
        tests_passed += 1
    
    print(f"\n{'=' * 50}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n✓ All tests passed! The analyzer is ready to use.")
        print("\nNext step: Run the full analyzer:")
        print("python claude_conversation_analyzer.py \\")
        print("  --conversations-dir /Users/jxstanford/claude_projects_folder_copy/projects \\")
        print("  --claude-md /Users/jxstanford/.claude/CLAUDE.md \\")
        print("  --analysis-depth quick")
    else:
        print("\n✗ Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()