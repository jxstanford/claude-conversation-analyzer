# Claude Conversation Analyzer

A tool to analyze Claude Code conversations and generate insights for improving CLAUDE.md files.

## Features

- 🔍 **Multi-tier Analysis**: Uses Claude Haiku, Sonnet, and Opus for different analysis depths
- 🎯 **Intervention Detection**: Identifies when users stop or redirect Claude
- 📊 **Pattern Recognition**: Finds success patterns, error patterns, and common workflows
- 📝 **CLAUDE.md Generation**: Automatically creates CLAUDE.md if none exists, or generates recommendations for existing ones
- 📈 **Comprehensive Reports**: Generates detailed analysis in multiple formats

## Quick Start

```bash
# Basic usage (uses defaults: ~/.claude/projects and ~/.claude/CLAUDE.md)
python claude_conversation_analyzer.py

# With custom paths
python claude_conversation_analyzer.py \
  --conversations-dir ./conversations \
  --claude-md ./CLAUDE.md

# Analyze without existing CLAUDE.md (will generate one)
python claude_conversation_analyzer.py \
  --conversations-dir ./conversations

# Analyze conversations (automatically focuses on problematic ones)
python claude_conversation_analyzer.py \
  --conversations-dir ./conversations

# Reprocess all conversations (ignore previous processing)
python claude_conversation_analyzer.py \
  --conversations-dir ./conversations \
  --all

# Update original CLAUDE.md file (creates timestamped backup)
python claude_conversation_analyzer.py --update-original
```

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## Configuration

### Environment Variables

The analyzer supports the following environment variables:

- `ANTHROPIC_API_KEY` - **Required**: Your Anthropic API key
- `CLAUDE_CLASSIFIER_MODEL` - Model for phase 1 classification (default: claude-3-5-haiku-20241022)
- `CLAUDE_ANALYZER_MODEL` - Model for phase 2 intervention analysis (default: claude-sonnet-4-20250514)
- `CLAUDE_SYNTHESIZER_MODEL` - Model for phase 3 deep analysis (default: claude-opus-4-1-20250805)
- `CLAUDE_ANALYZER_BATCH_SIZE` - Number of conversations to process in parallel (default: 10)
- `LOG_LEVEL` - Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)

The analyzer supports multiple configuration methods with the following precedence (highest to lowest):

1. **Command Line Arguments** - Highest priority, always override other settings
   ```bash
   python claude_conversation_analyzer.py \
     --conversations-dir ./my-conversations \
     --claude-md ./my-claude.md \
     --output-dir ./my-results
   ```

2. **Environment Variables** - Override config files and defaults
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"  # Required
   export LOG_LEVEL="DEBUG"  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
   export CLAUDE_CLASSIFIER_MODEL="claude-3-haiku-20240307"
   export CLAUDE_ANALYZER_MODEL="claude-3-sonnet-20240229"  
   export CLAUDE_SYNTHESIZER_MODEL="claude-3-opus-20240229"
   ```

3. **YAML Configuration File** - Project-level settings
   ```bash
   python claude_conversation_analyzer.py --config analyzer_config.yaml
   ```
   
   Example `analyzer_config.yaml`:
   ```yaml
   conversations_dir: "./conversations"
   claude_md_path: "./CLAUDE.md"  # optional
   output_dir: "./analysis_results"
   
   analysis:
     # The analyzer automatically processes all problematic conversations
   ```

4. **Default Values** - Used when not specified elsewhere
   - `conversations_dir`: `~/.claude/projects`
   - `claude_md_path`: `~/.claude/CLAUDE.md` (if exists)
   - `output_dir`: `./analysis_results`

### Configuration Examples

```bash
# Example 1: CLI args override everything
# Uses: ./specific-conversations (from CLI)
# Uses: API key from environment
# Uses: ./my-results (from config file)
export ANTHROPIC_API_KEY="sk-..."
python claude_conversation_analyzer.py \
  --config analyzer_config.yaml \
  --conversations-dir ./specific-conversations

# Example 2: Environment variables for deployment
# Uses custom model configuration from environment
export ANTHROPIC_API_KEY="sk-..."
export CLAUDE_CLASSIFIER_MODEL="claude-3-haiku-20240307"
export CLAUDE_ANALYZER_MODEL="claude-3-5-sonnet-20241022"
export CLAUDE_ANALYZER_BATCH_SIZE="20"  # Increase parallel processing
python claude_conversation_analyzer.py

# Example 3: Config file for project defaults
# All settings from analyzer_config.yaml
python claude_conversation_analyzer.py --config analyzer_config.yaml
```

## Output

The analyzer generates:
- Executive summary report
- Detailed pattern analysis
- CLAUDE.md recommendations
- Interactive HTML dashboard
- Example conversations demonstrating key patterns

## License

MIT