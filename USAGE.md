# Claude Conversation Analyzer - Usage Guide

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API key:**
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"
   # Or create a .env file
   ```

3. **Run basic analysis:**
   ```bash
   python claude_conversation_analyzer.py \
     --conversations-dir /path/to/conversations \
     --claude-md /path/to/CLAUDE.md
   ```

## Common Use Cases

### Analyze Personal Conversations
```bash
# Analyze using defaults (looks in ~/.claude/projects)
python claude_conversation_analyzer.py

# Specify custom conversation directory
python claude_conversation_analyzer.py \
  --conversations-dir ~/my-projects

# Reprocess all conversations
python claude_conversation_analyzer.py \
  --conversations-dir ~/.claude/conversations \
  --all
```

### Run Analysis
```bash
# Automatically analyzes all problematic conversations
python claude_conversation_analyzer.py \
  --conversations-dir ./conversations \
  --output-dir ./analysis_results
```

### Team Analysis
```bash
# Analyze team conversations with existing standards
python claude_conversation_analyzer.py \
  --config team_config.yaml
```

### Generate New CLAUDE.md
```bash
# Automatically creates CLAUDE.md if none exists
python claude_conversation_analyzer.py \
  --conversations-dir ./conversations
```

## Configuration File

Create `analyzer_config.yaml`:

```yaml
conversations_dir: "./my_conversations"
claude_md_path: "./CLAUDE.md"
output_dir: "./results"

analysis:
  depth: "deep"
  sample_size: 300
  focus_areas:
    - interventions
    - errors
```

Then run:
```bash
python claude_conversation_analyzer.py --config analyzer_config.yaml
```

## Understanding the Output

### Files Generated

1. **summary_report.md** - Executive summary with key metrics
2. **detailed_analysis.md** - In-depth findings
3. **intervention_analysis.md** - Analysis of user corrections
4. **claude_md_recommendations.md** - Specific improvements for CLAUDE.md
5. **dashboard.html** - Interactive visualization
6. **data/analysis_results.json** - Raw data for further analysis

### Key Metrics

- **Success Rate**: Percentage of tasks completed successfully
- **Intervention Rate**: How often users need to stop/correct Claude
- **Common Issues**: Most frequent problems encountered
- **Effective Patterns**: What approaches work well

## Advanced Options

### Updating Original CLAUDE.md

When generating a new CLAUDE.md, you can automatically update the original file:

```bash
# Creates backup and updates ~/.claude/CLAUDE.md
python claude_conversation_analyzer.py --update-original

# Updates custom location with backup
python claude_conversation_analyzer.py \
  --claude-md ./my-project/CLAUDE.md \
  --update-original
```

Backups are created with timestamps: `CLAUDE_backup_20240115_143022.md`

### Model Configuration

The analyzer always performs deep analysis using three phases:
- **Phase 1**: Classification of all conversations (default: Haiku)
- **Phase 2**: Deep dive on interventions (default: Sonnet)
- **Phase 3**: Pattern synthesis (default: Opus)

You can override models in your `.env` file:
```
CLAUDE_CLASSIFIER_MODEL=claude-3-haiku-20240307
CLAUDE_ANALYZER_MODEL=claude-3-5-sonnet-20241022
CLAUDE_SYNTHESIZER_MODEL=claude-3-opus-20240229
```

### Automatic Problem Detection

The analyzer automatically focuses on problematic conversations including:
- **Interventions**: When users stop or redirect Claude
- **Failed tasks**: Tasks that weren't completed  
- **Partial success**: Tasks only partially completed
- **Frustrated tone**: Conversations with user frustration
- **Technical errors**: Conversations with errors

### Custom Patterns

Add custom patterns to detect in `analyzer_config.yaml`:

```yaml
patterns:
  custom_interventions:
    - pattern: "please stop"
      category: "hard_stop"
      severity: "high"
```

## Tips for Best Results

1. **More conversations = better insights**: Aim for at least 50-100 conversations
2. **Include diverse tasks**: Mix of bug fixes, features, refactoring
3. **Include both successes and failures**: Learn from what works and what doesn't
4. **Review intervention analysis**: This shows clearest misalignments

## Interpreting Results

### High Intervention Rate?
- Check intervention_analysis.md for patterns
- Look for specific prevention rules suggested
- Update CLAUDE.md with recommended rules

### Low Success Rate?
- Review detailed_analysis.md for systemic issues
- Check if certain task types consistently fail
- Consider adding more specific guidance

### Common Patterns?
- Look for repeated issues across conversations
- These make the best CLAUDE.md rules
- Prioritize rules that would prevent multiple issues

## Privacy & Security

- All analysis is done via API calls to Claude
- Conversations are not stored or logged by the tool
- Sensitive data can be redacted via configuration
- Run locally for maximum privacy

## Troubleshooting

### "API key not found"
```bash
export ANTHROPIC_API_KEY="your-key"
# Or add to .env file
```

### "No conversations found"
- Check conversations directory path
- Ensure files are .jsonl format
- Check for nested directories

### "Analysis taking too long"
- Use cheaper models in .env (e.g., Haiku for all phases)
- Run incrementally (default) instead of `--all`
- The analyzer only processes problematic conversations

### Out of memory
- Process fewer conversations at once
- Split large conversation directories
- Use configuration file to limit conversation length