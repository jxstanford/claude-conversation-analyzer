# Claude Conversation Analyzer

A sophisticated tool that analyzes Claude Code conversations to identify user intervention patterns and generate actionable improvements for CLAUDE.md files. Through multi-tier LLM analysis, it transforms conversation pain points into behavioral frameworks that prevent future issues.

## 🌟 Key Value Proposition

**Turn your conversation history into actionable improvements.** This analyzer examines when users stop or redirect Claude, identifies root causes, and creates comprehensive behavioral frameworks that address 95%+ of intervention patterns.

## Features

- 🔍 **Multi-tier Analysis**: Uses Claude Haiku for classification, Sonnet for pattern detection, and Opus for deep synthesis
- 🎯 **Smart Intervention Detection**: Identifies and prioritizes high-value teaching moments while filtering out noise
- 📊 **Pattern Recognition**: Discovers root causes behind user corrections (scope creep, misunderstandings, incomplete tasks)
- 🧠 **Behavioral Framework Generation**: Creates comprehensive protocols that address root causes, not just symptoms
- 💰 **Cost-Effective**: ~$0.10 per conversation analyzed with intelligent filtering to minimize API costs
- 📈 **Interactive Dashboard**: Beautiful HTML reports with diffs, coverage metrics, and ROI calculations
- 🔄 **Resumable Analysis**: Interrupt and resume analysis runs without losing progress

## Quick Start

```bash
# 1. Estimate costs before running (recommended)
python claude_conversation_analyzer.py --estimate-cost

# 2. Run basic analysis (uses defaults: ~/.claude/projects and ~/.claude/CLAUDE.md)
python claude_conversation_analyzer.py
# Creates timestamped output in: ./analysis_results/YYYY-MM-DD_HH-MM-SS/

# 3. View the interactive dashboard
open analysis_results/*/dashboard.html
```

## Common Usage Patterns

```bash
# Analyze with custom paths
python claude_conversation_analyzer.py \
  --conversations-dir ./my-conversations \
  --claude-md ./my-project/CLAUDE.md

# Resume an interrupted analysis (automatically finds most recent)
python claude_conversation_analyzer.py --resume

# Resume a specific analysis run
python claude_conversation_analyzer.py --resume 2025-08-24_15-13-47

# Force reanalysis of all conversations
python claude_conversation_analyzer.py --all

# Update your CLAUDE.md directly (creates backup first)
python claude_conversation_analyzer.py --update-original
```

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

## What Gets Analyzed

The analyzer intelligently processes your conversations to find high-value improvements:

1. **Intervention Detection**: Finds moments where users stopped, corrected, or redirected Claude
2. **Pattern Classification**: Groups interventions into categories:
   - **Incomplete Task** (e.g., "fix the tests later")
   - **Scope Creep** (e.g., "while I'm here, I'll also...")
   - **Misunderstanding** (e.g., wrong interpretation of requirements)
   - **Technical Error** (e.g., using unavailable tools)
   - **Premature Action** (e.g., coding before understanding)
   - **Wrong Approach** (e.g., overengineering simple tasks)

3. **Root Cause Analysis**: Identifies why interventions happened
4. **Framework Generation**: Creates behavioral protocols that prevent future occurrences

## Understanding the Output

After analysis completes, you'll find in `analysis_results/YYYY-MM-DD_HH-MM-SS/`:

- **dashboard.html**: Interactive report with:
  - Executive summary and ROI calculations
  - Pattern distribution charts
  - Side-by-side CLAUDE.md diff view
  - Coverage metrics showing which patterns are addressed
  
- **CLAUDE.md.enhanced**: Your improved CLAUDE.md with behavioral frameworks

- **intervention_analysis.md**: Detailed breakdown of each intervention pattern

- **executive_summary.txt**: Quick overview of findings and value delivered

## Configuration

### Environment Variables

- `ANTHROPIC_API_KEY` - **Required**: Your Anthropic API key
- `CLAUDE_CLASSIFIER_MODEL` - Phase 1: Filters conversations (default: claude-3-5-haiku-20241022)
- `CLAUDE_ANALYZER_MODEL` - Phase 2: Analyzes patterns (default: claude-sonnet-4-20250514)
- `CLAUDE_SYNTHESIZER_MODEL` - Phase 3: Synthesizes improvements (default: claude-opus-4-1-20250805)
- `CLAUDE_ANALYZER_BATCH_SIZE` - Parallel processing (default: 10)
- `LOG_LEVEL` - Verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)

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

## Real-World Example

From analyzing 140 conversations with 3,405 interventions:

**Investment**: $14.11 in API costs  
**Patterns Found**: 7 major intervention categories  
**Coverage Achieved**: 95% of patterns addressed  
**Frameworks Created**: 4 behavioral protocols  

**Key Improvements Generated**:
- 🎯 **Scope Management Framework** - Prevents "while I'm here" syndrome (28 instances)
- 🗣️ **Active Understanding Protocol** - Ensures requirements are clear before coding (33 instances)  
- 🔄 **Session & Context Management** - Handles conversation drift and unclear tasks (71 instances)
- 🛤️ **Approach Validation Protocol** - Prevents overengineering (15 instances)

**Result**: 147 future interventions prevented through behavioral change vs. superficial rules.

## Tips for Best Results

1. **More Conversations = Better Insights**: Analyze at least 50+ conversations for meaningful patterns
2. **Quality Over Quantity**: The analyzer filters out low-value interventions automatically
3. **Review Before Applying**: The enhanced CLAUDE.md shows diffs - review before updating
4. **Track Improvement**: Run periodically to see if intervention rates decrease

## How It Works

The analyzer uses a sophisticated multi-phase approach:

```
Phase 1 (Haiku) → Phase 2 (Sonnet) → Phase 3 (Opus) → Phase 4 (Opus)
Classification    Pattern Detection   Deep Analysis    Synthesis
```

1. **Classification**: Quickly scans all conversations to find those with interventions
2. **Pattern Detection**: Analyzes intervention context to understand what went wrong
3. **Deep Analysis**: Examines patterns across conversations for root causes
4. **Synthesis**: Creates comprehensive behavioral frameworks that blend with your CLAUDE.md style

## Troubleshooting

**"JSON parse error during synthesis"**  
- The analyzer now handles escape characters properly. If you see this, ensure you're using the latest version.

**"Dashboard shows 0% coverage"**  
- Fixed in latest version. The analyzer now properly tracks implementation coverage.

**"Changes seem superficial"**  
- Ensure you have enough conversations (50+) and varied intervention types for meaningful patterns.

**"Analysis is taking too long"**  
- Use `--estimate-cost` first to see scope
- Interrupt with Ctrl+C and resume later with `--resume`
- Reduce batch size: `export CLAUDE_ANALYZER_BATCH_SIZE=5`

## Advanced Usage

### Custom Models
```bash
# Use faster/cheaper models for testing
export CLAUDE_CLASSIFIER_MODEL="claude-3-haiku-20240307"
export CLAUDE_ANALYZER_MODEL="claude-3-5-sonnet-20241022"
export CLAUDE_SYNTHESIZER_MODEL="claude-3-5-sonnet-20241022"
```

### Debugging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python claude_conversation_analyzer.py

# Check synthesis results
cat analysis_results/*/synthesis_result.json | jq .
```

### Integration with CI/CD
```bash
# Non-interactive mode with specific output
python claude_conversation_analyzer.py \
  --conversations-dir ./test-conversations \
  --output-dir ./ci-results \
  --estimate-cost
```

## Contributing

We welcome contributions! Key areas:
- Additional intervention pattern detectors
- Improved synthesis prompts for different CLAUDE.md styles
- Support for other conversation formats
- Performance optimizations

## License

MIT