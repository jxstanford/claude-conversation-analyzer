# Claude Conversation Analyzer - Project Summary

## What We Built

A comprehensive tool that analyzes Claude Code conversations to:
1. Identify patterns of misalignment (especially user interventions)
2. Generate insights for improving CLAUDE.md files
3. Create data-driven recommendations based on actual usage

## Key Components

### 1. **ConversationScanner** (`src/conversation_scanner.py`)
- Discovers and loads JSONL conversation files
- Validates and filters conversations
- Detects basic patterns like interventions

### 2. **ClaudeMdParser** (`src/claude_md_parser.py`)
- Parses existing CLAUDE.md files
- Extracts rules and guidelines
- Generates templates for new CLAUDE.md files

### 3. **Pattern Detectors** (`src/detectors/pattern_detectors.py`)
- **InterventionDetector**: Finds when users stop/correct Claude
- **SuccessPatternDetector**: Identifies what works well
- **ErrorPatternDetector**: Tracks common errors

### 4. **LLM Analyzer** (`src/analyzers/llm_analyzer.py`)
- Multi-tier analysis using Claude models:
  - **Haiku**: Quick classification of all conversations
  - **Sonnet**: Deep dive on interventions
  - **Opus**: Synthesis and pattern extraction
- Focuses on your key requirement: identifying misalignments

### 5. **Report Generator** (`src/generators/report_generator.py`)
- Creates multiple output formats:
  - Executive summary
  - Detailed analysis
  - Intervention analysis (your main focus)
  - CLAUDE.md recommendations
  - Interactive dashboard

### 6. **CLI Interface** (`claude_conversation_analyzer.py`)
- Easy-to-use command line interface
- Configuration file support
- Flexible analysis options

## How to Use It

### Basic Analysis
```bash
python claude_conversation_analyzer.py \
  --conversations-dir /Users/jxstanford/claude_projects_folder_copy/projects \
  --claude-md /Users/jxstanford/.claude/CLAUDE.md \
  --analysis-depth standard
```

### Focus on Interventions (Your Key Use Case)
```bash
python claude_conversation_analyzer.py \
  --conversations-dir /Users/jxstanford/claude_projects_folder_copy/projects \
  --claude-md /Users/jxstanford/.claude/CLAUDE.md \
  --focus-on interventions \
  --analysis-depth deep \
  --sample-size 300
```

### Using Configuration File
```bash
# Copy example config
cp analyzer_config.example.yaml analyzer_config.yaml
# Edit with your settings
python claude_conversation_analyzer.py --config analyzer_config.yaml
```

## Key Features for Your Use Case

### Intervention Detection
The tool specifically looks for patterns like:
- "stop", "wait", "hold on"
- "no don't", "actually", "instead"
- Tool rejections
- User takeovers

### Root Cause Analysis
For each intervention, the tool analyzes:
- What Claude was trying to do
- Why the user intervened
- What assumption Claude made
- How to prevent it in the future

### CLAUDE.md Recommendations
Based on interventions, generates specific rules like:
- "Always search before creating new files"
- "Confirm approach before implementing complex features"
- "Stay within explicitly stated scope"

## Next Steps

1. **Set API Key**:
   ```bash
   export ANTHROPIC_API_KEY="your-key"
   ```

2. **Run Test**:
   ```bash
   python test_analyzer.py
   ```

3. **Run Full Analysis**:
   ```bash
   python claude_conversation_analyzer.py \
     --conversations-dir /Users/jxstanford/claude_projects_folder_copy/projects \
     --claude-md /Users/jxstanford/.claude/CLAUDE.md \
     --focus-on interventions \
     --analysis-depth deep
   ```

4. **Review Results**:
   - Check `analysis_results/intervention_analysis.md` for patterns
   - Review `analysis_results/claude_md_recommendations.md` for improvements
   - Open `analysis_results/dashboard.html` for visualizations

## Design Decisions

1. **Multi-tier Analysis**: Uses different Claude models for efficiency
2. **Focus on Interventions**: Specifically designed to catch misalignments
3. **Generic Tool**: Can be used by anyone with Claude Code conversations
4. **Privacy First**: All processing via API, no data storage

## Potential Enhancements

1. Add more sophisticated intervention categorization
2. Track intervention resolution patterns
3. Compare before/after CLAUDE.md effectiveness
4. Add team collaboration features
5. Create intervention prevention playbook

The tool is ready to help you understand where Claude and your expectations diverge, giving you data-driven insights to improve your CLAUDE.md!