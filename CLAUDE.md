# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Claude Conversation Analyzer is a Python tool that analyzes Claude Code conversations to identify patterns, detect user interventions, and generate insights for improving CLAUDE.md files. It uses a multi-tier analysis approach with Claude Haiku, Sonnet, and Opus models.

## Development Commands

### Setup and Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Set required environment variable
export ANTHROPIC_API_KEY="your-key-here"
```

### Running the Analyzer
```bash
# Basic usage (uses defaults: ~/.claude/projects and ~/.claude/CLAUDE.md)
python claude_conversation_analyzer.py
# Creates output in: ./analysis_results/YYYY-MM-DD_HH-MM-SS/

# Estimate costs before running analysis
python claude_conversation_analyzer.py --estimate-cost

# Analyze with custom paths
python claude_conversation_analyzer.py \
  --conversations-dir ./conversations \
  --claude-md ./CLAUDE.md

# Resume the most recent interrupted analysis
python claude_conversation_analyzer.py --resume

# Resume a specific interrupted analysis
python claude_conversation_analyzer.py --resume 2024-11-15_14-30-45

# Run quick test to verify setup
python test_analyzer.py
```

### Code Quality Tools
```bash
# Format code with black (configured for 100 char lines)
black src/ *.py

# Lint with ruff (configured in pyproject.toml)
ruff check src/ *.py

# Run tests (when implemented)
pytest tests/
```

## Architecture Overview

### Multi-Phase Analysis Pipeline
1. **Phase 1 - Classification (Haiku)**: Scans conversations to identify those with user interventions
2. **Phase 2 - Intervention Analysis (Sonnet)**: Analyzes problematic conversations for specific patterns
3. **Phase 3 - Deep Analysis (Opus)**: Deep analysis of conversation patterns and issues
4. **Phase 4 - CLAUDE.md Synthesis (Opus)**: LLM-based analysis and natural integration of improvements

### Key Components

**Core Modules**:
- `claude_conversation_analyzer.py`: Main CLI entry point with run management and resumable analysis
- `src/conversation_scanner.py`: Scans and loads conversation JSON files from Claude Code
- `src/analyzers/llm_analyzer.py`: Orchestrates multi-phase LLM analysis with quality-based prioritization
- `src/processing_tracker.py`: Tracks processed conversations per run for incremental analysis

**Analysis Components**:
- `src/detectors/pattern_detectors.py`: Advanced intervention detection with quality scoring
- `src/claude_md_parser.py`: Parses CLAUDE.md for validation (synthesis now uses LLM)
- `src/generators/report_generator.py`: Generates reports with LLM-integrated CLAUDE.md improvements

### Data Flow
1. Scanner loads conversation JSON files from run-specific directory
2. Detector filters low-quality interventions and scores teaching value
3. Analyzer prioritizes high-value interventions for analysis
4. LLM synthesizes natural CLAUDE.md improvements preserving style
5. Generator creates run-specific reports and proposed changes

## Configuration

The analyzer uses a hierarchical configuration system:
1. Command-line arguments (highest priority)
2. Environment variables
3. YAML config file (`analyzer_config.yaml`)
4. Default values

Key environment variables:
- `ANTHROPIC_API_KEY` (required)
- `CLAUDE_CLASSIFIER_MODEL` (default: claude-3-5-haiku-20241022)
- `CLAUDE_ANALYZER_MODEL` (default: claude-sonnet-4-20250514)
- `CLAUDE_SYNTHESIZER_MODEL` (default: claude-opus-4-1-20250805)
- `LOG_LEVEL` (default: INFO)

## Key Features

### Intelligent Intervention Filtering
The analyzer now intelligently filters and prioritizes interventions:
- **Pre-filtering**: Removes system-generated messages, truncated content, and malformed data
- **Quality Scoring**: Uses LLM classifier to identify high-value teaching moments
- **Smart Prioritization**: Focuses on corrections and guidance over operational interruptions
- **Efficiency**: Reduces API costs by skipping low-value interventions

### LLM-Based CLAUDE.md Synthesis
Instead of mechanical rule extraction, the analyzer now:
- **Analyzes Style**: Understands the existing document's voice, tone, and structure
- **Natural Integration**: Creates improvements that blend seamlessly with existing content
- **Context Awareness**: Recognizes implicit rules and relationships
- **Placement Guidance**: Suggests where new content fits best in the document

### Run Management
Each analysis creates a timestamped run directory:
- **Run Format**: `./analysis_results/YYYY-MM-DD_HH-MM-SS/`
- **Resume Support**: Use `--resume` (most recent) or `--resume <run-id>` (specific)
- **Run Metadata**: Tracks parameters, timestamps, and results
- **Isolated State**: Each run maintains its own processing state

### Cost Estimation
Before running analysis, estimate API costs:
- **Usage**: Run with `--estimate-cost` flag for dry run
- **Token Counting**: Uses tiktoken for accurate token estimation
- **Phase Breakdown**: Shows costs for each analysis phase
- **Model-Aware**: Accounts for different pricing across Haiku/Sonnet/Opus
- **Assumptions**: 
  - ~30% of conversations have interventions
  - ~10% require deep analysis
  - ~5 interventions per problematic conversation

## Common Development Tasks

### Adding New Detectors
Pattern detectors in `src/detectors/pattern_detectors.py` now support:
- `is_obviously_low_quality()`: Quick filtering without LLM calls
- `score_intervention_quality()`: Quality scoring (currently unused, LLM classifier preferred)
- System message pattern filtering

### Modifying Analysis Pipeline
The pipeline in `src/analyzers/llm_analyzer.py` includes:
- Configurable prompts for each phase
- Quality-based intervention selection
- LLM synthesis with style preservation

### Customizing CLAUDE.md Generation
The synthesis process can be tuned by modifying:
- `synthesize_claude_md` prompt: Controls how the LLM analyzes and improves CLAUDE.md
- `_generate_integrated_claude_md()`: Formats the final output using synthesis results
- Integration guidance: Helps maintain document consistency