#!/usr/bin/env python3
"""Main CLI for Claude Conversation Analyzer."""

import click
import logging
import asyncio
from pathlib import Path
import yaml
import os
from typing import Optional, List
from datetime import datetime
import shutil
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.logging import RichHandler
from dotenv import load_dotenv

from src.conversation_scanner import ConversationScanner
# ClaudeMdParser no longer needed - we just pass raw CLAUDE.md content
from src.analyzers import LLMAnalyzer, AnalysisDepth
from src.generators import ReportGenerator
from src.detectors import InterventionDetector
from src.processing_tracker import ProcessingTracker
from src.cost_estimator import CostEstimator

# Load environment variables
load_dotenv()

# Set up rich console for pretty output
console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)




@click.command()
@click.option(
    '--conversations-dir', '-c',
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default=os.path.expanduser('~/.claude/projects'),
    help='Directory containing Claude Code conversation files'
)
@click.option(
    '--claude-md', '-m',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='Path to existing CLAUDE.md file (optional)'
)
@click.option(
    '--output-dir', '-o',
    type=click.Path(file_okay=False, dir_okay=True),
    default='./analysis_results',
    help='Output directory for analysis results'
)
@click.option(
    '--all', '-a',
    is_flag=True,
    help='Process all conversations, including previously analyzed ones'
)
@click.option(
    '--config', '-C',
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help='Path to configuration file (YAML)'
)
@click.option(
    '--api-key',
    envvar='ANTHROPIC_API_KEY',
    help='Anthropic API key (can also use ANTHROPIC_API_KEY env var)'
)
@click.option(
    '--resume', '-R',
    type=click.STRING,
    is_flag=False,
    flag_value='LATEST',
    default=None,
    help='Resume a previous analysis run. Use alone for most recent, or provide specific run ID'
)
@click.option(
    '--estimate-cost',
    is_flag=True,
    help='Estimate API costs without running analysis'
)
def main(
    conversations_dir: str,
    claude_md: Optional[str],
    output_dir: str,
    all: bool,
    config: Optional[str],
    api_key: Optional[str],
    resume: Optional[str],
    estimate_cost: bool
):
    """Analyze Claude Code conversations to improve CLAUDE.md files."""
    
    # Set log level from environment variable if provided
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Load configuration if provided
    if config:
        console.print(f"[blue]Loading configuration from {config}...[/blue]")
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Use config values only if CLI arguments were not explicitly provided
        # Check if values are still at their defaults
        if conversations_dir == os.path.expanduser('~/.claude/projects'):
            conversations_dir = config_data.get('conversations_dir', conversations_dir)
        if claude_md is None:
            claude_md = config_data.get('claude_md_path', claude_md)
        if output_dir == './analysis_results':
            output_dir = config_data.get('output_dir', output_dir)
    
    # Validate API key (not needed for cost estimation)
    if not estimate_cost and not api_key:
        console.print("[red]Error: Anthropic API key not found![/red]")
        console.print("Please set ANTHROPIC_API_KEY environment variable or use --api-key option")
        return
    
    console.print("[bold green]Claude Conversation Analyzer[/bold green]")
    
    # Handle run ID and output directory
    if resume:
        # Resuming a previous run
        if resume == 'LATEST' or resume == '':
            # Find the most recent run
            run_dirs = sorted([d for d in Path(output_dir).glob("20*_*-*-*") if d.is_dir()], reverse=True)
            if not run_dirs:
                console.print(f"[red]Error: No previous runs found in {output_dir}[/red]")
                return
            run_id = run_dirs[0].name
            run_output_dir = run_dirs[0]
            console.print(f"[blue]Resuming most recent run: {run_id}[/blue]")
        else:
            # Resume specific run
            run_id = resume
            run_output_dir = Path(output_dir) / run_id
            if not run_output_dir.exists():
                console.print(f"[red]Error: Cannot find previous run '{run_id}' in {output_dir}[/red]")
                console.print(f"[yellow]Available runs:[/yellow]")
                for run_dir in sorted(Path(output_dir).glob("20*_*-*-*"), reverse=True):
                    if run_dir.is_dir():
                        console.print(f"  • {run_dir.name}")
                return
            console.print(f"[blue]Resuming analysis run: {run_id}[/blue]")
    else:
        # Create new run with datetime-based ID
        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_output_dir = Path(output_dir) / run_id
        run_output_dir.mkdir(parents=True, exist_ok=True)
        console.print(f"[blue]Starting new analysis run: {run_id}[/blue]")
    
    console.print(f"Analyzing conversations in: {conversations_dir}")
    console.print(f"Output directory: {run_output_dir}")
    
    # Save run metadata
    metadata_file = run_output_dir / "run_metadata.json"
    import json
    metadata = {
        "run_id": run_id,
        "started_at": datetime.now().isoformat(),
        "conversations_dir": conversations_dir,
        "claude_md_path": claude_md,
        "resumed": resume is not None,
        "parameters": {
            "process_all": all,
            "config_file": config
        }
    }
    
    if not resume:
        # Save initial metadata for new runs
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    # Run analysis or estimate costs
    try:
        asyncio.run(analyze_conversations(
            conversations_dir=conversations_dir,
            claude_md_path=claude_md,
            output_dir=str(run_output_dir),
            process_all=all,
            api_key=api_key,
            run_id=run_id,
            estimate_only=estimate_cost
        ))
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        console.print_exception()
        raise


async def analyze_conversations(
    conversations_dir: str,
    claude_md_path: Optional[str],
    output_dir: str,
    process_all: bool,
    api_key: str,
    run_id: str,
    estimate_only: bool = False
):
    """Main analysis workflow."""
    
    # Initialize processing tracker
    state_file = Path(output_dir) / "processing_state.json"
    tracker = ProcessingTracker(state_file)
    
    # Show tracker statistics
    stats = tracker.get_statistics()
    if stats['total_conversations'] > 0:
        console.print(f"[blue]Previously seen: {stats['total_conversations']} conversations[/blue]")
        console.print(f"[blue]Fully processed: {stats['fully_processed']} conversations[/blue]")
        if stats['last_processed']:
            console.print(f"[blue]Last run: {stats['last_processed']}[/blue]")
        
        # Show phase statistics
        phase_stats = stats.get('phase_statistics', {})
        if phase_stats:
            console.print(f"[blue]Phase completion: Classification={phase_stats.get('classification', 0)}, " +
                         f"Analysis={phase_stats.get('intervention_analysis', 0)}, " +
                         f"Deep={phase_stats.get('deep_analysis', 0)}[/blue]")
    
    # Step 1: Scan conversations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        scan_task = progress.add_task("Scanning conversations...", total=None)
        
        scanner = ConversationScanner()
        all_conversation_files = scanner.scan_directory(conversations_dir)
        
        progress.update(scan_task, completed=True)
        
        console.print(f"[green]Found {len(all_conversation_files)} total conversation files[/green]")
        
        # Filter based on processing state
        conversation_files = tracker.get_unprocessed_conversations(all_conversation_files, force_all=process_all)
        
        if not conversation_files:
            console.print("[yellow]No new conversations to process. Use --all to reprocess all conversations.[/yellow]")
            return
        
        # Validate conversations
        validation_task = progress.add_task("Validating conversations...", total=None)
        validation_report = scanner.validate_conversations(conversation_files)
        progress.update(validation_task, completed=True)
        
        console.print(f"[green]Valid conversations to process: {validation_report.valid_files}[/green]")
        if validation_report.corrupted_files:
            console.print(f"[yellow]Corrupted files: {len(validation_report.corrupted_files)}[/yellow]")
        if validation_report.empty_files:
            console.print(f"[yellow]Empty files: {len(validation_report.empty_files)}[/yellow]")
    
    # Step 2: Read CLAUDE.md if provided or check default location
    claude_md_content = None
    
    # If no claude_md_path provided, check default location
    if not claude_md_path:
        default_claude_md = os.path.expanduser('~/.claude/CLAUDE.md')
        if os.path.exists(default_claude_md):
            claude_md_path = default_claude_md
            console.print(f"[blue]Found CLAUDE.md at default location: {default_claude_md}[/blue]")
    
    if claude_md_path:
        console.print(f"[blue]Reading existing CLAUDE.md...[/blue]")
        try:
            with open(claude_md_path, 'r', encoding='utf-8') as f:
                claude_md_content = f.read()
            console.print(f"[green]Loaded CLAUDE.md ({len(claude_md_content)} characters)[/green]")
        except Exception as e:
            logger.error(f"Failed to read CLAUDE.md: {e}")
            claude_md_content = None
    
    # Step 3: Load conversations
    console.print("[blue]Loading conversation messages...[/blue]")
    conversations_with_messages = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        load_task = progress.add_task(
            f"Loading {len(conversation_files)} conversations...", 
            total=len(conversation_files)
        )
        
        for conv_file in conversation_files:
            try:
                messages = scanner.load_conversation(conv_file)
                conversations_with_messages.append((conv_file, messages))
                progress.update(load_task, advance=1)
            except Exception as e:
                logger.warning(f"Failed to load {conv_file.conversation_id}: {e}")
    
    # Load model configurations from environment
    models = {}
    if os.getenv('CLAUDE_CLASSIFIER_MODEL'):
        models['classifier'] = os.getenv('CLAUDE_CLASSIFIER_MODEL')
    if os.getenv('CLAUDE_ANALYZER_MODEL'):
        models['analyzer'] = os.getenv('CLAUDE_ANALYZER_MODEL')
    if os.getenv('CLAUDE_SYNTHESIZER_MODEL'):
        models['synthesizer'] = os.getenv('CLAUDE_SYNTHESIZER_MODEL')
    
    # Step 3.5: Cost Estimation (if requested)
    if estimate_only:
        console.print("\n[bold blue]Estimating API Costs...[/bold blue]")
        
        estimator = CostEstimator(models=models if models else None)
        estimates = estimator.estimate_phase_costs(
            conversations_with_messages, 
            claude_md_content=claude_md_content
        )
        
        # Display cost estimates
        estimator.display_cost_estimate(estimates, show_details=True)
        
        # Save estimate to file
        estimate_file = Path(output_dir) / "cost_estimate.json"
        import json
        with open(estimate_file, 'w') as f:
            json.dump(estimates, f, indent=2)
        
        console.print(f"\n[green]Cost estimate saved to: {estimate_file}[/green]")
        console.print("\n[yellow]To proceed with analysis, run without --estimate-cost flag[/yellow]")
        return
    
    # Step 4: Run LLM analysis
    console.print("\n[bold blue]Running deep analysis...[/bold blue]")
    
    if models:
        logger.info(f"Using custom models: {models}")
    
    # Get batch size from environment or use default
    batch_size = os.getenv('CLAUDE_ANALYZER_BATCH_SIZE')
    analyzer = LLMAnalyzer(
        api_key=api_key, 
        models=models if models else None,
        batch_size=int(batch_size) if batch_size else None
    )
    
    with console.status("[bold green]Analyzing conversations with Claude..."):
        analysis_results = await analyzer.analyze_conversations(
            conversations=conversations_with_messages,
            claude_md_content=claude_md_content,  # Pass raw CLAUDE.md content
            depth=AnalysisDepth.DEEP,
            tracker=tracker,
            force_all=process_all
        )
    
    # Step 5: Generate reports
    console.print("\n[bold blue]Generating reports...[/bold blue]")
    
    report_generator = ReportGenerator(output_dir=output_dir)
    generated_files = report_generator.generate_all_reports(
        analysis_results=analysis_results
    )
    
    # Display results based on whether existing CLAUDE.md was found
    stats = analysis_results.get('summary_statistics', {})
    
    if not claude_md_content:
        # No existing CLAUDE.md - generated a new one
        console.print(f"[green]Generated new CLAUDE.md with insights from {stats.get('total_conversations', 0)} conversations[/green]")
    else:
        # Existing CLAUDE.md found - generated diff analysis
        console.print(f"[green]Generated CLAUDE.md improvements based on {stats.get('total_conversations', 0)} conversations[/green]")
    
    # Display results
    console.print("\n[bold green]Analysis Complete![/bold green]")
    logger.info(f"Analysis completed successfully. Results saved to {output_dir}")
    
    console.print("\n[bold]Generated Files:[/bold]")
    for file_type, file_path in generated_files.items():
        console.print(f"  • {file_type}: [blue]{file_path}[/blue]")
    
    # Display key insights
    if stats:
        console.print("\n[bold]Key Insights:[/bold]")
        console.print(f"  • Success Rate: [green]{stats.get('success_rate', 0) * 100:.1f}%[/green]")
        console.print(f"  • Conversations with Interventions: [yellow]{stats.get('conversations_with_interventions', 0)}[/yellow]")
        
        top_issues = stats.get('top_systemic_issues', [])
        if top_issues:
            console.print("\n[bold]Top Issues Found:[/bold]")
            for issue, count in top_issues[:3]:
                console.print(f"  • {issue} ([yellow]{count} occurrences[/yellow])")
        
        top_recs = stats.get('top_recommendations', [])
        if top_recs:
            console.print("\n[bold]Top Recommendations:[/bold]")
            for rec, count in top_recs[:3]:
                console.print(f"  • {rec}")
    
    console.print(f"\n[green]View the full analysis at:[/green] [blue]{Path(output_dir).absolute()}[/blue]")
    
    # Show diff commands if CLAUDE.md analysis was generated
    if claude_md_content and 'diff_analysis' in generated_files:
        console.print("\n[bold]To review CLAUDE.md changes:[/bold]")
        
        # Determine the original CLAUDE.md path
        original_path = claude_md_path or os.path.expanduser('~/.claude/CLAUDE.md')
        proposed_path = generated_files.get('proposed_claude_md', '')
        
        console.print(f"  • VS Code: [cyan]code -d {original_path} {proposed_path}[/cyan]")
        console.print(f"  • PyCharm: [cyan]pycharm diff {original_path} {proposed_path}[/cyan]")
        console.print(f"  • Vim: [cyan]vimdiff {original_path} {proposed_path}[/cyan]")
        
        if 'patch' in generated_files:
            console.print(f"  • Apply patch: [cyan]cd ~ && patch -p0 < {generated_files['patch']}[/cyan]")
    
    # Mark conversations as processed
    logger.info("Marking conversations as processed")
    successfully_analyzed = [cf for cf, _ in conversations_with_messages]
    tracker.mark_all_processed(successfully_analyzed)
    console.print(f"[green]Marked {len(successfully_analyzed)} conversations as processed[/green]")
    
    # Update run metadata with completion info
    metadata_file = Path(output_dir) / "run_metadata.json"
    import json
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"run_id": run_id}
    
    metadata["completed_at"] = datetime.now().isoformat()
    metadata["conversations_analyzed"] = len(successfully_analyzed)
    metadata["success_rate"] = stats.get('success_rate', 0) if stats else 0
    metadata["generated_files"] = generated_files
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()