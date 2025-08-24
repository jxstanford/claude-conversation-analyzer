"""Cost estimation for Claude API usage."""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tiktoken
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .conversation_scanner import ConversationFile, Message

logger = logging.getLogger(__name__)
console = Console()


@dataclass
class ModelPricing:
    """Pricing for a specific Claude model."""
    name: str
    input_cost_per_million: float  # USD per million tokens
    output_cost_per_million: float  # USD per million tokens
    context_window: int


# Claude model pricing as of January 2025
CLAUDE_PRICING = {
    "claude-3-5-haiku-20241022": ModelPricing(
        name="Claude 3.5 Haiku",
        input_cost_per_million=1.00,
        output_cost_per_million=5.00,
        context_window=200000
    ),
    "claude-3-5-sonnet-20241022": ModelPricing(
        name="Claude 3.5 Sonnet",
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        context_window=200000
    ),
    "claude-3-opus-20240229": ModelPricing(
        name="Claude 3 Opus",
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
        context_window=200000
    ),
    # Latest models
    "claude-3-5-haiku-latest": ModelPricing(
        name="Claude 3.5 Haiku",
        input_cost_per_million=1.00,
        output_cost_per_million=5.00,
        context_window=200000
    ),
    "claude-3-5-sonnet-latest": ModelPricing(
        name="Claude 3.5 Sonnet",
        input_cost_per_million=3.00,
        output_cost_per_million=15.00,
        context_window=200000
    ),
    "claude-3-opus-latest": ModelPricing(
        name="Claude 3 Opus",
        input_cost_per_million=15.00,
        output_cost_per_million=75.00,
        context_window=200000
    ),
}


class CostEstimator:
    """Estimates API costs for conversation analysis."""
    
    def __init__(self, models: Optional[Dict[str, str]] = None):
        """Initialize cost estimator.
        
        Args:
            models: Dictionary mapping role to model ID
                   (classifier, analyzer, synthesizer)
        """
        # Default models
        default_models = {
            'classifier': 'claude-3-5-haiku-latest',
            'analyzer': 'claude-3-5-sonnet-latest',
            'synthesizer': 'claude-3-opus-latest'
        }
        
        self.models = models or default_models
        
        # Use cl100k_base encoding as approximation for Claude
        # Claude's actual tokenizer may differ slightly
        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except:
            logger.warning("tiktoken not available, using character-based estimation")
            self.encoder = None
    
    def count_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        if self.encoder:
            return len(self.encoder.encode(text))
        else:
            # Rough approximation: 1 token ≈ 4 characters
            return len(text) // 4
    
    def estimate_conversation_tokens(self, messages: List[Message]) -> int:
        """Estimate total tokens in a conversation."""
        total = 0
        for msg in messages:
            # Add role tokens
            total += 10  # Approximate overhead for role/metadata
            # Add content tokens
            total += self.count_tokens(msg.content)
        return total
    
    def estimate_phase_costs(self, 
                           conversations: List[Tuple[ConversationFile, List[Message]]],
                           claude_md_content: Optional[str] = None) -> Dict[str, Dict]:
        """Estimate costs for each analysis phase.
        
        Returns:
            Dictionary with phase-specific cost estimates
        """
        estimates = {}
        
        # Count conversations and messages
        total_conversations = len(conversations)
        total_messages = sum(len(msgs) for _, msgs in conversations)
        total_tokens = sum(self.estimate_conversation_tokens(msgs) 
                          for _, msgs in conversations)
        
        # Phase 1: Classification (Haiku) - Quick scan of all conversations
        classifier_model = self.models.get('classifier', 'claude-3-5-haiku-latest')
        classifier_pricing = CLAUDE_PRICING.get(classifier_model, 
                                               CLAUDE_PRICING['claude-3-5-haiku-latest'])
        
        # Estimate: ~500 tokens input per conversation (summary), ~100 tokens output
        phase1_input_tokens = total_conversations * 500
        phase1_output_tokens = total_conversations * 100
        phase1_cost = self.calculate_cost(phase1_input_tokens, phase1_output_tokens, 
                                         classifier_pricing)
        
        estimates['classification'] = {
            'model': classifier_pricing.name,
            'conversations': total_conversations,
            'input_tokens': phase1_input_tokens,
            'output_tokens': phase1_output_tokens,
            'cost_usd': phase1_cost,
            'description': 'Quick classification of all conversations'
        }
        
        # Phase 2: Intervention Analysis (Sonnet) - Deep dive on ~30% of conversations
        analyzer_model = self.models.get('analyzer', 'claude-3-5-sonnet-latest')
        analyzer_pricing = CLAUDE_PRICING.get(analyzer_model,
                                             CLAUDE_PRICING['claude-3-5-sonnet-latest'])
        
        intervention_conversations = int(total_conversations * 0.3)  # Estimate 30% have interventions
        
        # Estimate: ~2000 tokens input per conversation, ~500 tokens output
        phase2_input_tokens = intervention_conversations * 2000
        phase2_output_tokens = intervention_conversations * 500
        phase2_cost = self.calculate_cost(phase2_input_tokens, phase2_output_tokens,
                                         analyzer_pricing)
        
        estimates['intervention_analysis'] = {
            'model': analyzer_pricing.name,
            'conversations': intervention_conversations,
            'input_tokens': phase2_input_tokens,
            'output_tokens': phase2_output_tokens,
            'cost_usd': phase2_cost,
            'description': 'Deep analysis of conversations with interventions'
        }
        
        # Phase 2.5: Quality Classification (Haiku) - For filtering interventions
        # Estimate ~5 interventions per problematic conversation
        estimated_interventions = intervention_conversations * 5
        quality_input_tokens = estimated_interventions * 200  # Per intervention
        quality_output_tokens = estimated_interventions * 50
        quality_cost = self.calculate_cost(quality_input_tokens, quality_output_tokens,
                                          classifier_pricing)
        
        estimates['quality_classification'] = {
            'model': classifier_pricing.name,
            'interventions': estimated_interventions,
            'input_tokens': quality_input_tokens,
            'output_tokens': quality_output_tokens,
            'cost_usd': quality_cost,
            'description': 'Quality scoring of interventions'
        }
        
        # Phase 3: Deep Analysis (Opus) - Top 10% most problematic
        deep_conversations = max(5, int(intervention_conversations * 0.33)) if intervention_conversations > 0 else 0  # ~10% of total
        
        synthesizer_model = self.models.get('synthesizer', 'claude-3-opus-latest')
        synthesizer_pricing = CLAUDE_PRICING.get(synthesizer_model,
                                                CLAUDE_PRICING['claude-3-opus-latest'])
        
        # Estimate: Full conversation content + context
        phase3_input_tokens = deep_conversations * 5000  # Full conversations
        phase3_output_tokens = deep_conversations * 1000  # Detailed analysis
        phase3_cost = self.calculate_cost(phase3_input_tokens, phase3_output_tokens,
                                         synthesizer_pricing)
        
        estimates['deep_analysis'] = {
            'model': synthesizer_pricing.name,
            'conversations': deep_conversations,
            'input_tokens': phase3_input_tokens,
            'output_tokens': phase3_output_tokens,
            'cost_usd': phase3_cost,
            'description': 'Deep analysis of most problematic conversations'
        }
        
        # Phase 4: CLAUDE.md Synthesis (Opus) - If CLAUDE.md exists
        if claude_md_content:
            claude_md_tokens = self.count_tokens(claude_md_content)
            # Add summaries and patterns from analysis
            synthesis_input_tokens = claude_md_tokens + 10000  # CLAUDE.md + analysis summaries
            synthesis_output_tokens = 3000  # New recommendations
            synthesis_cost = self.calculate_cost(synthesis_input_tokens, synthesis_output_tokens,
                                                synthesizer_pricing)
            
            estimates['claude_md_synthesis'] = {
                'model': synthesizer_pricing.name,
                'claude_md_tokens': claude_md_tokens,
                'input_tokens': synthesis_input_tokens,
                'output_tokens': synthesis_output_tokens,
                'cost_usd': synthesis_cost,
                'description': 'Synthesis of CLAUDE.md improvements'
            }
        
        # Calculate totals
        total_input_tokens = sum(phase['input_tokens'] for phase in estimates.values())
        total_output_tokens = sum(phase['output_tokens'] for phase in estimates.values())
        total_cost = sum(phase['cost_usd'] for phase in estimates.values())
        
        estimates['total'] = {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_input_tokens + total_output_tokens,
            'cost_usd': total_cost,
            'conversations_analyzed': total_conversations,
            'messages_processed': total_messages
        }
        
        return estimates
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, 
                      pricing: ModelPricing) -> float:
        """Calculate cost in USD for given token counts."""
        input_cost = (input_tokens / 1_000_000) * pricing.input_cost_per_million
        output_cost = (output_tokens / 1_000_000) * pricing.output_cost_per_million
        return input_cost + output_cost
    
    def display_cost_estimate(self, estimates: Dict[str, Dict], 
                            show_details: bool = True) -> None:
        """Display cost estimates in a formatted table."""
        
        # Create summary panel
        total = estimates['total']
        summary = Panel(
            f"[bold]Estimated Total Cost: [green]${total['cost_usd']:.2f}[/green][/bold]\n"
            f"Conversations: {total['conversations_analyzed']:,}\n"
            f"Messages: {total['messages_processed']:,}\n"
            f"Total Tokens: {total['total_tokens']:,}",
            title="[bold blue]Cost Estimate Summary[/bold blue]",
            border_style="blue"
        )
        console.print(summary)
        
        if show_details:
            # Create detailed breakdown table
            table = Table(title="\n[bold]Detailed Cost Breakdown by Phase[/bold]")
            table.add_column("Phase", style="cyan", no_wrap=True)
            table.add_column("Model", style="magenta")
            table.add_column("Items", justify="right")
            table.add_column("Input Tokens", justify="right")
            table.add_column("Output Tokens", justify="right")
            table.add_column("Cost (USD)", justify="right", style="green")
            
            for phase_name, phase_data in estimates.items():
                if phase_name == 'total':
                    continue
                
                items = phase_data.get('conversations', 
                                      phase_data.get('interventions', '-'))
                
                table.add_row(
                    phase_name.replace('_', ' ').title(),
                    phase_data.get('model', '-'),
                    str(items),
                    f"{phase_data['input_tokens']:,}",
                    f"{phase_data['output_tokens']:,}",
                    f"${phase_data['cost_usd']:.3f}"
                )
            
            # Add total row
            table.add_row(
                "[bold]TOTAL[/bold]",
                "-",
                str(total['conversations_analyzed']),
                f"[bold]{total['input_tokens']:,}[/bold]",
                f"[bold]{total['output_tokens']:,}[/bold]",
                f"[bold green]${total['cost_usd']:.2f}[/bold green]",
                style="bold"
            )
            
            console.print(table)
            
            # Add notes
            console.print("\n[dim]Notes:[/dim]")
            console.print("[dim]• Estimates assume ~30% of conversations have interventions[/dim]")
            console.print("[dim]• Deep analysis performed on ~10% most problematic conversations[/dim]")
            console.print("[dim]• Actual costs may vary based on conversation length and complexity[/dim]")
            console.print("[dim]• Token counts are estimates using tiktoken (cl100k_base encoding)[/dim]")
    
    def get_cost_breakdown_string(self, estimates: Dict[str, Dict]) -> str:
        """Get cost breakdown as a formatted string."""
        total = estimates['total']
        
        lines = [
            "=" * 50,
            "COST ESTIMATE",
            "=" * 50,
            f"Total Estimated Cost: ${total['cost_usd']:.2f}",
            f"Conversations to Analyze: {total['conversations_analyzed']}",
            f"Total Messages: {total['messages_processed']}",
            f"Estimated Tokens: {total['total_tokens']:,}",
            "",
            "Breakdown by Phase:",
        ]
        
        for phase_name, phase_data in estimates.items():
            if phase_name == 'total':
                continue
            lines.append(f"  {phase_name.replace('_', ' ').title()}: ${phase_data['cost_usd']:.3f}")
            lines.append(f"    - {phase_data['description']}")
        
        lines.extend([
            "",
            "Model Configuration:",
            f"  Classifier: {self.models.get('classifier', 'default')}",
            f"  Analyzer: {self.models.get('analyzer', 'default')}",
            f"  Synthesizer: {self.models.get('synthesizer', 'default')}",
            "=" * 50
        ])
        
        return "\n".join(lines)