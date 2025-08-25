"""Metrics collection for Claude Conversation Analyzer."""

import time
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics

logger = logging.getLogger(__name__)


@dataclass
class APICallMetrics:
    """Metrics for a single API call."""
    phase: str
    model: str
    start_time: float
    end_time: float
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    error: Optional[str] = None
    retry_count: int = 0
    
    @property
    def duration(self) -> float:
        """Duration in seconds."""
        return self.end_time - self.start_time
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.input_tokens + self.output_tokens


@dataclass
class PhaseMetrics:
    """Aggregated metrics for an analysis phase."""
    phase_name: str
    call_count: int = 0
    total_duration: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    error_count: int = 0
    retry_count: int = 0
    response_times: List[float] = field(default_factory=list)
    
    @property
    def avg_response_time(self) -> float:
        """Average response time in seconds."""
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def p95_response_time(self) -> float:
        """95th percentile response time."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        index = int(len(sorted_times) * 0.95)
        return sorted_times[min(index, len(sorted_times) - 1)]
    
    @property
    def max_response_time(self) -> float:
        """Maximum response time."""
        return max(self.response_times) if self.response_times else 0.0
    
    @property
    def min_response_time(self) -> float:
        """Minimum response time."""
        return min(self.response_times) if self.response_times else 0.0
    
    @property
    def total_tokens(self) -> int:
        """Total tokens used."""
        return self.total_input_tokens + self.total_output_tokens
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.call_count == 0:
            return 100.0
        return ((self.call_count - self.error_count) / self.call_count) * 100


class MetricsCollector:
    """Collects and aggregates metrics for analysis runs."""
    
    def __init__(self):
        self.start_time = time.time()
        self.api_calls: List[APICallMetrics] = []
        self.phase_metrics: Dict[str, PhaseMetrics] = {}
        self.conversation_count = 0
        self.intervention_count = 0
        self.high_quality_intervention_count = 0
        
    def record_api_call(self, call_metrics: APICallMetrics):
        """Record metrics for a single API call."""
        self.api_calls.append(call_metrics)
        
        # Update phase metrics
        phase = call_metrics.phase
        if phase not in self.phase_metrics:
            self.phase_metrics[phase] = PhaseMetrics(phase_name=phase)
        
        pm = self.phase_metrics[phase]
        pm.call_count += 1
        pm.total_duration += call_metrics.duration
        pm.total_input_tokens += call_metrics.input_tokens
        pm.total_output_tokens += call_metrics.output_tokens
        pm.total_cache_read_tokens += call_metrics.cache_read_tokens
        pm.total_cache_creation_tokens += call_metrics.cache_creation_tokens
        pm.response_times.append(call_metrics.duration)
        
        if call_metrics.error:
            pm.error_count += 1
        if call_metrics.retry_count > 0:
            pm.retry_count += call_metrics.retry_count
    
    def set_conversation_count(self, count: int):
        """Set total conversation count."""
        self.conversation_count = count
    
    def set_intervention_counts(self, total: int, high_quality: int):
        """Set intervention counts."""
        self.intervention_count = total
        self.high_quality_intervention_count = high_quality
    
    def get_total_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics for the entire run."""
        total_duration = time.time() - self.start_time
        
        # Aggregate across all phases
        total_api_calls = len(self.api_calls)
        total_input_tokens = sum(pm.total_input_tokens for pm in self.phase_metrics.values())
        total_output_tokens = sum(pm.total_output_tokens for pm in self.phase_metrics.values())
        total_cache_read = sum(pm.total_cache_read_tokens for pm in self.phase_metrics.values())
        total_cache_creation = sum(pm.total_cache_creation_tokens for pm in self.phase_metrics.values())
        total_errors = sum(pm.error_count for pm in self.phase_metrics.values())
        total_retries = sum(pm.retry_count for pm in self.phase_metrics.values())
        
        # Calculate rates
        conversations_per_minute = (self.conversation_count / total_duration) * 60 if total_duration > 0 else 0
        tokens_per_second = (total_input_tokens + total_output_tokens) / total_duration if total_duration > 0 else 0
        
        # Get all response times for overall statistics
        all_response_times = []
        for call in self.api_calls:
            if not call.error:
                all_response_times.append(call.duration)
        
        return {
            "run_duration_seconds": total_duration,
            "run_duration_formatted": self._format_duration(total_duration),
            "total_api_calls": total_api_calls,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_creation_tokens": total_cache_creation,
            "total_errors": total_errors,
            "total_retries": total_retries,
            "overall_success_rate": ((total_api_calls - total_errors) / total_api_calls * 100) if total_api_calls > 0 else 100.0,
            "conversations_per_minute": round(conversations_per_minute, 2),
            "tokens_per_second": round(tokens_per_second, 2),
            "avg_response_time": statistics.mean(all_response_times) if all_response_times else 0.0,
            "p95_response_time": self._calculate_percentile(all_response_times, 0.95),
            "max_response_time": max(all_response_times) if all_response_times else 0.0,
            "conversation_count": self.conversation_count,
            "intervention_count": self.intervention_count,
            "high_quality_intervention_count": self.high_quality_intervention_count,
            "phase_metrics": {
                phase: self._phase_metrics_to_dict(pm) 
                for phase, pm in self.phase_metrics.items()
            }
        }
    
    def _phase_metrics_to_dict(self, pm: PhaseMetrics) -> Dict[str, Any]:
        """Convert PhaseMetrics to dictionary."""
        return {
            "call_count": pm.call_count,
            "total_duration": pm.total_duration,
            "total_input_tokens": pm.total_input_tokens,
            "total_output_tokens": pm.total_output_tokens,
            "total_tokens": pm.total_tokens,
            "total_cache_read_tokens": pm.total_cache_read_tokens,
            "total_cache_creation_tokens": pm.total_cache_creation_tokens,
            "error_count": pm.error_count,
            "retry_count": pm.retry_count,
            "success_rate": pm.success_rate,
            "avg_response_time": round(pm.avg_response_time, 3),
            "p95_response_time": round(pm.p95_response_time, 3),
            "max_response_time": round(pm.max_response_time, 3),
            "min_response_time": round(pm.min_response_time, 3),
        }
    
    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile)
        return sorted_values[min(index, len(sorted_values) - 1)]
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def calculate_actual_costs(self, pricing: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate actual costs based on token usage and pricing."""
        costs_by_phase = {}
        total_cost = 0.0
        
        for phase_name, pm in self.phase_metrics.items():
            # Determine which model pricing to use based on phase
            if "classification" in phase_name.lower():
                model_key = "classifier"
            elif "intervention" in phase_name.lower() or "quality" in phase_name.lower():
                model_key = "analyzer"  
            elif "deep" in phase_name.lower():
                model_key = "deep_analyzer"
            elif "synthesis" in phase_name.lower():
                model_key = "synthesizer"
            else:
                model_key = "analyzer"  # default
            
            model_pricing = pricing.get(model_key, pricing.get("analyzer"))
            
            input_cost = (pm.total_input_tokens / 1_000_000) * model_pricing.input_cost_per_million
            output_cost = (pm.total_output_tokens / 1_000_000) * model_pricing.output_cost_per_million
            phase_cost = input_cost + output_cost
            
            costs_by_phase[phase_name] = {
                "input_cost": round(input_cost, 4),
                "output_cost": round(output_cost, 4),
                "total_cost": round(phase_cost, 4),
                "model": model_pricing.name
            }
            
            total_cost += phase_cost
        
        # Calculate per-unit costs
        cost_per_conversation = total_cost / self.conversation_count if self.conversation_count > 0 else 0
        cost_per_intervention = total_cost / self.intervention_count if self.intervention_count > 0 else 0
        
        return {
            "total_cost": round(total_cost, 4),
            "cost_per_conversation": round(cost_per_conversation, 4),
            "cost_per_intervention": round(cost_per_intervention, 4),
            "costs_by_phase": costs_by_phase
        }