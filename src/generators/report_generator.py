"""Report generators for analysis results."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd

from ..analyzers import (
    ConversationClassification, 
    InterventionAnalysis,
    DeepConversationAnalysis
)
# ClaudeMdStructure no longer needed - simplified to use raw content
from .modern_dashboard import generate_modern_dashboard

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports in various formats."""
    
    def __init__(self, output_dir: str = "./analysis_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.examples_dir = self.output_dir / "examples"
        self.data_dir = self.output_dir / "data"
        
        for dir_path in [self.examples_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Set up Jinja2 for markdown templates
        self.env = Environment(
            loader=FileSystemLoader(Path(__file__).parent / "templates"),
            autoescape=select_autoescape(['html', 'xml'])
        )
    
    def generate_all_reports(self, 
                           analysis_results: Dict[str, Any]) -> Dict[str, Path]:
        """Generate all report types."""
        logger.info(f"Generating reports in {self.output_dir}")
        generated_files = {}
        
        # Generate summary report
        logger.info("Generating summary report...")
        summary_path = self.generate_summary_report(analysis_results)
        generated_files['summary'] = summary_path
        logger.debug(f"Summary report saved to: {summary_path}")
        
        # Generate detailed analysis
        logger.info("Generating detailed analysis...")
        detailed_path = self.generate_detailed_analysis(analysis_results)
        generated_files['detailed_analysis'] = detailed_path
        logger.debug(f"Detailed analysis saved to: {detailed_path}")
        
        # Generate intervention analysis
        if analysis_results.get('intervention_analyses'):
            logger.info(f"Generating intervention analysis for {len(analysis_results['intervention_analyses'])} interventions...")
            intervention_path = self.generate_intervention_analysis(
                analysis_results['intervention_analyses']
            )
            generated_files['intervention_analysis'] = intervention_path
            logger.debug(f"Intervention analysis saved to: {intervention_path}")
        else:
            logger.info("No interventions found - skipping intervention analysis")
        
        # Generate CLAUDE.md from analysis insights
        # The LLM-based synthesis now handles improvements if existing CLAUDE.md was provided
        logger.info("Generating CLAUDE.md from analysis...")
        new_claude_md_path = self.output_dir / "CLAUDE.md.proposed"
        claude_md_content = self.generate_claude_md_from_analysis(analysis_results)
        new_claude_md_path.write_text(claude_md_content)
        generated_files['proposed_claude_md'] = new_claude_md_path
        
        # Generate interactive dashboard
        logger.info("Generating interactive dashboard...")
        dashboard_path = self.generate_dashboard(analysis_results)
        generated_files['dashboard'] = dashboard_path
        logger.debug(f"Dashboard saved to: {dashboard_path}")
        
        # Save raw data
        logger.info("Saving raw analysis data...")
        data_path = self.save_raw_data(analysis_results)
        generated_files['raw_data'] = data_path
        logger.debug(f"Raw data saved to: {data_path}")
        
        # Generate examples
        logger.info("Extracting conversation examples...")
        self._extract_examples(analysis_results)
        
        logger.info(f"Report generation complete: {len(generated_files)} files created")
        return generated_files
    
    def generate_summary_report(self, analysis_results: Dict[str, Any]) -> Path:
        """Generate executive summary report."""
        stats = analysis_results.get('summary_statistics', {})
        
        content = f"""# Claude Conversation Analysis - Executive Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

- **Total Conversations Analyzed:** {stats.get('total_conversations', 0)}
- **Conversations with Interventions:** {stats.get('conversations_with_interventions', 0)} ({stats.get('conversations_with_interventions', 0) / max(stats.get('total_conversations', 1), 1) * 100:.1f}%)
- **Total User Interventions:** {stats.get('total_interventions', 0)}
- **Overall Success Rate:** {stats.get('success_rate', 0) * 100:.1f}%

## Key Findings

### Task Distribution
"""
        
        # Add intent distribution
        intent_dist = stats.get('intent_distribution', {})
        if intent_dist:
            content += "\n**User Intent Distribution:**\n"
            for intent, count in sorted(intent_dist.items(), key=lambda x: x[1], reverse=True):
                percentage = count / stats.get('total_conversations', 1) * 100
                content += f"- {intent}: {count} ({percentage:.1f}%)\n"
        
        # Add complexity distribution
        complexity_dist = stats.get('complexity_distribution', {})
        if complexity_dist:
            content += "\n**Task Complexity:**\n"
            # Handle None keys by converting to string for sorting
            for complexity, count in sorted(complexity_dist.items(), key=lambda x: (x[0] is None, str(x[0]))):
                percentage = count / stats.get('total_conversations', 1) * 100
                complexity_display = complexity if complexity is not None else "Unknown"
                content += f"- {complexity_display}: {count} ({percentage:.1f}%)\n"
        
        # Add success distribution
        success_dist = stats.get('success_distribution', {})
        if success_dist:
            content += "\n### Success Levels\n"
            # Handle None keys by converting to string for sorting
            for level, count in sorted(success_dist.items(), key=lambda x: (x[0] is None, str(x[0]))):
                percentage = count / stats.get('total_conversations', 1) * 100
                level_display = level if level is not None else "Unknown"
                content += f"- {level_display}: {count} ({percentage:.1f}%)\n"
        
        # Add intervention insights
        intervention_cats = stats.get('intervention_categories', {})
        if intervention_cats:
            content += "\n### Intervention Patterns\n"
            content += "\n**Most Common Intervention Types:**\n"
            for category, count in sorted(intervention_cats.items(), key=lambda x: x[1], reverse=True)[:5]:
                content += f"- {category}: {count} occurrences\n"
        
        # Add top issues
        top_issues = stats.get('top_systemic_issues', [])
        if top_issues:
            content += "\n### Top Systemic Issues\n"
            for issue, count in top_issues[:5]:
                content += f"1. {issue} (found in {count} conversations)\n"
        
        # Add top recommendations
        top_recs = stats.get('top_recommendations', [])
        if top_recs:
            content += "\n### Top Recommendations\n"
            for rec, count in top_recs[:5]:
                content += f"1. {rec} (suggested {count} times)\n"
        
        # Save report
        report_path = self.output_dir / "summary_report.md"
        report_path.write_text(content)
        return report_path
    
    def generate_detailed_analysis(self, analysis_results: Dict[str, Any]) -> Path:
        """Generate detailed analysis report."""
        content = f"""# Claude Conversation Analysis - Detailed Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Analysis Methodology

This analysis used a multi-tier approach:
1. **Quick Classification** - All conversations classified using Claude Haiku
2. **Intervention Analysis** - Detailed analysis of user interventions using Claude Sonnet  
3. **Deep Analysis** - Comprehensive analysis of selected conversations using Claude Opus

## Detailed Findings

"""
        
        # Add classification insights
        classifications = analysis_results.get('classifications', [])
        if classifications:
            content += "### Conversation Characteristics\n\n"
            
            # Group by tone
            tone_groups = {}
            for c in classifications:
                tone = c.conversation_tone
                if tone not in tone_groups:
                    tone_groups[tone] = []
                tone_groups[tone].append(c)
            
            # Handle None keys by converting to string for sorting
            for tone, convs in sorted(tone_groups.items(), key=lambda x: (x[0] is None, str(x[0]))):
                tone_display = tone.title() if tone is not None else "Unknown"
                content += f"\n**{tone_display} Conversations ({len(convs)}):**\n"
                # Sample characteristics
                sample = convs[:3]
                for conv in sample:
                    content += f"- {conv.conversation_id[:8]}...: {conv.user_intent} task, {conv.complexity} complexity\n"
                if len(convs) > 3:
                    content += f"- ... and {len(convs) - 3} more\n"
        
        # Add intervention insights
        intervention_analyses = analysis_results.get('intervention_analyses', [])
        if intervention_analyses:
            content += "\n### Intervention Analysis\n\n"
            
            # Group by pattern category
            pattern_groups = {}
            for ia in intervention_analyses:
                cat = ia.pattern_category
                if cat not in pattern_groups:
                    pattern_groups[cat] = []
                pattern_groups[cat].append(ia)
            
            for category, analyses in sorted(pattern_groups.items(), key=lambda x: len(x[1]), reverse=True):
                content += f"\n**{category.replace('_', ' ').title()} ({len(analyses)} cases):**\n"
                
                # Show examples
                for ia in analyses[:2]:
                    content += f"\n*Example:*\n"
                    content += f"- User said: \"{ia.intervention.user_message[:100]}...\"\n"
                    content += f"- Root cause: {ia.root_cause}\n"
                    content += f"- Prevention: {ia.prevention_rule}\n"
        
        # Add deep analysis insights
        deep_analyses = analysis_results.get('deep_analyses', [])
        if deep_analyses:
            content += "\n### Deep Analysis Insights\n\n"
            
            # Aggregate findings
            all_issues = []
            all_patterns = []
            all_lessons = []
            
            for da in deep_analyses:
                all_issues.extend(da.systemic_issues)
                all_patterns.extend(da.successful_patterns)
                all_lessons.extend(da.generalizable_lessons)
            
            if all_issues:
                content += "**Systemic Issues Identified:**\n"
                issue_counts = {}
                for issue in all_issues:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                
                for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    content += f"- {issue} (found {count} times)\n"
            
            if all_patterns:
                content += "\n**Successful Patterns:**\n"
                pattern_counts = {}
                for pattern in all_patterns:
                    pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
                
                for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    content += f"- {pattern} (observed {count} times)\n"
            
            if all_lessons:
                content += "\n**Generalizable Lessons:**\n"
                # Deduplicate similar lessons
                unique_lessons = list(set(all_lessons))[:10]
                for lesson in unique_lessons:
                    content += f"- {lesson}\n"
        
        # Save report
        report_path = self.output_dir / "detailed_analysis.md"
        report_path.write_text(content)
        return report_path
    
    def generate_intervention_analysis(self, intervention_analyses: List[InterventionAnalysis]) -> Path:
        """Generate detailed intervention analysis report."""
        content = f"""# User Intervention Analysis

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Interventions Analyzed:** {len(intervention_analyses)}

## Overview

User interventions are critical moments where Claude's actions didn't align with user expectations. Understanding these moments helps improve future interactions.

## Intervention Categories

"""
        
        # Group by pattern category
        category_groups = {}
        for ia in intervention_analyses:
            cat = ia.pattern_category
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(ia)
        
        # Add category summaries
        for category, analyses in sorted(category_groups.items(), key=lambda x: len(x[1]), reverse=True):
            content += f"\n### {category.replace('_', ' ').title()} ({len(analyses)} occurrences)\n\n"
            
            # Add description
            if category == "scope_creep":
                content += "Claude attempted to do more than requested.\n\n"
            elif category == "premature_action":
                content += "Claude acted before fully understanding the requirement.\n\n"
            elif category == "wrong_approach":
                content += "Claude chose an incorrect solution strategy.\n\n"
            elif category == "misunderstanding":
                content += "Claude misunderstood the user's intent.\n\n"
            
            # Show detailed examples
            for i, ia in enumerate(analyses[:3]):
                content += f"**Example {i+1}:**\n"
                content += f"- **User intervention:** \"{ia.intervention.user_message}\"\n"
                content += f"- **What went wrong:** {ia.what_went_wrong}\n"
                content += f"- **Claude's assumption:** {ia.claude_assumption}\n"
                content += f"- **User expectation:** {ia.user_expectation}\n"
                content += f"- **Prevention rule:** {ia.prevention_rule}\n"
                content += f"- **Severity:** {ia.severity_assessment}\n\n"
        
        # Add prevention strategies
        content += "\n## Prevention Strategies\n\n"
        
        # Collect all prevention rules
        prevention_rules = {}
        for ia in intervention_analyses:
            rule = ia.prevention_rule
            if rule not in prevention_rules:
                prevention_rules[rule] = 0
            prevention_rules[rule] += 1
        
        content += "**Most Recommended Prevention Rules:**\n"
        for rule, count in sorted(prevention_rules.items(), key=lambda x: x[1], reverse=True)[:10]:
            content += f"1. {rule} (would prevent {count} interventions)\n"
        
        # Save report
        report_path = self.output_dir / "intervention_analysis.md"
        report_path.write_text(content)
        return report_path
    
    def generate_claude_md_diff_analysis(self, 
                                       analysis_results: Dict[str, Any],
                                       existing_claude_md: Optional[Any] = None) -> Optional[Path]:
        """Generate comprehensive diff analysis for CLAUDE.md."""
        synthesis = analysis_results.get('claude_md_synthesis', {})
        stats = analysis_results['summary_statistics']
        
        content = f"""# CLAUDE.md Proposed Changes

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis based on:** {stats.get('total_conversations', 0)} conversations
**Success rate:** {stats.get('success_rate', 0) * 100:.1f}%
**Conversations with interventions:** {stats.get('conversations_with_interventions', 0)}

## Summary

{synthesis.get('synthesis_summary', 'Based on the analysis, several improvements to CLAUDE.md have been identified to better guide Claude and prevent common issues.')}

## Rules to Keep ({len(synthesis.get('rules_to_keep', []))})

These rules have proven effective and should remain unchanged:

"""
        
        # Add rules to keep
        for rule_info in synthesis.get('rules_to_keep', []):
            content += f"\n✅ **{rule_info['rule']}**\n"
            content += f"   - Reason: {rule_info['reason']}\n"
            content += f"   - Evidence: Prevented issues in {rule_info.get('evidence_count', 0)} conversations\n"
        
        # Add rules to modify
        content += f"\n## Rules to Modify ({len(synthesis.get('rules_to_modify', []))})\n\n"
        content += "These rules need refinement based on observed patterns:\n"
        
        for rule_info in synthesis.get('rules_to_modify', []):
            content += f"\n📝 **Current:** {rule_info['current']}\n"
            content += f"   **Proposed:** {rule_info['proposed']}\n"
            content += f"   - Reason: {rule_info['reason']}\n"
            content += f"   - Evidence: Issues in {rule_info.get('evidence_count', 0)} conversations\n"
        
        # Add rules to remove
        if synthesis.get('rules_to_remove'):
            content += f"\n## Rules to Remove ({len(synthesis.get('rules_to_remove', []))})\n\n"
            content += "These rules are ineffective or counterproductive:\n"
            
            for rule_info in synthesis.get('rules_to_remove', []):
                content += f"\n❌ **{rule_info['rule']}**\n"
                content += f"   - Reason: {rule_info['reason']}\n"
                content += f"   - Evidence: {rule_info.get('evidence_count', 0)} conversations\n"
        
        # Add new rules
        content += f"\n## New Rules to Add ({len(synthesis.get('rules_to_add', []))})\n\n"
        content += "These new rules would prevent observed issues:\n"
        
        # Sort by priority
        new_rules = sorted(synthesis.get('rules_to_add', []), 
                          key=lambda x: {'high': 0, 'medium': 1, 'low': 2}.get(x.get('priority', 'low'), 2))
        
        for rule_info in new_rules:
            priority_emoji = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}.get(rule_info.get('priority', 'medium'), '🟡')
            content += f"\n➕ {priority_emoji} **{rule_info['rule']}**\n"
            content += f"   - Reason: {rule_info['reason']}\n"
            content += f"   - Evidence: Would prevent issues in {rule_info.get('evidence_count', 0)} conversations\n"
            content += f"   - Priority: {rule_info.get('priority', 'medium')}\n"
        
        # Add implementation guidance
        content += "\n## Implementation Guide\n\n"
        content += "1. Review each proposed change carefully\n"
        content += "2. High priority additions address the most common issues\n"
        content += "3. Test modifications in a subset of conversations first\n"
        content += "4. Consider your specific use cases when applying changes\n"
        
        # Add examples if available
        if analysis_results.get('intervention_analyses'):
            content += "\n## Supporting Evidence\n\n"
            content += "### Example Interventions That Would Be Prevented\n\n"
            
            # Show a few examples
            for i, ia in enumerate(analysis_results['intervention_analyses'][:3]):
                content += f"**Example {i+1}:** {ia.pattern_category}\n"
                content += f"- User said: \"{ia.intervention.user_message[:100]}...\"\n"
                content += f"- Prevention: {ia.prevention_rule}\n\n"
        
        # Save report
        report_path = self.output_dir / "claude_md_analysis.md"
        report_path.write_text(content)
        return report_path
    
    def generate_proposed_claude_md(self, 
                                   analysis_results: Dict[str, Any],
                                   existing_claude_md: Optional[Any] = None) -> Optional[Path]:
        """Generate a complete proposed CLAUDE.md incorporating all changes."""
        synthesis = analysis_results.get('claude_md_synthesis', {})
        
        # Start with a copy of the existing structure
        proposed_content = "# CLAUDE.md\n\n"
        proposed_content += "This file provides guidance to Claude Code when working with code in this repository.\n\n"
        
        # Keep existing sections that aren't being removed
        rules_to_remove = {r['rule'] for r in synthesis.get('rules_to_remove', [])}
        rules_to_modify = {r['current']: r['proposed'] for r in synthesis.get('rules_to_modify', [])}
        
        # Process existing sections if available
        current_section = None
        if existing_claude_md and hasattr(existing_claude_md, 'rules'):
            for rule in existing_claude_md.rules:
                # Skip removed rules
                if rule.title in rules_to_remove:
                    continue
                
                # Check if this starts a new section
                if rule.section != current_section:
                    current_section = rule.section
                    proposed_content += f"\n## {current_section}\n\n"
                
                # Apply modifications or keep as is
                if rule.title in rules_to_modify:
                    proposed_content += f"### {rule.title}\n"
                    proposed_content += f"{rules_to_modify[rule.title]}\n\n"
                else:
                    proposed_content += f"### {rule.title}\n"
                    proposed_content += f"{rule.content}\n\n"
        
        # Add new rules grouped by priority
        new_rules = synthesis.get('rules_to_add', [])
        if new_rules:
            proposed_content += "\n## New Guidelines\n\n"
            proposed_content += "*These guidelines are based on analysis of conversation patterns.*\n\n"
            
            # Group by priority
            high_priority = [r for r in new_rules if r.get('priority') == 'high']
            medium_priority = [r for r in new_rules if r.get('priority') == 'medium']
            low_priority = [r for r in new_rules if r.get('priority') == 'low']
            
            if high_priority:
                proposed_content += "### Critical Guidelines\n\n"
                for rule in high_priority:
                    proposed_content += f"**{rule['rule']}**\n\n"
            
            if medium_priority:
                proposed_content += "### Important Guidelines\n\n" 
                for rule in medium_priority:
                    proposed_content += f"{rule['rule']}\n\n"
            
            if low_priority:
                proposed_content += "### Additional Guidelines\n\n"
                for rule in low_priority:
                    proposed_content += f"{rule['rule']}\n\n"
        
        # Save the proposed file
        proposed_path = self.output_dir / "CLAUDE.md.proposed"
        proposed_path.write_text(proposed_content)
        return proposed_path
    
    def generate_claude_md_patch(self, 
                                existing_claude_md: Optional[Any] = None,
                                proposed_path: Optional[Path] = None) -> Optional[Path]:
        """Generate a unified diff patch file."""
        import subprocess
        
        # Check if we can generate a patch
        if not existing_claude_md or not proposed_path:
            logger.warning("Cannot generate patch - missing required inputs")
            return None
            
        # Get the original CLAUDE.md path
        original_path = Path(existing_claude_md.file_path) if hasattr(existing_claude_md, 'file_path') else None
        if not original_path or not original_path.exists():
            logger.warning("Cannot generate patch - original CLAUDE.md path not found")
            return None
        
        try:
            # Generate unified diff
            result = subprocess.run(
                ['diff', '-u', str(original_path), str(proposed_path)],
                capture_output=True,
                text=True
            )
            
            if result.returncode in [0, 1]:  # 0 = no diff, 1 = diff found
                patch_content = result.stdout
                if patch_content:
                    patch_path = self.output_dir / "claude_md.patch"
                    patch_path.write_text(patch_content)
                    return patch_path
            else:
                logger.error(f"Diff command failed: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to generate patch: {e}")
        
        return None
    
    def generate_dashboard(self, analysis_results: Dict[str, Any]) -> Path:
        """Generate interactive HTML dashboard."""
        # Add timestamp if not present
        if 'timestamp' not in analysis_results:
            analysis_results['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Use the modern dashboard generator
        return generate_modern_dashboard(self.output_dir, analysis_results)
    
    def save_raw_data(self, analysis_results: Dict[str, Any]) -> Path:
        """Save raw analysis data as JSON."""
        # Convert dataclasses to dicts for JSON serialization
        serializable_results = {
            'summary_statistics': analysis_results.get('summary_statistics', {}),
            'classifications': [
                {
                    'conversation_id': c.conversation_id,
                    'user_intent': c.user_intent,
                    'task_type': c.task_type,
                    'complexity': c.complexity,
                    'has_interventions': c.has_interventions,
                    'intervention_count': c.intervention_count,
                    'task_completed': c.task_completed,
                    'success_level': c.success_level,
                    'conversation_tone': c.conversation_tone,
                    'notable_features': c.notable_features
                }
                for c in analysis_results.get('classifications', [])
            ],
            'intervention_analyses': [
                {
                    'pattern_category': ia.pattern_category,
                    'severity': ia.severity_assessment,
                    'root_cause': ia.root_cause,
                    'prevention_rule': ia.prevention_rule
                }
                for ia in analysis_results.get('intervention_analyses', [])
            ],
            'deep_analyses': [
                {
                    'conversation_id': da.conversation_id,
                    'patterns_identified': da.patterns_identified,
                    'systemic_issues': da.systemic_issues,
                    'successful_patterns': da.successful_patterns,
                    'claude_md_recommendations': da.claude_md_recommendations
                }
                for da in analysis_results.get('deep_analyses', [])
            ]
        }
        
        # Save data
        data_path = self.data_dir / "analysis_results.json"
        with open(data_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        return data_path
    
    def _extract_examples(self, analysis_results: Dict[str, Any]):
        """Extract and save conversation examples."""
        # Create example directories
        intervention_dir = self.examples_dir / "interventions"
        success_dir = self.examples_dir / "successes"
        failure_dir = self.examples_dir / "failures"
        
        for dir_path in [intervention_dir, success_dir, failure_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Save intervention examples
        intervention_analyses = analysis_results.get('intervention_analyses', [])
        for i, ia in enumerate(intervention_analyses[:10]):
            example = f"""# Intervention Example {i+1}

**Type:** {ia.pattern_category}  
**Severity:** {ia.severity_assessment}

## What Happened
{ia.what_went_wrong}

## User Intervention
"{ia.intervention.user_message}"

## Root Cause
{ia.root_cause}

## Prevention
{ia.prevention_rule}
"""
            example_path = intervention_dir / f"example_{i+1}.md"
            example_path.write_text(example)
        
        logger.info(f"Extracted {min(len(intervention_analyses), 10)} intervention examples")
    
    def _generate_integrated_claude_md(self, analysis_results: Dict[str, Any]) -> str:
        """Generate CLAUDE.md using LLM synthesis for natural integration."""
        synthesis = analysis_results.get('claude_md_synthesis', {})
        stats = analysis_results.get('summary_statistics', {})
        
        # Start with a base template that respects existing style
        integration = synthesis.get('integration_guidance', {})
        style_notes = integration.get('style_notes', '')
        
        content = f"""# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Updated based on analysis of {stats.get('total_conversations', 0)} conversations*

"""
        
        # Add synthesis summary as overview
        if synthesis.get('synthesis_summary'):
            content += f"## Overview\n\n{synthesis['synthesis_summary']}\n\n"
        
        # Add new rules in appropriate sections
        rules_to_add = synthesis.get('rules_to_add', [])
        if rules_to_add:
            content += "## Key Guidelines\n\n"
            for rule in rules_to_add:
                if rule.get('priority') == 'high':
                    content += f"### {rule['rule']}\n"
                    content += f"{rule['reason']}\n\n"
        
        # Add modifications to existing rules
        rules_to_modify = synthesis.get('rules_to_modify', [])
        if rules_to_modify:
            content += "## Enhanced Guidelines\n\n"
            for mod in rules_to_modify:
                content += f"### {mod['proposed']}\n"
                content += f"*Enhanced from: {mod['current'][:50]}...*\n"
                content += f"Reason: {mod['reason']}\n\n"
        
        # Add notes about what's working well
        rules_to_keep = synthesis.get('rules_to_keep', [])
        if rules_to_keep:
            content += "## Proven Effective Practices\n\n"
            for keep in rules_to_keep[:5]:
                content += f"- {keep['rule']}\n"
        
        return content
    
    def generate_claude_md_from_analysis(self, analysis_results: Dict[str, Any]) -> str:
        """Generate a comprehensive CLAUDE.md file based on analysis insights."""
        stats = analysis_results.get('summary_statistics', {})
        synthesis = analysis_results.get('claude_md_synthesis', {})
        
        # If we have LLM synthesis with integration guidance, use it
        if synthesis and 'integration_guidance' in synthesis:
            return self._generate_integrated_claude_md(analysis_results)
        
        # Otherwise fall back to the mechanical generation
        content = f"""# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

*Generated automatically from analysis of {stats.get('total_conversations', 0)} conversations*

## Core Principles

Based on the analysis, these principles will help ensure successful interactions:

"""
        
        # Add principles based on common issues
        intervention_analyses = analysis_results.get('intervention_analyses', [])
        deep_analyses = analysis_results.get('deep_analyses', [])
        
        # Check for common patterns and add appropriate principles
        scope_issues = sum(1 for ia in intervention_analyses if ia.pattern_category == "scope_creep")
        if scope_issues > 0:
            content += """### 1. Stay Within Scope
- Do EXACTLY what was asked - nothing more, nothing less
- Avoid refactoring unrelated code
- Don't add features that weren't requested
- Don't create documentation unless asked

"""

        premature_issues = sum(1 for ia in intervention_analyses if ia.pattern_category == "premature_action")
        if premature_issues > 0:
            content += """### 2. Understand Before Acting
- Read and understand the full request before starting
- For complex tasks, confirm understanding first
- Search for existing implementations before creating new ones
- Ask for clarification when requirements are ambiguous

"""

        # Add search-first principle if relevant
        search_mentioned = any("search" in rec.lower() for da in deep_analyses for rec in da.recommendations)
        if search_mentioned or len(intervention_analyses) > 5:
            content += """### 3. Search Before Creating
Always search the codebase before implementing new functionality:
- Use Grep to search for similar code patterns
- Use Glob to find related files
- Read existing implementations thoroughly
- Only create new code if existing solutions won't work

"""

        # Add patterns section
        content += "## Common Patterns\n\n"
        
        # Extract successful patterns
        success_patterns = []
        for da in deep_analyses:
            success_patterns.extend(da.successful_patterns)
        
        if success_patterns:
            unique_patterns = list(set(success_patterns))[:10]
            for i, pattern in enumerate(unique_patterns, 1):
                content += f"{i}. {pattern}\n"
        else:
            # Default patterns based on best practices
            content += """1. Start by understanding the existing codebase structure
2. Use appropriate tools for each task (Read for files, Grep for searching, etc.)
3. Test changes incrementally
4. Keep commits small and focused
5. Follow existing code conventions and patterns
"""

        content += "\n## Anti-patterns to Avoid\n\n"
        
        # Group interventions by category for anti-patterns
        category_groups = {}
        for ia in intervention_analyses:
            cat = ia.pattern_category
            if cat not in category_groups:
                category_groups[cat] = []
            category_groups[cat].append(ia)
        
        for category, analyses in sorted(category_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            category_name = category.replace('_', ' ').title()
            content += f"### {category_name}\n"
            
            # Add description and examples
            if category == "scope_creep":
                content += "Doing more than requested. Examples:\n"
            elif category == "premature_action":
                content += "Acting before fully understanding. Examples:\n"
            elif category == "wrong_approach":
                content += "Choosing incorrect solutions. Examples:\n"
            elif category == "misunderstanding":
                content += "Misinterpreting requirements. Examples:\n"
            
            # Add specific examples
            for ia in analyses[:2]:
                content += f"- {ia.what_went_wrong}\n"
            content += f"\n**Prevention:** {analyses[0].prevention_rule}\n\n"

        # Add tool usage section
        content += """## Tool Usage Guidelines

### File Operations
- Always use Read before Edit to understand context
- Use Glob to find files by pattern
- Prefer Edit over Write for existing files

### Search Operations  
- Use Grep for searching code patterns
- Use Glob for finding files by name/extension
- Never use bash commands like find or grep

### Code Changes
- Make small, focused changes
- Test after each significant change
- Follow existing patterns in the codebase

"""

        # Add specific rules based on analysis
        content += "## Specific Rules\n\n"
        
        # Collect all prevention rules and recommendations
        all_rules = set()
        
        # From intervention analyses
        for ia in intervention_analyses:
            all_rules.add(ia.prevention_rule)
        
        # From deep analyses recommendations
        for da in deep_analyses:
            all_rules.update(da.recommendations)
        
        # Add top rules
        for i, rule in enumerate(list(all_rules)[:20], 1):
            content += f"{i}. {rule}\n"

        # Add working patterns section
        content += "\n## What Works Well\n\n"
        
        # Find conversations with high success
        classifications = analysis_results.get('classifications', [])
        successful_patterns = [
            c for c in classifications 
            if c.success_level == "complete" and not c.has_interventions
        ]
        
        if successful_patterns:
            content += "Based on successful conversations:\n\n"
            # Extract patterns from successful conversations
            task_types = {}
            for c in successful_patterns:
                task_types[c.task_type] = task_types.get(c.task_type, 0) + 1
            
            for task_type, count in sorted(task_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                content += f"- {task_type.title()} tasks: {count} successful completions\n"

        # Add footer
        content += f"""

---

*This CLAUDE.md was automatically generated based on analysis of {stats.get('total_conversations', 0)} conversations.*
*Success rate: {stats.get('success_rate', 0) * 100:.1f}% | Conversations with interventions: {stats.get('conversations_with_interventions', 0)}*
"""
        
        return content