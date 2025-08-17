"""Multi-tier LLM analysis system for conversation analysis."""

import json
import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
from anthropic import Anthropic, AsyncAnthropic
from tqdm import tqdm

from ..conversation_scanner import Message, ConversationFile
from ..claude_md_parser import ClaudeMdStructure, Rule
from ..detectors import Intervention, InterventionType

logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    """Depth of analysis to perform."""
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


@dataclass
class ConversationClassification:
    """Quick classification of a conversation."""
    conversation_id: str
    user_intent: str
    task_type: str
    complexity: str  # simple, medium, complex
    has_interventions: bool
    intervention_count: int
    task_completed: bool
    success_level: str  # complete, partial, failed
    conversation_tone: str  # smooth, bumpy, frustrated
    notable_features: List[str] = field(default_factory=list)


@dataclass
class InterventionAnalysis:
    """Deep analysis of an intervention."""
    intervention: Intervention
    root_cause: str
    what_went_wrong: str
    claude_assumption: Optional[str]
    user_expectation: str
    prevention_rule: str
    severity_assessment: str
    pattern_category: str


@dataclass
class DeepConversationAnalysis:
    """Deep analysis of a full conversation."""
    conversation_id: str
    classification: ConversationClassification
    intervention_analyses: List[InterventionAnalysis]
    systemic_issues: List[str]
    missing_claude_md_guidance: List[str]
    successful_patterns: List[str]
    generalizable_lessons: List[str]
    tool_usage_effectiveness: Dict[str, str]
    claude_md_compliance: Dict[str, bool]
    recommendations: List[str]


class LLMAnalyzer:
    """Orchestrates multi-tier LLM analysis of conversations."""
    
    def __init__(self, api_key: str, models: Optional[Dict[str, str]] = None, batch_size: Optional[int] = None):
        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        
        # Default models
        self.models = models or {
            "classifier": "claude-3-5-haiku-20241022",
            "analyzer": "claude-sonnet-4-20250514",
            "synthesizer": "claude-opus-4-1-20250805"
        }
        
        # Batch size for parallel processing (default: 10)
        self.batch_size = batch_size or int(os.getenv('CLAUDE_ANALYZER_BATCH_SIZE', '10'))
        logger.info(f"Using batch size: {self.batch_size}")
        
        # Analysis prompts
        self.prompts = self._load_prompts()
    
    def _load_prompts(self) -> Dict[str, str]:
        """Load analysis prompts."""
        return {
            "classify": """Analyze this Claude Code conversation and classify it:

User's first message: {first_user_message}
Total messages: {message_count}
Tool usage: {tools_used}
Final messages: {final_messages}

Classify:
1. user_intent: What did the user want? (bug_fix/feature_add/refactor/debug/explain/other)
2. task_type: Type of task (coding/debugging/planning/analysis/other)
3. complexity: Task complexity (simple/medium/complex)
4. has_interventions: Did user stop or redirect Claude? (true/false)
5. intervention_count: Number of times user intervened
6. task_completed: Was the task completed? (true/false)
7. success_level: How successful? (complete/partial/failed)
8. conversation_tone: Overall tone (smooth/bumpy/frustrated)
9. notable_features: List any notable patterns or issues

Return ONLY valid JSON matching this structure:
{{
    "user_intent": "...",
    "task_type": "...",
    "complexity": "...",
    "has_interventions": true/false,
    "intervention_count": 0,
    "task_completed": true/false,
    "success_level": "...",
    "conversation_tone": "...",
    "notable_features": ["..."]
}}""",

            "analyze_intervention": """Analyze this problematic conversation:

Conversation context:
{context}

Problem indicators:
- Has interventions: {has_interventions}
- Success level: {success_level}  
- Task completed: {task_completed}
- Conversation tone: {conversation_tone}
- Complexity: {complexity}

Key messages:
{key_messages}

Analyze what went wrong:
1. root_cause: Why did this conversation have problems?
2. what_went_wrong: What specific mistake or issue occurred?
3. claude_assumption: What incorrect assumption did Claude make? (if any)
4. user_expectation: What did the user actually want?
5. prevention_rule: What rule could prevent this?
6. severity_assessment: How serious was this issue? (low/medium/high)
7. pattern_category: Category (scope_creep/premature_action/wrong_approach/misunderstanding/incomplete_task/technical_error/other)

Return ONLY valid JSON:
{{
    "root_cause": "...",
    "what_went_wrong": "...",
    "claude_assumption": "...",
    "user_expectation": "...",
    "prevention_rule": "...",
    "severity_assessment": "...",
    "pattern_category": "..."
}}""",

            "analyze_multiple_interventions": """Analyze multiple interventions from this conversation:

Conversation classification:
- Intent: {user_intent}
- Complexity: {complexity}
- Success level: {success_level}
- Tone: {conversation_tone}

Interventions to analyze:
{interventions_list}

For EACH intervention listed above, analyze and return ONLY valid JSON with NO comments:
{{
    "interventions": [
        {{
            "index": 0,
            "root_cause": "Why this intervention happened",
            "what_went_wrong": "Specific mistake or issue",
            "claude_assumption": "What Claude incorrectly assumed",
            "user_expectation": "What user actually wanted",
            "prevention_rule": "Rule to prevent this",
            "severity_assessment": "low/medium/high",
            "pattern_category": "scope_creep/premature_action/wrong_approach/misunderstanding/incomplete_task/technical_error/other"
        }}
    ]
}}

Important: Return ONLY the JSON object, no markdown, no explanations.""",

            "deep_analysis": """Perform a deep analysis of this complete Claude Code conversation:

{full_conversation}

Current CLAUDE.md rules (if any):
{claude_md_rules}

Classification results:
{classification}

Intervention analyses:
{intervention_analyses}

Analyze:
1. systemic_issues: List fundamental problems in this interaction
2. missing_claude_md_guidance: What guidance would have helped?
3. successful_patterns: What worked well?
4. generalizable_lessons: Lessons that apply broadly
5. tool_usage_effectiveness: How effectively were tools used?
6. claude_md_compliance: Which rules were followed/violated?
7. recommendations: Specific improvements for CLAUDE.md

Focus on actionable insights that would prevent similar issues.

Return ONLY valid JSON:
{{
    "systemic_issues": ["..."],
    "missing_claude_md_guidance": ["..."],
    "successful_patterns": ["..."],
    "generalizable_lessons": ["..."],
    "tool_usage_effectiveness": {{"tool_name": "assessment", ...}},
    "claude_md_compliance": {{"rule": true/false, ...}},
    "recommendations": ["..."]
}}""",

            "synthesize_claude_md": """Synthesize CLAUDE.md improvements based on comprehensive analysis.

Current CLAUDE.md Rules:
{existing_rules}

Analysis Findings:
- Total conversations analyzed: {total_conversations}
- Success rate: {success_rate}%
- Common intervention patterns: {intervention_patterns}
- Top systemic issues: {top_issues}
- Top recommendations from analysis: {top_recommendations}

For each existing rule, evaluate:
1. effectiveness: Did following this rule prevent issues? (keep/modify/remove)
2. evidence: How many conversations support this decision?
3. suggested_change: If modify, what's the improved version?

Also identify:
- gaps: What new rules would prevent observed issues?
- priority: Which changes would have the most impact?

Return a comprehensive synthesis:
{{
    "rules_to_keep": [
        {{
            "rule": "existing rule text",
            "reason": "why it's effective",
            "evidence_count": number
        }}
    ],
    "rules_to_modify": [
        {{
            "current": "existing rule text",
            "proposed": "improved rule text",
            "reason": "why the change helps",
            "evidence_count": number
        }}
    ],
    "rules_to_remove": [
        {{
            "rule": "existing rule text",
            "reason": "why it's not helpful",
            "evidence_count": number
        }}
    ],
    "rules_to_add": [
        {{
            "rule": "new rule text",
            "reason": "what problem it solves",
            "evidence_count": number,
            "priority": "high/medium/low"
        }}
    ],
    "synthesis_summary": "Overall assessment of CLAUDE.md effectiveness and key improvements"
}}"""
        }
    
    async def analyze_conversations(self,
                                  conversations: List[Tuple[ConversationFile, List[Message]]],
                                  claude_md: Optional[ClaudeMdStructure] = None,
                                  depth: AnalysisDepth = AnalysisDepth.STANDARD,
                                  tracker: Optional['ProcessingTracker'] = None,
                                  force_all: bool = False) -> Dict[str, Any]:
        """Analyze conversations using multi-tier approach with state tracking."""
        results = {
            "classifications": [],
            "intervention_analyses": [],
            "deep_analyses": [],
            "summary_statistics": {}
        }
        
        # Phase 1: Quick classification of all conversations
        logger.info(f"Phase 1: Classifying conversations with {self.models['classifier']}...")
        classifications = await self._classify_all_conversations(conversations, tracker, force_all)
        results["classifications"] = classifications
        
        # Calculate classification statistics
        intervention_count = sum(1 for c in classifications if c.has_interventions)
        completed_count = sum(1 for c in classifications if c.task_completed)
        logger.info(f"Classification results: {intervention_count} with interventions, {completed_count} completed successfully")
        
        # Identify problematic conversations for deeper analysis
        problematic_conversations = []
        for (conv, msgs), classif in zip(conversations, classifications):
            # Include any conversation that wasn't fully successful
            if (classif.has_interventions or 
                classif.success_level in ["failed", "partial"] or
                classif.conversation_tone in ["frustrated", "bumpy"] or
                not classif.task_completed):
                problematic_conversations.append((conv, msgs))
        
        # Phase 2: Analyze problematic conversations
        if problematic_conversations:
            logger.info(f"Phase 2: Analyzing problematic conversations with {self.models['analyzer']}...")
            intervention_analyses = await self._analyze_interventions(
                problematic_conversations,
                classifications,
                tracker,
                force_all
            )
            results["intervention_analyses"] = intervention_analyses
            logger.info(f"Found {len(intervention_analyses)} detailed problem patterns")
        else:
            logger.info("Phase 2: No problematic conversations found to analyze")
        
        # Phase 3: Deep analysis on all problematic conversations
        if depth == AnalysisDepth.DEEP and problematic_conversations:
            logger.info(f"Phase 3: Deep analysis with {self.models['synthesizer']}...")
            deep_analyses = await self._deep_analyze_conversations(
                problematic_conversations, classifications, intervention_analyses, claude_md, tracker, force_all
            )
            results["deep_analyses"] = deep_analyses
            logger.info(f"Generated {len(deep_analyses)} deep conversation analyses")
        
        # Generate summary statistics
        results["summary_statistics"] = self._generate_summary_statistics(
            classifications, results.get("intervention_analyses", []), results.get("deep_analyses", [])
        )
        
        # Add intervention analysis metadata if we have problematic conversations
        if problematic_conversations:
            from ..detectors import InterventionDetector
            detector = InterventionDetector()
            total_interventions = sum(len(detector.detect_interventions(msgs)) for _, msgs in problematic_conversations)
            analyzed_interventions = len(results.get("intervention_analyses", []))
            if total_interventions > analyzed_interventions:
                results["summary_statistics"]["intervention_analysis_note"] = (
                    f"Analyzed {analyzed_interventions} out of {total_interventions} total interventions. "
                    f"High-severity and critical intervention types were prioritized."
                )
        
        # Phase 4: Synthesize CLAUDE.md improvements if existing CLAUDE.md provided
        if claude_md and depth == AnalysisDepth.DEEP:
            logger.info(f"Phase 4: Synthesizing CLAUDE.md improvements with {self.models['synthesizer']}...")
            synthesis = await self._synthesize_claude_md_improvements(
                claude_md, results["summary_statistics"], results["deep_analyses"]
            )
            results["claude_md_synthesis"] = synthesis
            logger.info("Generated CLAUDE.md improvement synthesis")
        
        return results
    
    async def _classify_all_conversations(self, 
                                        conversations: List[Tuple[ConversationFile, List[Message]]],
                                        tracker: Optional['ProcessingTracker'] = None,
                                        force_all: bool = False) -> List[ConversationClassification]:
        """Classify all conversations using Haiku with state tracking."""
        classifications = []
        
        # Check which conversations need classification
        conversations_to_process = []
        for conv_file, messages in conversations:
            if tracker and not force_all:
                # Check if already classified
                cached_result = tracker.get_phase_result(conv_file.conversation_id, 'classification')
                if cached_result:
                    # Convert dict back to ConversationClassification
                    classifications.append(ConversationClassification(**cached_result))
                    continue
            conversations_to_process.append((conv_file, messages))
        
        if not conversations_to_process:
            logger.info("All conversations already classified")
            return classifications
        
        logger.info(f"Classifying {len(conversations_to_process)} conversations (skipping {len(conversations) - len(conversations_to_process)} already classified)")
        
        # Process in batches for efficiency
        total_batches = (len(conversations_to_process) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(conversations_to_process), desc="Classifying conversations") as pbar:
            for i in range(0, len(conversations_to_process), self.batch_size):
                batch = conversations_to_process[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} conversations)")
                
                batch_tasks = [
                    self._classify_single_conversation(conv_file, messages)
                    for conv_file, messages in batch
                ]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result, (conv_file, _) in zip(batch_results, batch):
                    if isinstance(result, Exception):
                        logger.error(f"Classification failed for {conv_file.conversation_id}: {result}")
                        # Create a default classification
                        result = ConversationClassification(
                            conversation_id=conv_file.conversation_id,
                            user_intent="unknown",
                            task_type="unknown",
                            complexity="unknown",
                            has_interventions=conv_file.has_interventions,
                            intervention_count=0,
                            task_completed=False,
                            success_level="unknown",
                            conversation_tone="unknown"
                        )
                    
                    classifications.append(result)
                    
                    # Save to tracker if available
                    if tracker:
                        tracker.save_phase_result(
                            conv_file.conversation_id,
                            conv_file.file_path,
                            'classification',
                            asdict(result)
                        )
                
                # Update progress bar by the number of conversations processed in this batch
                pbar.update(len(batch))
        
        return classifications
    
    async def _classify_single_conversation(self,
                                          conv_file: ConversationFile,
                                          messages: List[Message]) -> ConversationClassification:
        """Classify a single conversation."""
        # Extract key information
        first_user_message = next((msg.content for msg in messages if msg.role == 'user'), "")
        if isinstance(first_user_message, dict):
            first_user_message = str(first_user_message)
        
        tools_used = list(set(
            msg.tool_use.get('name', '') 
            for msg in messages 
            if msg.tool_use and isinstance(msg.tool_use, dict)
        ))
        
        final_messages = [
            f"{msg.role}: {str(msg.content)[:100]}..." 
            for msg in messages[-3:]
        ]
        
        prompt = self.prompts["classify"].format(
            first_user_message=first_user_message[:500],
            message_count=len(messages),
            tools_used=", ".join(tools_used) if tools_used else "none",
            final_messages="\n".join(final_messages)
        )
        
        try:
            response = await self.async_client.messages.create(
                model=self.models["classifier"],
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            content = response.content[0].text
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                
                # Ensure no None values in critical fields
                return ConversationClassification(
                    conversation_id=conv_file.conversation_id,
                    user_intent=data.get('user_intent') or 'unknown',
                    task_type=data.get('task_type') or 'unknown',
                    complexity=data.get('complexity') or 'unknown',
                    has_interventions=data.get('has_interventions', False),
                    intervention_count=data.get('intervention_count', 0),
                    task_completed=data.get('task_completed', False),
                    success_level=data.get('success_level') or 'unknown',
                    conversation_tone=data.get('conversation_tone') or 'unknown',
                    notable_features=data.get('notable_features', [])
                )
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            logger.error(f"Classification error: {e}")
            raise
    
    async def _analyze_interventions(self,
                                   conversations: List[Tuple[ConversationFile, List[Message]]],
                                   classifications: List[ConversationClassification],
                                   tracker: Optional['ProcessingTracker'] = None,
                                   force_all: bool = False) -> List[InterventionAnalysis]:
        """Analyze problematic conversations using Sonnet with state tracking."""
        from ..detectors import InterventionDetector
        
        all_analyses = []
        detector = InterventionDetector()
        
        # Check which conversations need intervention analysis
        conversations_to_analyze = []
        for conv_file, messages in conversations:
            if tracker and not force_all:
                # Check if already analyzed
                cached_result = tracker.get_phase_result(conv_file.conversation_id, 'intervention_analysis')
                if cached_result:
                    # Convert cached results back to InterventionAnalysis objects
                    for analysis_data in cached_result:
                        # Reconstruct the Intervention object
                        from ..detectors import Intervention, InterventionType
                        intervention_data = analysis_data['intervention']
                        intervention = Intervention(
                            type=InterventionType[intervention_data['type']],
                            message_index=intervention_data.get('message_index', 0),
                            user_message=intervention_data['user_message'],
                            context_before=intervention_data.get('context_before'),
                            claude_action_interrupted=intervention_data.get('claude_action_interrupted'),
                            severity=intervention_data.get('severity', 'medium')
                        )
                        # Remove intervention from dict to get other fields
                        analysis_fields = {k: v for k, v in analysis_data.items() if k != 'intervention'}
                        all_analyses.append(InterventionAnalysis(intervention=intervention, **analysis_fields))
                    continue
            conversations_to_analyze.append((conv_file, messages))
        
        if not conversations_to_analyze:
            logger.info("All problematic conversations already analyzed")
            return all_analyses
        
        logger.info(f"Analyzing {len(conversations_to_analyze)} conversations (skipping {len(conversations) - len(conversations_to_analyze)} already analyzed)")
        
        # Process in batches for efficiency
        
        with tqdm(total=len(conversations_to_analyze), desc="Analyzing interventions") as pbar:
            for i in range(0, len(conversations_to_analyze), self.batch_size):
                batch = conversations_to_analyze[i:i + self.batch_size]
                
                # Prepare all analysis tasks for this batch
                batch_tasks = []
                batch_metadata = []  # Track which analyses belong to which conversation
                
                for conv_file, messages in batch:
                    # Find the classification for this conversation
                    conv_classification = next(
                        (c for c in classifications if c.conversation_id == conv_file.conversation_id),
                        None
                    )
                    if not conv_classification:
                        logger.warning(f"No classification found for {conv_file.conversation_id}")
                        continue
                    
                    # Detect interventions for this conversation
                    interventions = detector.detect_interventions(messages)
                    
                    if interventions:
                        logger.debug(f"Found {len(interventions)} interventions in {conv_file.conversation_id}")
                        # Analyze multiple interventions in one call
                        task = self._analyze_multiple_interventions(
                            interventions, messages, conv_classification
                        )
                        batch_tasks.append(task)
                        batch_metadata.append((conv_file, 'multiple_interventions'))
                    else:
                        # Create task for problematic conversation without interventions
                        task = self._analyze_problematic_conversation(
                            conv_file, messages, conv_classification
                        )
                        batch_tasks.append(task)
                        batch_metadata.append((conv_file, 'problematic'))
                
                # Execute all tasks in parallel
                if batch_tasks:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # Group results by conversation for saving
                    conv_results = {}
                    
                    for result, (conv_file, analysis_type) in zip(batch_results, batch_metadata):
                        if isinstance(result, Exception):
                            logger.error(f"Analysis error for {conv_file.conversation_id}: {result}")
                            continue
                        
                        if analysis_type == 'problematic' and result is None:
                            # Skip None results from problematic conversation analysis
                            continue
                        
                        if result:
                            # Handle multiple interventions result
                            if analysis_type == 'multiple_interventions' and isinstance(result, list):
                                all_analyses.extend(result)
                                # Group by conversation for saving
                                if conv_file.conversation_id not in conv_results:
                                    conv_results[conv_file.conversation_id] = []
                                conv_results[conv_file.conversation_id].extend(result)
                            else:
                                all_analyses.append(result)
                                # Group by conversation for saving
                                if conv_file.conversation_id not in conv_results:
                                    conv_results[conv_file.conversation_id] = []
                                conv_results[conv_file.conversation_id].append(result)
                    
                    # Save results for each conversation
                    if tracker:
                        for conv_id, analyses in conv_results.items():
                            # Find the conv_file for this ID
                            conv_file = next((cf for cf, _ in batch if cf.conversation_id == conv_id), None)
                            if conv_file:
                                # Convert to serializable format
                                serializable_analyses = []
                                for analysis in analyses:
                                    analysis_dict = asdict(analysis)
                                    # Convert intervention type enum to string
                                    if 'intervention' in analysis_dict and 'type' in analysis_dict['intervention']:
                                        analysis_dict['intervention']['type'] = analysis_dict['intervention']['type'].name
                                    serializable_analyses.append(analysis_dict)
                                
                                tracker.save_phase_result(
                                    conv_file.conversation_id,
                                    conv_file.file_path,
                                    'intervention_analysis',
                                    serializable_analyses
                                )
                
                # Update progress bar
                pbar.update(len(batch))
        
        return all_analyses
    
    async def _analyze_single_intervention(self,
                                         intervention: Intervention,
                                         messages: List[Message],
                                         classification: ConversationClassification) -> InterventionAnalysis:
        """Analyze a single intervention."""
        # Get context around the intervention
        context = f"Context before: {intervention.context_before or 'No context'}\n"
        context += f"User message: {intervention.user_message}\n"
        context += f"Claude action: {intervention.claude_action_interrupted or 'Unknown'}"
        
        # Get key messages around the intervention
        key_messages = f"User: {intervention.user_message}\n"
        if intervention.claude_action_interrupted:
            key_messages += f"Claude was attempting: {intervention.claude_action_interrupted}"
        
        prompt = self.prompts["analyze_intervention"].format(
            context=context,
            has_interventions=True,
            success_level=classification.success_level,
            task_completed=classification.task_completed,
            conversation_tone=classification.conversation_tone,
            complexity=classification.complexity,
            key_messages=key_messages
        )
        
        response = await self.async_client.messages.create(
            model=self.models["analyzer"],
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        content = response.content[0].text
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(content[json_start:json_end])
            
            return InterventionAnalysis(
                intervention=intervention,
                **data
            )
        else:
            raise ValueError("No JSON found in response")
    
    async def _analyze_problematic_conversation(self,
                                              conv_file: ConversationFile,
                                              messages: List[Message],
                                              classification: ConversationClassification) -> Optional[InterventionAnalysis]:
        """Analyze a problematic conversation without explicit interventions."""
        # Skip if conversation was actually successful
        if classification.task_completed and classification.success_level == "complete":
            return None
        
        # Create a summary of the conversation
        context = f"Task: {classification.user_intent}\n"
        context += f"Type: {classification.task_type}\n"
        context += f"Outcome: {classification.success_level} - Task completed: {classification.task_completed}"
        
        # Get first and last few messages
        key_messages = ""
        if messages:
            # First user message
            first_user_msg = next((m for m in messages if m.role == "user"), None)
            if first_user_msg:
                key_messages += f"First request: {self._truncate_message(first_user_msg.content)}\n"
            
            # Last few messages
            last_messages = messages[-3:]
            for msg in last_messages:
                key_messages += f"{msg.role}: {self._truncate_message(msg.content)}\n"
        
        prompt = self.prompts["analyze_intervention"].format(
            context=context,
            has_interventions=False,
            success_level=classification.success_level,
            task_completed=classification.task_completed,
            conversation_tone=classification.conversation_tone,
            complexity=classification.complexity,
            key_messages=key_messages
        )
        
        response = await self.async_client.messages.create(
            model=self.models["analyzer"],
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        content = response.content[0].text
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(content[json_start:json_end])
            
            # Create a pseudo-intervention for the whole conversation
            from ..detectors import Intervention, InterventionType
            pseudo_intervention = Intervention(
                type=InterventionType.CORRECTION,
                message_index=len(messages) - 1 if messages else 0,
                user_message="[Conversation ended with issues]",
                context_before=context,
                severity="medium"
            )
            
            return InterventionAnalysis(
                intervention=pseudo_intervention,
                **data
            )
        
        return None
    
    async def _analyze_multiple_interventions(self,
                                            interventions: List[Intervention],
                                            messages: List[Message],
                                            classification: ConversationClassification) -> List[InterventionAnalysis]:
        """Analyze multiple interventions in a single API call."""
        # Prioritize interventions by severity and type
        # Higher priority for interventions that indicate fundamental issues
        priority_order = {
            InterventionType.STOP_ACTION: 1,      # Urgent prevention
            InterventionType.TAKEOVER: 2,         # User taking control
            InterventionType.CORRECTION: 3,       # Fundamental misunderstanding
            InterventionType.TOOL_REJECTION: 4,   # Prevented risky action
            InterventionType.APPROACH_CHANGE: 5,  # Wrong method
            InterventionType.SCOPE_REDIRECT: 6,   # Scope creep
            InterventionType.CLARIFICATION: 7     # Minor clarification
        }
        
        # Sort interventions by priority
        # Consider severity first for truly critical issues
        sorted_interventions = sorted(
            interventions,
            key=lambda i: (
                # High severity always comes first regardless of type
                0 if i.severity == 'high' else 1,
                # Then by intervention type priority
                priority_order.get(i.type, 8),
                # Then by remaining severity levels
                0 if i.severity == 'medium' else 1,
                # Finally by order in conversation
                i.message_index
            )
        )
        
        # Take the most important interventions
        interventions_to_analyze = sorted_interventions[:10]
        
        if len(interventions) > 10:
            # Log what we're keeping and what we're skipping
            kept_types = {}
            skipped_types = {}
            
            for i in interventions_to_analyze:
                kept_types[i.type.value] = kept_types.get(i.type.value, 0) + 1
            
            for i in sorted_interventions[10:]:
                skipped_types[i.type.value] = skipped_types.get(i.type.value, 0) + 1
            
            logger.info(f"Analyzing 10 most important interventions out of {len(interventions)}")
            logger.info(f"Keeping: {kept_types}")
            logger.info(f"Skipping: {skipped_types}")
        
        # Format interventions list
        interventions_list = ""
        for i, intervention in enumerate(interventions_to_analyze):
            interventions_list += f"\nIntervention {i}:\n"
            interventions_list += f"- User message: {intervention.user_message[:200]}\n"
            if intervention.context_before:
                interventions_list += f"- Context: {intervention.context_before[:200]}\n"
            if intervention.claude_action_interrupted:
                interventions_list += f"- Claude was: {intervention.claude_action_interrupted[:200]}\n"
        
        prompt = self.prompts["analyze_multiple_interventions"].format(
            user_intent=classification.user_intent,
            complexity=classification.complexity,
            success_level=classification.success_level,
            conversation_tone=classification.conversation_tone,
            interventions_list=interventions_list
        )
        
        try:
            response = await self.async_client.messages.create(
                model=self.models["analyzer"],
                max_tokens=2000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            content = response.content[0].text
            
            # Try to extract JSON - look for the interventions array
            try:
                # First try to parse the whole response
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from the response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    # Clean up common JSON issues
                    json_str = json_str.replace('// ...', '')  # Remove comments
                    json_str = re.sub(r'//.*$', '', json_str, flags=re.MULTILINE)  # Remove all comments
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                    json_str = json_str.replace('...', '"..."')  # Fix ellipsis
                    # Remove any text after the last closing brace
                    last_brace = json_str.rfind('}')
                    if last_brace > 0:
                        json_str = json_str[:last_brace + 1]
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON parse error: {e}")
                        logger.debug(f"Failed JSON: {json_str[:500]}...")
                        raise
                else:
                    raise ValueError("No JSON found in response")
            
            # Convert each intervention analysis
            analyses = []
            for i, analysis_data in enumerate(data.get('interventions', [])):
                if i < len(interventions_to_analyze):
                    intervention = interventions_to_analyze[i]
                    analysis = InterventionAnalysis(
                        intervention=intervention,
                        root_cause=analysis_data.get('root_cause', ''),
                        what_went_wrong=analysis_data.get('what_went_wrong', ''),
                        claude_assumption=analysis_data.get('claude_assumption'),
                        user_expectation=analysis_data.get('user_expectation', ''),
                        prevention_rule=analysis_data.get('prevention_rule', ''),
                        severity_assessment=analysis_data.get('severity_assessment', 'medium'),
                        pattern_category=analysis_data.get('pattern_category', 'other')
                    )
                    analyses.append(analysis)
            
            return analyses
                
        except Exception as e:
            logger.error(f"Multi-intervention analysis error: {e}")
            # Fall back to individual analysis
            return await self._analyze_interventions_individually(interventions_to_analyze, messages, classification)
    
    async def _analyze_interventions_individually(self,
                                                 interventions: List[Intervention],
                                                 messages: List[Message],
                                                 classification: ConversationClassification) -> List[InterventionAnalysis]:
        """Fallback to analyze interventions one by one."""
        analyses = []
        for intervention in interventions[:5]:  # Limit to 5 for fallback
            try:
                analysis = await self._analyze_single_intervention(intervention, messages, classification)
                analyses.append(analysis)
            except Exception as e:
                logger.error(f"Individual intervention analysis error: {e}")
        return analyses
    
    def _truncate_message(self, content: Any, max_length: int = 200) -> str:
        """Truncate a message for display."""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
        else:
            text = str(content)
        
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    async def _deep_analyze_conversations(self,
                                        sample_conversations: List[Tuple[ConversationFile, List[Message]]],
                                        classifications: List[ConversationClassification],
                                        intervention_analyses: List[InterventionAnalysis],
                                        claude_md: Optional[ClaudeMdStructure],
                                        tracker: Optional['ProcessingTracker'] = None,
                                        force_all: bool = False) -> List[DeepConversationAnalysis]:
        """Perform deep analysis using Opus with state tracking."""
        deep_analyses = []
        
        # Check which conversations need deep analysis
        conversations_to_analyze = []
        for conv_file, messages in sample_conversations:
            if tracker and not force_all:
                # Check if already analyzed
                cached_result = tracker.get_phase_result(conv_file.conversation_id, 'deep_analysis')
                if cached_result:
                    # Reconstruct DeepConversationAnalysis from cached data
                    # First find the classification
                    classification = next(
                        (c for c in classifications if c.conversation_id == conv_file.conversation_id),
                        None
                    )
                    if classification:
                        # Find intervention analyses for this conversation
                        conv_intervention_analyses = [
                            ia for ia in intervention_analyses
                            if any(msg.content == ia.intervention.user_message for msg in messages if msg.role == 'user')
                        ]
                        
                        deep_analysis = DeepConversationAnalysis(
                            conversation_id=conv_file.conversation_id,
                            classification=classification,
                            intervention_analyses=conv_intervention_analyses,
                            **{k: v for k, v in cached_result.items() if k not in ['conversation_id', 'classification', 'intervention_analyses']}
                        )
                        deep_analyses.append(deep_analysis)
                    continue
            conversations_to_analyze.append((conv_file, messages))
        
        if not conversations_to_analyze:
            logger.info("All conversations already have deep analysis")
            return deep_analyses
        
        logger.info(f"Deep analyzing {len(conversations_to_analyze)} conversations (skipping {len(sample_conversations) - len(conversations_to_analyze)} already analyzed)")
        
        # Process in batches for efficiency
        
        with tqdm(total=len(conversations_to_analyze), desc="Deep analysis") as pbar:
            for i in range(0, len(conversations_to_analyze), self.batch_size):
                batch = conversations_to_analyze[i:i + self.batch_size]
                
                # Prepare tasks for this batch
                batch_tasks = []
                batch_conv_files = []
                
                for conv_file, messages in batch:
                    # Find relevant classification
                    classification = next(
                        (c for c in classifications if c.conversation_id == conv_file.conversation_id),
                        None
                    )
                    if not classification:
                        continue
                    
                    # Find relevant intervention analyses
                    conv_interventions = [
                        ia for ia in intervention_analyses
                        if any(msg.uuid == ia.intervention.user_message for msg in messages)
                    ]
                    
                    # Create task for deep analysis
                    task = self._deep_analyze_single_conversation(
                        conv_file, messages, classification, conv_interventions, claude_md
                    )
                    batch_tasks.append(task)
                    batch_conv_files.append(conv_file)
                
                # Execute all tasks in parallel
                if batch_tasks:
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    for result, conv_file in zip(batch_results, batch_conv_files):
                        if isinstance(result, Exception):
                            logger.error(f"Deep analysis error for {conv_file.conversation_id}: {result}")
                            continue
                        
                        deep_analyses.append(result)
                        
                        # Save to tracker if available
                        if tracker:
                            # Convert to serializable format
                            analysis_dict = {
                                'systemic_issues': result.systemic_issues,
                                'missing_claude_md_guidance': result.missing_claude_md_guidance,
                                'successful_patterns': result.successful_patterns,
                                'generalizable_lessons': result.generalizable_lessons,
                                'tool_usage_effectiveness': result.tool_usage_effectiveness,
                                'claude_md_compliance': result.claude_md_compliance,
                                'recommendations': result.recommendations
                            }
                            tracker.save_phase_result(
                                conv_file.conversation_id,
                                conv_file.file_path,
                                'deep_analysis',
                                analysis_dict
                            )
                
                # Update progress bar
                pbar.update(len(batch))
        
        return deep_analyses
    
    async def _deep_analyze_single_conversation(self,
                                              conv_file: ConversationFile,
                                              messages: List[Message],
                                              classification: ConversationClassification,
                                              intervention_analyses: List[InterventionAnalysis],
                                              claude_md: Optional[ClaudeMdStructure]) -> DeepConversationAnalysis:
        """Deep analyze a single conversation."""
        # Format conversation
        conversation_text = self._format_conversation_for_analysis(messages)
        
        # Format CLAUDE.md rules if available
        claude_md_rules = ""
        if claude_md:
            rule_summaries = [f"- {rule.title}" for rule in claude_md.rules[:10]]
            claude_md_rules = "\n".join(rule_summaries)
        
        # Format intervention analyses
        intervention_summaries = [
            f"- {ia.pattern_category}: {ia.what_went_wrong}"
            for ia in intervention_analyses
        ]
        
        prompt = self.prompts["deep_analysis"].format(
            full_conversation=conversation_text[:8000],  # Limit length
            claude_md_rules=claude_md_rules or "No CLAUDE.md provided",
            classification=json.dumps(asdict(classification), indent=2),
            intervention_analyses="\n".join(intervention_summaries) or "No interventions"
        )
        
        response = await self.async_client.messages.create(
            model=self.models["synthesizer"],
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse response
        content = response.content[0].text
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            data = json.loads(content[json_start:json_end])
            
            return DeepConversationAnalysis(
                conversation_id=conv_file.conversation_id,
                classification=classification,
                intervention_analyses=intervention_analyses,
                **data
            )
        else:
            raise ValueError("No JSON found in response")
    
    def _format_conversation_for_analysis(self, messages: List[Message]) -> str:
        """Format conversation for analysis."""
        formatted = []
        for i, msg in enumerate(messages):
            role = msg.role.upper()
            content = str(msg.content)[:200] + "..." if len(str(msg.content)) > 200 else str(msg.content)
            
            if msg.tool_use and isinstance(msg.tool_use, dict):
                tool_info = f" [Tool: {msg.tool_use.get('name', 'unknown')}]"
            else:
                tool_info = ""
            
            formatted.append(f"{i}. {role}{tool_info}: {content}")
        
        return "\n".join(formatted)
    
    def _generate_summary_statistics(self,
                                   classifications: List[ConversationClassification],
                                   intervention_analyses: List[InterventionAnalysis],
                                   deep_analyses: List[DeepConversationAnalysis]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_conversations = len(classifications)
        
        # Classification statistics
        intent_counts = {}
        complexity_counts = {}
        success_counts = {}
        tone_counts = {}
        
        for c in classifications:
            intent_counts[c.user_intent] = intent_counts.get(c.user_intent, 0) + 1
            complexity_counts[c.complexity] = complexity_counts.get(c.complexity, 0) + 1
            success_counts[c.success_level] = success_counts.get(c.success_level, 0) + 1
            tone_counts[c.conversation_tone] = tone_counts.get(c.conversation_tone, 0) + 1
        
        # Intervention statistics
        intervention_categories = {}
        intervention_severities = {}
        
        for ia in intervention_analyses:
            intervention_categories[ia.pattern_category] = intervention_categories.get(ia.pattern_category, 0) + 1
            intervention_severities[ia.severity_assessment] = intervention_severities.get(ia.severity_assessment, 0) + 1
        
        # Deep analysis insights
        common_issues = {}
        common_recommendations = {}
        
        for da in deep_analyses:
            for issue in da.systemic_issues:
                common_issues[issue] = common_issues.get(issue, 0) + 1
            for rec in da.recommendations:
                common_recommendations[rec] = common_recommendations.get(rec, 0) + 1
        
        return {
            "total_conversations": total_conversations,
            "conversations_with_interventions": sum(1 for c in classifications if c.has_interventions),
            "total_interventions": sum(c.intervention_count for c in classifications),
            "success_rate": sum(1 for c in classifications if c.success_level == "complete") / total_conversations,
            "intent_distribution": intent_counts,
            "complexity_distribution": complexity_counts,
            "success_distribution": success_counts,
            "tone_distribution": tone_counts,
            "intervention_categories": intervention_categories,
            "intervention_severities": intervention_severities,
            "top_systemic_issues": sorted(common_issues.items(), key=lambda x: x[1], reverse=True)[:10],
            "top_recommendations": sorted(common_recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
        }
    
    async def _synthesize_claude_md_improvements(self,
                                                claude_md: ClaudeMdStructure,
                                                summary_stats: Dict[str, Any],
                                                deep_analyses: List[DeepConversationAnalysis]) -> Dict[str, Any]:
        """Synthesize improvements to CLAUDE.md based on analysis."""
        # Format existing rules
        existing_rules = "\n".join([
            f"- {rule.title}: {rule.content[:100]}..."
            for rule in claude_md.rules[:20]  # Limit to avoid token limits
        ])
        
        # Format intervention patterns
        intervention_patterns = []
        if summary_stats.get('intervention_categories'):
            for category, count in list(summary_stats['intervention_categories'].items())[:5]:
                intervention_patterns.append(f"{category}: {count}")
        
        # Format top issues and recommendations
        top_issues = [issue for issue, _ in summary_stats.get('top_systemic_issues', [])[:5]]
        top_recommendations = [rec for rec, _ in summary_stats.get('top_recommendations', [])[:5]]
        
        prompt = self.prompts["synthesize_claude_md"].format(
            existing_rules=existing_rules,
            total_conversations=summary_stats.get('total_conversations', 0),
            success_rate=summary_stats.get('success_rate', 0) * 100,
            intervention_patterns=", ".join(intervention_patterns),
            top_issues=", ".join(top_issues),
            top_recommendations=", ".join(top_recommendations)
        )
        
        try:
            response = await self.async_client.messages.create(
                model=self.models["synthesizer"],
                max_tokens=3000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            content = response.content[0].text
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            else:
                logger.error("No JSON found in synthesis response")
                return {}
                
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return {}