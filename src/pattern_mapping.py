"""Pattern mapping configuration for comprehensive CLAUDE.md improvements."""

# Comprehensive mapping of intervention patterns to CLAUDE.md sections and improvements
INTERVENTION_PATTERN_MAPPING = {
    "incomplete_task": {
        "count": 73,
        "severity": "high",
        "target_sections": ["Task Management Protocol", "Task Completion Gate"],
        "required_improvements": [
            {
                "type": "new_guideline",
                "title": "Task Completion Verification Protocol",
                "content": """### Task Completion Verification
Before marking any task complete:
1. Run all tests and verify they pass
2. Check that ALL original requirements are met
3. Review git diff to ensure no unintended changes
4. Verify no TODOs or FIXMEs remain
5. Confirm user's success criteria are satisfied

**Red flags that task is NOT complete:**
- "I'll fix the tests later"
- "This mostly works"
- "The main functionality is done"
- Uncommitted changes remain
- Any errors in console/logs""",
                "placement": "After 'Task Completion Gate' section"
            },
            {
                "type": "clarification",
                "existing": "A task is ONLY complete when",
                "addition": "Use this as a HARD STOP checklist - if ANY item is unchecked, the task is NOT complete."
            }
        ]
    },
    
    "other": {
        "count": 71,
        "severity": "medium",
        "target_sections": ["Antipattern Prevention", "Recovery Procedures"],
        "required_improvements": [
            {
                "type": "new_guideline", 
                "title": "Session Management Protocol",
                "content": """### Session Management Protocol
When conversations become unclear or drift:
1. STOP and run context check:
   - pwd && git status
   - Review original task definition
   - Ask: "What specific outcome do you need?"
2. If no clear task exists:
   - "I notice we haven't defined a specific task. What would you like to accomplish?"
3. For continuation sessions:
   - Verify previous work state
   - Confirm next steps before proceeding""",
                "placement": "In 'Recovery Procedures' section"
            }
        ]
    },
    
    "technical_error": {
        "count": 46,
        "severity": "high",
        "target_sections": ["Project Discovery Protocol", "Verification Commands"],
        "required_improvements": [
            {
                "type": "section_enhancement",
                "section": "Project Discovery Protocol",
                "addition": """### Tool Availability Check
Before using ANY external tool or service:
```bash
# Verify critical tools
command -v npm >/dev/null 2>&1 || echo "npm not available"
command -v docker >/dev/null 2>&1 || echo "docker not available"
command -v git >/dev/null 2>&1 || { echo "CRITICAL: git not available"; exit 1; }

# Test service connectivity
curl -s http://localhost:3000/health || echo "Service not responding"
nc -zv localhost 5432 2>/dev/null || echo "Database not accessible"
```

**Fallback strategies:**
- If npm fails: Use yarn or manual node execution
- If docker fails: Run services directly
- If curl fails: Document API calls as comments"""
            }
        ]
    },
    
    "premature_action": {
        "count": 42,
        "severity": "high",
        "target_sections": ["Task Management Protocol", "Before You Start"],
        "required_improvements": [
            {
                "type": "clarification",
                "existing": "Before You Start: Task Definition",
                "addition": "MANDATORY: Do not write ANY code until this template is completed and confirmed with user."
            },
            {
                "type": "example",
                "concept": "Premature action prevention",
                "example": """❌ WRONG:
User: "Add user authentication"
Claude: *immediately starts coding*

✅ CORRECT:
User: "Add user authentication"
Claude: "I'll help add authentication. First, let me understand the requirements:
- What type of auth? (JWT/Session/OAuth?)
- Existing user model?
- Required endpoints?
*[fills out task template before coding]*"""
            }
        ]
    },
    
    "misunderstanding": {
        "count": 33,
        "severity": "medium",
        "target_sections": ["Communication Templates", "Task Management Protocol"],
        "required_improvements": [
            {
                "type": "new_guideline",
                "title": "Clarification Protocol",
                "content": """### Clarification Protocol
When user intent is unclear:
1. State what you understood
2. Ask for specific clarification
3. Provide examples of possible interpretations

Template:
"I understand you want to [what I think you want]. 
Specifically, are you asking me to:
- Option A: [specific interpretation]
- Option B: [alternative interpretation]
- Something else?

This helps me ensure I'm solving the right problem."
""",
                "placement": "In 'Communication Templates' section"
            }
        ]
    },
    
    "scope_creep": {
        "count": 28,
        "severity": "medium",
        "target_sections": ["Antipattern Prevention", "Task Management Protocol"],
        "required_improvements": [
            {
                "type": "section_enhancement",
                "section": "Scope Creep Prevention",
                "addition": """#### Active Scope Management
During implementation, maintain a "scope firewall":
- Write down the EXACT requirement at start
- For EVERY code change, ask: "Is this in scope?"
- When tempted to improve something:
  1. Add to "Future Improvements" list
  2. Complete original scope FIRST
  3. Ask user before expanding

**Scope creep phrases to catch yourself:**
- "While I'm here..."
- "It would be better if..."
- "I should also fix..."
- "This could use..."
→ STOP! Complete original task first."""
            }
        ]
    },
    
    "wrong_approach": {
        "count": 15,
        "severity": "low",
        "target_sections": ["Task Management Protocol", "Core Philosophy"],
        "required_improvements": [
            {
                "type": "new_guideline",
                "title": "Approach Validation",
                "content": """### Approach Validation Gate
Before implementing, validate approach:
1. State intended approach in 1-2 sentences
2. Identify simpler alternatives
3. Choose simplest approach that works
4. If user questions complexity → immediately switch to simpler

Example:
"I can solve this with:
A) Custom state management system (complex)
B) Existing Context API (simple)
→ Starting with B unless specific requirements need A"
""",
                "placement": "After 'Task Execution Rules'"
            }
        ]
    }
}

# Top recommendations that MUST be included
TOP_RECOMMENDATIONS = [
    {
        "text": "Always start with project discovery commands (pwd, ls, tree) to establish context",
        "implementation": "Already in 'Always Start Here' - add emphasis"
    },
    {
        "text": "Verify all external services and tools are available before use with simple test commands",
        "implementation": "Add to 'Tool Availability Check' section"
    },
    {
        "text": "When complexity is questioned, immediately propose and implement simpler alternatives",
        "implementation": "Add to 'Approach Validation' section"
    },
    {
        "text": "Use advanced planning tools only for tasks with 5+ interdependent steps",
        "implementation": "Add to 'Task Management Protocol'"
    },
    {
        "text": "If a tool fails twice, switch to manual alternatives or basic shell commands",
        "implementation": "Add to 'Tool Failure Recovery' section"
    }
]

# Prevention rules that MUST be converted to guidelines
HIGH_VALUE_PREVENTION_RULES = [
    {
        "rule": "Always acknowledge and respond to user questions, especially when they express frustration or confusion",
        "frequency": 5,
        "target_section": "Communication Templates"
    },
    {
        "rule": "Always test the complete pipeline with a small sample before running large batch operations",
        "frequency": 3,
        "target_section": "Testing Strategy"
    },
    {
        "rule": "When user requests naming convention changes, immediately acknowledge and apply the new convention",
        "frequency": 3,
        "target_section": "Task Management Protocol"
    }
]

def get_required_improvements_count():
    """Calculate minimum number of improvements required."""
    total = 0
    for pattern in INTERVENTION_PATTERN_MAPPING.values():
        total += len(pattern["required_improvements"])
    total += len(TOP_RECOMMENDATIONS)
    total += len(HIGH_VALUE_PREVENTION_RULES)
    return total

def validate_synthesis_coverage(synthesis_result):
    """Validate that synthesis addresses all required patterns."""
    coverage = {
        "intervention_types": [],
        "missing_patterns": [],
        "recommendations_included": 0,
        "total_improvements": 0
    }
    
    # Count improvements
    for section in ["sections_to_enhance", "new_guidelines", "clarifications_to_add", "examples_to_add"]:
        if section in synthesis_result:
            coverage["total_improvements"] += len(synthesis_result[section])
    
    # Check intervention coverage
    for pattern, config in INTERVENTION_PATTERN_MAPPING.items():
        found = False
        # Check if pattern is mentioned in any improvement
        for improvement in synthesis_result.get("sections_to_enhance", []):
            if pattern in improvement.get("reason", "").lower():
                found = True
                break
        
        if found:
            coverage["intervention_types"].append(pattern)
        else:
            coverage["missing_patterns"].append(f"{pattern} ({config['count']} occurrences)")
    
    coverage["coverage_percentage"] = len(coverage["intervention_types"]) / len(INTERVENTION_PATTERN_MAPPING) * 100
    coverage["meets_requirements"] = coverage["total_improvements"] >= 15 and coverage["coverage_percentage"] >= 80
    
    return coverage