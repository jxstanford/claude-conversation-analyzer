"""Parser for CLAUDE.md files to extract rules and guidelines."""

import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class RuleCategory(Enum):
    """Categories of rules in CLAUDE.md."""
    TASK_MANAGEMENT = "task_management"
    SEARCH_FIRST = "search_first"
    TESTING = "testing"
    SCOPE_CONTROL = "scope_control"
    ERROR_HANDLING = "error_handling"
    COMMIT_PRACTICES = "commit_practices"
    CODE_STYLE = "code_style"
    ANTI_PATTERNS = "anti_patterns"
    GENERAL = "general"


@dataclass
class Rule:
    """Represents a single rule or guideline."""
    category: RuleCategory
    title: str
    description: str
    content: str = ""
    section: str = ""
    keywords: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    anti_examples: List[str] = field(default_factory=list)


@dataclass
class ClaudeMdStructure:
    """Structured representation of CLAUDE.md content."""
    rules: List[Rule] = field(default_factory=list)
    principles: List[str] = field(default_factory=list)
    antipatterns: List[str] = field(default_factory=list)
    size_limits: Dict[str, int] = field(default_factory=dict)
    commands: Dict[str, str] = field(default_factory=dict)
    raw_content: str = ""
    file_path: Optional[str] = None


class ClaudeMdParser:
    """Parses CLAUDE.md files to extract rules and guidelines."""
    
    def __init__(self):
        self.section_patterns = {
            'principles': r'##\s*(?:Core\s*)?(?:Philosophy|Principles)',
            'task_management': r'##\s*Task\s*Management',
            'antipatterns': r'##\s*(?:Antipattern|Anti-pattern)',
            'testing': r'##\s*Testing',
            'git': r'##\s*Git',
            'size_limits': r'##\s*Size\s*Limits',
        }
        
        self.rule_indicators = [
            'always', 'never', 'must', 'should', 'avoid',
            'prefer', 'don\'t', 'do not', 'required', 'mandatory'
        ]
    
    def parse(self, file_path: str) -> Optional[ClaudeMdStructure]:
        """Parse CLAUDE.md file and extract structure."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"CLAUDE.md not found at {file_path}")
            return None
        
        try:
            content = path.read_text(encoding='utf-8')
            structure = ClaudeMdStructure(raw_content=content, file_path=str(path))
            
            # Extract different sections
            self._extract_principles(content, structure)
            self._extract_rules(content, structure)
            self._extract_antipatterns(content, structure)
            self._extract_size_limits(content, structure)
            self._extract_commands(content, structure)
            
            return structure
            
        except Exception as e:
            logger.error(f"Failed to parse CLAUDE.md: {e}")
            return None
    
    def _extract_principles(self, content: str, structure: ClaudeMdStructure):
        """Extract core principles."""
        # Look for numbered principles or prime directives
        principle_patterns = [
            r'(?:Prime\s*Directives?|Core\s*Principles?).*?\n((?:[\d.]+.*?\n)+)',
            r'###\s*The\s*Prime\s*Directives?\n((?:[\d.]+.*?\n)+)',
        ]
        
        for pattern in principle_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches:
                # Extract individual principles
                principles = re.findall(r'[\d.]+\s*[*-]?\s*(.+?)(?=\n[\d.]|\n\n|\Z)', match, re.DOTALL)
                for principle in principles:
                    cleaned = principle.strip().replace('\n', ' ')
                    if cleaned and len(cleaned) > 10:
                        structure.principles.append(cleaned)
    
    def _extract_rules(self, content: str, structure: ClaudeMdStructure):
        """Extract rules from content."""
        # Split content into sections
        sections = self._split_into_sections(content)
        
        for section_title, section_content in sections.items():
            category = self._determine_category(section_title)
            
            # Look for rules in lists
            list_items = re.findall(r'[-*]\s+(.+?)(?=\n[-*]|\n\n|\Z)', section_content, re.DOTALL)
            
            for item in list_items:
                if self._is_rule(item):
                    rule = self._create_rule(item, category)
                    if rule:
                        structure.rules.append(rule)
            
            # Look for rules in paragraphs with strong indicators
            paragraphs = re.split(r'\n\n+', section_content)
            for para in paragraphs:
                if self._is_rule(para) and len(para) > 20:
                    rule = self._create_rule(para, category)
                    if rule:
                        structure.rules.append(rule)
    
    def _extract_antipatterns(self, content: str, structure: ClaudeMdStructure):
        """Extract antipatterns."""
        # Find antipattern section
        antipattern_match = re.search(
            r'##\s*(?:Antipattern|Anti-pattern).*?\n(.*?)(?=\n##|\Z)',
            content,
            re.IGNORECASE | re.DOTALL
        )
        
        if antipattern_match:
            section = antipattern_match.group(1)
            
            # Extract phrases that signal problems
            problem_patterns = [
                r'(?:Phrases?\s*That\s*Signal\s*Problems?).*?\n((?:[-*].*?\n)+)',
                r'(?:Actions?\s*That\s*Signal\s*Problems?).*?\n((?:[-*].*?\n)+)',
            ]
            
            for pattern in problem_patterns:
                matches = re.findall(pattern, section, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    items = re.findall(r'[-*]\s+"?([^"\n]+)"?', match)
                    structure.antipatterns.extend(items)
    
    def _extract_size_limits(self, content: str, structure: ClaudeMdStructure):
        """Extract size limits."""
        # Look for size limit tables or lists
        size_patterns = [
            r'(\w+).*?(\d+)\s*lines?\s*max',
            r'(\w+).*?Max\s*Size.*?(\d+)\s*lines?',
            r'(\w+):\s*(\d+)\s*lines?\s*max',
        ]
        
        for pattern in size_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                item_type = match[0].lower()
                limit = int(match[1])
                structure.size_limits[item_type] = limit
    
    def _extract_commands(self, content: str, structure: ClaudeMdStructure):
        """Extract command patterns."""
        # Look for verification commands, test commands, etc.
        command_sections = re.findall(
            r'```(?:bash|sh)\n(.*?)\n```',
            content,
            re.DOTALL
        )
        
        for section in command_sections:
            lines = section.strip().split('\n')
            for line in lines:
                # Extract command patterns
                if 'test' in line.lower():
                    structure.commands['test'] = line.strip()
                elif 'lint' in line.lower():
                    structure.commands['lint'] = line.strip()
                elif 'status' in line.lower():
                    structure.commands['status'] = line.strip()
    
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Split content into sections by headers."""
        sections = {}
        
        # Find all headers
        header_pattern = r'^(#{1,3})\s+(.+?)$'
        headers = list(re.finditer(header_pattern, content, re.MULTILINE))
        
        for i, match in enumerate(headers):
            title = match.group(2).strip()
            start = match.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            
            sections[title] = content[start:end].strip()
        
        return sections
    
    def _determine_category(self, section_title: str) -> RuleCategory:
        """Determine rule category from section title."""
        title_lower = section_title.lower()
        
        if 'task' in title_lower:
            return RuleCategory.TASK_MANAGEMENT
        elif 'search' in title_lower:
            return RuleCategory.SEARCH_FIRST
        elif 'test' in title_lower:
            return RuleCategory.TESTING
        elif 'scope' in title_lower:
            return RuleCategory.SCOPE_CONTROL
        elif 'error' in title_lower or 'recovery' in title_lower:
            return RuleCategory.ERROR_HANDLING
        elif 'git' in title_lower or 'commit' in title_lower:
            return RuleCategory.COMMIT_PRACTICES
        elif 'style' in title_lower or 'pattern' in title_lower:
            return RuleCategory.CODE_STYLE
        elif 'anti' in title_lower:
            return RuleCategory.ANTI_PATTERNS
        else:
            return RuleCategory.GENERAL
    
    def _is_rule(self, text: str) -> bool:
        """Check if text contains a rule."""
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in self.rule_indicators)
    
    def _create_rule(self, text: str, category: RuleCategory) -> Optional[Rule]:
        """Create a Rule object from text."""
        # Clean up text
        text = text.strip().replace('\n', ' ')
        
        # Extract title (first sentence or line)
        title_match = re.match(r'^([^.!?]+[.!?]?)', text)
        title = title_match.group(1) if title_match else text[:50] + '...'
        
        # Extract keywords
        keywords = []
        for indicator in self.rule_indicators:
            if indicator in text.lower():
                keywords.append(indicator)
        
        return Rule(
            category=category,
            title=title,
            description=text,
            keywords=keywords
        )
    
    def generate_template(self) -> str:
        """Generate a starter CLAUDE.md template."""
        template = """# Claude.md - AI Assistant Guidelines

## Mission Statement
[Your project's mission and Claude's role]

## Core Principles

1. **Think First, Code Second** - Always plan before implementing
2. **Search Before Creating** - Check existing code before writing new
3. **Test Everything** - No code without tests
4. **Maintain Scope** - Do only what's asked, nothing more

## Task Management

### Before Starting
- [ ] Understand the requirement
- [ ] Search for existing implementations
- [ ] Plan the approach
- [ ] Consider edge cases

### During Implementation
- [ ] Write tests first or alongside code
- [ ] Keep changes focused
- [ ] Run tests frequently
- [ ] Check for regressions

### After Completion
- [ ] Run all tests
- [ ] Check lint/format
- [ ] Review changes
- [ ] Ensure requirements are met

## Code Style

- Follow existing patterns in the codebase
- Use meaningful variable names
- Keep functions small and focused
- Add comments only when necessary

## Testing Requirements

- Unit tests for all new functions
- Integration tests for new features
- No failing tests in commits
- Test edge cases and error conditions

## Git Practices

- Small, focused commits
- Clear commit messages
- No generated files in commits
- Test before committing

## Size Limits

- Functions: 30 lines max
- Files: 200 lines max
- Classes: 5 public methods max

## Common Commands

```bash
# Run tests
npm test

# Lint code
npm run lint

# Check types
npm run typecheck
```

## Anti-patterns to Avoid

- Creating files without searching first
- Adding features not requested
- Skipping tests to save time
- Large, unfocused commits
- Assuming library availability

## Recovery Procedures

### When Lost
1. Check current directory
2. Review git status
3. Re-read requirements
4. Ask for clarification

### When Tests Fail
1. Read error messages carefully
2. Check recent changes
3. Run tests in isolation
4. Revert if necessary

---
*Customize this template based on your project's specific needs*
"""
        return template
    
    def check_compliance(self, rule: Rule, text: str) -> bool:
        """Check if text complies with a rule."""
        text_lower = text.lower()
        
        # Check for rule keywords
        for keyword in rule.keywords:
            if keyword in ['always', 'must', 'required']:
                # For positive rules, check if action is present
                # This is simplified - real implementation would be more sophisticated
                return True
            elif keyword in ['never', 'avoid', "don't"]:
                # For negative rules, check if action is absent
                return True
        
        return True  # Default to compliant
    
    def get_rules_by_category(self, structure: ClaudeMdStructure, category: RuleCategory) -> List[Rule]:
        """Get all rules in a specific category."""
        return [rule for rule in structure.rules if rule.category == category]