# Claude.md - Production Development Guide

## Mission Statement
Build production software through disciplined, task-based development with continuous verification and zero tolerance for deviations from specifications.

---

## 🔍 Project Discovery Protocol

### Always Start Here
```bash
# 1. Understand where you are
pwd
ls -la
tree -L 2 -I 'node_modules|__pycache__|.git'

# 2. Identify project type and tooling
cat package.json 2>/dev/null          # Node.js project?
cat pyproject.toml 2>/dev/null        # Python project?
cat go.mod 2>/dev/null                # Go project?
cat Cargo.toml 2>/dev/null            # Rust project?
cat docker-compose.yml 2>/dev/null    # Services architecture?

# 3. Check for existing conventions
cat README.md
cat CONTRIBUTING.md 2>/dev/null
cat .github/workflows/*.yml 2>/dev/null
cat Makefile 2>/dev/null

# 4. Look for project-specific Claude instructions
cat CLAUDE.md 2>/dev/null
cat .claude/instructions.md 2>/dev/null
```

### Project Context Management

#### Check for Existing CLAUDE.md
If found, respect existing team conventions. Create `CLAUDE.md.local` for additional context:

```markdown
# CLAUDE.md.local - Project-Specific Context

## Project Analysis
- **Type**: [CLI/API/Web App/Library]
- **Stack**: [Languages and frameworks found]
- **Build Tool**: [npm/yarn/pnpm/pip/cargo/go]
- **Test Runner**: [jest/pytest/go test]

## Commands Discovered
- **Install**: [command]
- **Test**: [command]
- **Lint**: [command]
- **Build**: [command]
- **Run**: [command]

## Patterns Observed
- [Architecture patterns]
- [File organization]
- [Naming conventions]
```

**Always add CLAUDE.md.local to .gitignore**

### Tool Selection Hierarchy
1. **Use existing project tools** if found
2. **Check team conventions** in CONTRIBUTING.md
3. **Fall back to defaults** only when no preference exists

#### Default Tool Preferences
**Python**: uv → poetry → pip | pytest | ruff → black+flake8  
**JavaScript**: pnpm → npm | vitest → jest | eslint+prettier  
**TypeScript**: Same as JS + tsc for type checking  
**Bash**: shellcheck for linting | bats for testing

---

## Core Philosophy

### The Prime Directives
1. **Think First, Code Second** - Always plan before implementing
2. **One Task at a Time** - Complete vertically before moving horizontally  
3. **Search Before Creating** - Existing code is your friend
4. **Simple Beats Clever** - YAGNI, KISS, and boring solutions win
5. **Tests Are Mandatory** - No exceptions, no "fix it later"

### Engineering Principles

#### YAGNI (You Aren't Gonna Need It)
Only build what's required RIGHT NOW. If the test passes without it, you don't need it.

#### KISS (Keep It Simple, Stupid)
The best solution is the most boring solution. If you can't explain it in one sentence, it's too complex.

#### The Rule of Three (DRY with Discipline)
1. First occurrence: Write it
2. Second occurrence: Copy it, note duplication
3. Third occurrence: Extract to shared function

#### Walking Skeleton (Incremental Development)
Start with the thinnest working version. Add flesh incrementally. Always have something demonstrable.

---

## 📋 Task Management Protocol

### Before You Start: Task Definition

```markdown
# Task: [Clear, specific title]

## Current State
- Existing files: [List from search]
- Current functionality: [What works now]

## Desired State  
- Requirement: [Single specific need]
- Success criteria: [Observable outcome]

## Scope
IN: [What to build]
OUT: [What to ignore]

## Implementation Plan
1. [ ] Search existing code
2. [ ] Identify files to modify
3. [ ] Write simplest solution
4. [ ] Add tests
5. [ ] Verify all tests pass
6. [ ] Check lint/format
7. [ ] Review changes

## Complexity: [Low/Medium/High]
If High → STOP and decompose
```

### Task Execution Rules

#### 1. One Task at a Time
- No parallel work
- No "while I'm here" improvements  
- Complete current task fully before next

#### 2. Vertical Slices Only
```
✅ Correct: UI → API → DB for one feature
❌ Wrong: All controllers, then all services
```

#### 3. Checkpoint Every 30 Minutes
```bash
pwd                    # Verify location
git status            # Check changes  
npm test              # Or project test command
# Am I still on task?
```

### Task Completion Gate

A task is ONLY complete when:
- ✅ Original requirement met (nothing more)
- ✅ Tests written and passing
- ✅ Lint/format passing  
- ✅ No temporary files
- ✅ Git diff reviewed
- ✅ Scope maintained

---

## 🚫 Antipattern Prevention

### Context Loss Prevention
```bash
# Start every task with:
pwd
ls -la
git status

# Every 5 interactions:
pwd                    # Still in right place?
git diff --name-only   # What have I changed?
npm test              # Still working?
```

### Search-First Development
```bash
# Before creating ANY new code:
grep -r "functionName" . --exclude-dir=node_modules
find . -name "*similar*" -type f
git log --oneline | grep -i "feature"

# Only create new if you can explain why existing won't work
```

### Complexity Prevention

#### Size Limits
- Functions: 30 lines max
- Files: 200 lines max
- Classes: 5 public methods max
- Commits: Small and frequent (every 15-30 min)

When you hit limits → STOP and decompose

### Scope Creep Prevention
When tempted to add "improvements":
```
STOP. 
1. Complete current task
2. Document improvement separately
3. Get approval before implementing
```

---

## 🧪 Testing Strategy

### Test Categories
**Unit Tests**: Fast, isolated, no external dependencies  
**Integration Tests**: Component interactions, may use test DB  
**E2E Tests**: Full workflow validation  
**Shell Tests**: BATS for bash scripts (if applicable)

### Testing Rules
1. **Write tests first** when fixing bugs
2. **Write tests during** feature development
3. **Never skip tests** to "save time"
4. **Each commit** must have passing tests

### Test Structure
```javascript
// JavaScript/TypeScript
describe('Component', () => {
  it('should handle normal case', () => {
    // Arrange
    // Act
    // Assert
  });
  
  it('should handle edge case', () => {});
  it('should handle error case', () => {});
});
```

```python
# Python
def test_normal_case():
    # Arrange
    # Act  
    # Assert
    
def test_edge_case():
    pass
    
def test_error_case():
    pass
```

---

## 💻 Language-Specific Patterns

### JavaScript/TypeScript
```javascript
// Prefer async/await over callbacks
✅ const data = await fetchData();
❌ fetchData().then(data => {});

// Use optional chaining
✅ const value = obj?.nested?.value ?? defaultValue;
❌ const value = obj && obj.nested && obj.nested.value || defaultValue;

// Explicit types in TypeScript
✅ function process(data: string[]): Promise<Result>
❌ function process(data): any
```

### Python
```python
# Use type hints
✅ def process(data: list[str]) -> dict[str, Any]:
❌ def process(data):

# Prefer pathlib over os.path
✅ from pathlib import Path
❌ import os.path

# Use context managers
✅ with open('file.txt') as f:
❌ f = open('file.txt')
```

### Bash
```bash
# Always use strict mode
set -euo pipefail

# Quote variables
✅ echo "${VAR}"
❌ echo $VAR

# Check commands exist
✅ command -v git >/dev/null 2>&1 || { echo "git required"; exit 1; }
❌ git status  # Might not exist
```

---

## 🔄 Git Workflow

### Commit Strategy
```bash
# Frequent small commits during development
git add -p  # Stage selectively
git commit -m "Add user validation"
git commit -m "Add user validation tests"
git commit -m "Fix edge case in validation"

# Each commit should:
- Be focused on one change
- Include its tests
- Leave code in working state
- Have clear message explaining "why"
```

### Commit Messages
```
✅ Add user email validation with regex
✅ Fix null pointer in payment processor
✅ Refactor order service for testability

❌ Update code
❌ Fix bug
❌ WIP
```

**NEVER reference Claude in commits or code**

---

## 🛡️ Quality Gates

### Cannot Start Coding Until
1. ✅ Project discovery complete
2. ✅ Task defined in writing
3. ✅ Existing code searched
4. ✅ Test approach planned

### Must Stop and Review If
- File exceeds 200 lines
- Function exceeds 30 lines  
- Tests start failing
- Scope expanding
- Can't explain in one sentence

### Cannot Commit Until
1. ✅ Tests pass
2. ✅ Lint passes (use project's linter)
3. ✅ No temp files
4. ✅ Changes reviewed
5. ✅ Scope verified

---

## 🔍 Verification Commands

### Universal Checks
```bash
# Context
pwd
git status
git diff --staged

# Find test command
npm test 2>/dev/null || yarn test 2>/dev/null || pytest 2>/dev/null || go test ./... 2>/dev/null

# Find lint command  
npm run lint 2>/dev/null || ruff check 2>/dev/null || golangci-lint run 2>/dev/null

# Clean workspace
find . -name "*.tmp" -o -name "*.bak" -o -name ".DS_Store"
```

---

## 🔄 Recovery Procedures

### When Lost in Codebase
```bash
# STOP and reorient
pwd
git status
git diff
# Reread task definition
# Confirm next action
```

### When Tests Fail
```bash
# STOP and isolate
git diff HEAD  # What changed?
git stash      # Save work
git checkout HEAD~1  # Go to last working state
# Apply changes incrementally
```

### When Overengineering
1. Comment out abstractions
2. Write direct implementation
3. Verify it works
4. Only add complexity if truly needed

---

## 📝 Communication Templates

### Complexity Check
```
⚠️ COMPLEXITY CHECK

Task becoming complex:
- Files to modify: [count]
- Lines of code: [estimate]

Recommendation: Decompose
Proceeding with simplest approach.
```

### Scope Creep Alert
```
⚠️ SCOPE CREEP DETECTED

Original: [requirement]
Found: [additional work]

Completing original only.
Additional work noted for separate task.
```

---

## 🚨 Red Flags - Stop Immediately

### Phrases That Signal Problems
- "Let me create a base class"
- "I'll implement all endpoints first"
- "While I'm here, I'll also..."
- "I'll fix the tests later"
- "This might be useful"

### Actions That Signal Problems
- Creating new files without searching
- Adding abstractions on first implementation
- Working on multiple tasks
- Skipping tests
- Refactoring unrelated code

---

## Daily Workflow

### Session Start
```bash
# 1. Project discovery
pwd
cat package.json 2>/dev/null || cat pyproject.toml 2>/dev/null
git pull
git status

# 2. Verify working state  
npm test 2>/dev/null || pytest 2>/dev/null

# 3. Review task
# 4. Search codebase
```

### During Development
- Commit every 15-30 minutes
- Test after every function
- Checkpoint every 5 steps
- Stay on task

### Session End
```bash
# Clean up
git status
npm test  # Or appropriate test command
git add -A && git commit -m "WIP: [description]" || git stash
# Document progress
```

---

## Quick Reference

### Size Limits
| Item | Max Size | Action if Exceeded |
|------|----------|-------------------|
| Function | 30 lines | Extract helper functions |
| File | 200 lines | Split into modules |
| Class | 5 methods | Decompose responsibility |
| Task | 2-4 hours | Break into subtasks |
| Commit | 30 minutes | Commit more frequently |

### Commands to Run Constantly
```bash
pwd                    # Where am I?
git status            # What changed?
[test command]        # What broke?
git diff              # What exactly changed?
```

---

## Remember

**Honor existing patterns.** When in Rome, do as the Romans do.

**Build boring code that works.** Clever code is a bug waiting to happen.

**Every task is a contract.** Define it, execute it, complete it.

**Small commits win.** Easier to review, easier to revert.

**When in doubt, grep it out.** Search before you create.

---

*This guide is your guardrail. When something feels wrong, it probably is. Stop and check this guide.*