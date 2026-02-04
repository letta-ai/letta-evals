"""Generate synthetic bloated memory blocks for defrag eval.

Strategies:
1. Merge: Combine multiple focused blocks into one monolithic block
2. Duplicate: Add redundant info with slight variations
3. Degrade: Take clean structure, add inconsistent formatting
4. Expand: Use LLM to expand seed topics into realistic bloat
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

from agentfile import AgentFile


# Seed topics for synthetic generation - realistic detail level
TOPIC_SEEDS = {
    "git_workflow": [
        """## Git Safety - CRITICAL
- **ABSOLUTELY NEVER use `git add .` or `git add -A` or `git add --all`**
  - This has caused serious security incidents with sensitive files pushed to public repos
  - Agent files, API keys, conversation history can be exposed
  - ALWAYS explicitly list files: `git add file1.ts file2.ts file3.ts`
- If unsure what changed, run `git status` and `git diff --stat` first
- **NEVER force push unless intentionally patching git history**
  - Force push is ONLY for hiding something bad (secrets, credentials)
  - This applies to ALL branches, not just main
  - For ANY other change (even small fixes), make a follow-up commit
  - History is fine. Extra commits are fine. Force pushing is NOT fine.""",
        """## Commit Messages
- Use conventional commits format: `type(scope): subject`
- Types: feat | fix | docs | style | refactor | perf | test | build | ci | chore
- Keep subject line under 50 chars, body explains why not how
- Read existing commits in codebase first, match that style
- Example: `fix(auth): handle expired token refresh race condition`""",
        """## Branch Strategy
- NEVER push directly to main - ALWAYS create a feature branch and PR
- Branch naming: `username/feature-name` or `fix/issue-123`
- Delete branches after merge
- If a PR was merged, the branch name will appear in merge commit - create NEW branch for new work
- When pushing new branch: `git push -u origin HEAD`""",
        """## PR Process
- Always run `npm run lint` (not just typecheck/build) before declaring PR ready
- Lint catches formatting issues that typecheck misses
- Include test plan in description
- Link related issues with `Fixes #123` or `Related to #456`
- Request review from appropriate team members""",
    ],
    "coding_style": [
        """## Error Handling
- Don't use bare `except:` - always catch specific exceptions
- Log errors with full context: what operation, what inputs, what state
- Fail fast on invalid input - validate at system boundaries
- Don't add error handling for scenarios that can't happen
- Trust internal code and framework guarantees

```python
# BAD
try:
    result = do_something()
except:
    pass

# GOOD  
try:
    result = do_something()
except ConnectionError as e:
    logger.error(f"Failed to connect to {url}: {e}")
    raise
```""",
        """## Type Hints
- Use type hints for public APIs and when it genuinely helps
- Match codebase style for Optional - some use `Optional[X]`, others `X | None`
- Don't over-annotate internal/private functions unless complex
- Use `TypedDict` for dict shapes, `Protocol` for duck typing

```python
def process_user(user_id: str, options: ProcessOptions | None = None) -> UserResult:
    ...
```""",
        """## Testing
- Unit tests for edge cases: parsers, validators, pure functions
- Smoke tests for pipelines: small dataset, full flow
- Mock external services (DB, APIs) but prefer real objects when not too heavy
- Tests go in `tests/` directory mirroring src structure

```bash
uv run pytest tests/ -v
uv run pytest tests/test_foo.py::test_specific -v
```""",
        """## Code Quality
- Self-documenting code preferred over comments
- Comment "why" not "what" - the code shows what
- Long functions okay if clear; clarity over line count
- No classes unless necessary - prefer functions
- Ship smallest change that solves the request
- Don't add features beyond what was asked""",
    ],
    "project_knowledge": [
        """## Architecture
- `src/` contains main application code
  - `src/api/` - REST endpoints
  - `src/models/` - Database models
  - `src/services/` - Business logic
  - `src/utils/` - Shared utilities
- `tests/` mirrors src/ structure
- Configs in root: `config.yaml`, `.env`, `pyproject.toml`
- Static assets in `public/` or `static/`""",
        """## Key Files
- `main.py` or `app.py`: Entry point, sets up FastAPI/Flask
- `config.yaml`: Runtime settings (loaded via pydantic-settings)
- `pyproject.toml`: Dependencies and tool configs
- `alembic/`: Database migrations
- `.env`: Local secrets (never commit!)
- `Makefile` or `justfile`: Common commands""",
        """## Database Gotchas
- Connections need explicit close or use context manager
- Always use transactions for multi-step operations
- Index frequently queried columns
- Beware N+1 queries - use `joinedload` or `selectinload`
- Run migrations in order: `alembic upgrade head`

```python
# BAD - connection leak
conn = db.connect()
result = conn.execute(query)
# forgot to close!

# GOOD
with db.connect() as conn:
    result = conn.execute(query)
```""",
        """## External API Gotchas
- Rate limits: implement exponential backoff
- Timeouts: always set reasonable timeouts (30s default)
- Cache responses when appropriate
- Handle partial failures gracefully
- Log request/response for debugging (sanitize secrets!)""",
    ],
    "user_preferences": [
        """## Communication Style
- Be terse. Skip explanations unless asked.
- Just show results and code.
- Describe, don't dramatize. Technical notes style.
- Don't use sycophantic openers ("Great question!")
- Point out mistakes directly, no hedging - but be constructive

When uncertain:
- Ask with your most reasonable assumption
- Request confirmation
- If truly underspecified: give options, don't guess""",
        """## User Shortcuts
- "NEXT" at start of message = moving on to new feature/bug
  - Previous work is already committed/pushed
  - User has checked out main or fresh branch
  - Ready to start something new
- "LGTM" = looks good to me, proceed
- "WIP" = work in progress, don't commit yet
- Never ask about "taking breaks" - answer is always no""",
        """## Tool Preferences
- Python: Always use `uv` not pip
- Formatting: Use Ruff
- Testing: pytest with `-v` flag
- Clipboard: pipe to `pbcopy` directly
- Shell: Prefer one-liners over scripts when simple

```bash
# User prefers
uv run pytest tests/ -v
uv pip install package

# NOT
pip install package
python -m pytest tests/
```""",
        """## Planning Preferences
- When user says to update "the plan", they mean the PLAN FILE (e.g., ~/.letta/plans/*.md)
- NOT memory blocks
- Plans should be iterated until there are "zero surprises at implementation time"
- Expect thorough planning with terminal behavior matrices, edge cases, decision trees""",
    ],
    "framework_gotchas": [
        """## React/Ink Rendering
When switching between completely different React component trees in Ink, the old render can leave artifacts/whitespace.

**Fix**: Return `null` during transitional states before determining which component to render.

```typescript
// BAD: Renders App first, then switches to Selector (causes whitespace)
if (loadingState === "selecting_global") return <Selector />;
if (!agentId) return <App loadingState="assembling" />;

// GOOD: Return null during initial state
if (loadingState === "selecting") return null;
if (loadingState === "selecting_global") return <Selector />;
if (!agentId) return <App />;
```

**Don't use ANSI escape codes** (`\\x1b[H\\x1b[2J`) to clear screen - it's a hack that doesn't work robustly.""",
        """## TypeScript Gotchas
- `tsconfig.types.json` controls which files get `.d.ts` generation
- Can't import files that import `.md/.mdx` in type-exported files
- Solution: Keep type files pure - document in JSDoc, validate at runtime
- Use `satisfies` for type narrowing with inference
- Avoid `any`, prefer `unknown` then narrow

```typescript
// Type narrowing
const config = loadConfig() satisfies Config;

// Unknown vs any
function process(data: unknown) {
  if (isValidData(data)) {
    // now data is typed
  }
}
```""",
        """## API Error Handling Flow
- All `LLMError` subclasses → `stop_reason = "llm_api_error"`
- Generic exceptions → `stop_reason = "error"`
- `run.metadata.error` is a `LettaErrorMessage` with `error_type: "llm_error"`
- **Gotcha:** Display may show provider's error (Anthropic's `'type': 'api_error'`) embedded in detail string
- Retry logic: check `stopReason === "llm_api_error"`, fallback check `metadata.error.error_type`""",
        """## Unicode Character Width
- ✔ (U+2714 HEAVY CHECK MARK) renders as 2 columns wide - causes layout issues
- ✓ (U+2713 CHECK MARK) renders as 1 column wide - safe to use
- When toggling causes newlines/layout shifts, check character widths first
- Test with `wcwidth` or manually in terminal""",
    ],
    "debugging": [
        """## Debugging Approach
- **Don't add debug logging and wait for reproduction** - user can't sit around waiting
- Use **static code analysis** to trace through code paths and find bugs
- Trace the full flow: what values are passed, what conditions are checked
- When user reports "X should have happened but didn't", trace code to find why

Steps:
1. Read error message carefully
2. Simplify/isolate the failing case
3. Add targeted prints around the failure point
4. Check assumptions - are inputs what you expect?""",
        """## File Reading Strategy
- When a search "fails", distinguish between: (1) content not in file vs (2) search too narrow
- Read whole files when they fit in context - don't over-rely on narrow grep
- Especially for config/mapping files: read the whole thing to see all cases
- Narrow searches can miss things - when debugging, read more context

```bash
# Don't just grep for one thing
grep "error" file.py

# Read the whole function/file for context
cat file.py | head -100
```""",
        """## Common Bug Patterns
- Off-by-one errors in loops and slices
- Race conditions in async code
- Null/undefined not checked before access
- String vs number comparison (`"1" == 1`)
- Mutable default arguments in Python
- Forgetting to await async functions
- Import cycles causing undefined values""",
    ],
    "security": [
        """## Security Basics
- **NEVER commit secrets** (.env, API keys, credentials)
- Use environment variables for all secrets
- Add sensitive patterns to .gitignore BEFORE creating files
- If secrets were committed, rotate them immediately (git history persists)

```bash
# Add to .gitignore
.env
*.pem
credentials.json
```""",
        """## Input Validation
- Validate ALL user input at system boundaries
- Sanitize before database queries (use parameterized queries)
- Escape HTML output to prevent XSS
- Validate file paths to prevent directory traversal
- Rate limit API endpoints

```python
# BAD - SQL injection
query = f"SELECT * FROM users WHERE id = {user_id}"

# GOOD - parameterized
query = "SELECT * FROM users WHERE id = ?"
cursor.execute(query, (user_id,))
```""",
        """## Authentication Gotchas
- Always hash passwords (bcrypt, argon2)
- Use constant-time comparison for tokens
- Set secure cookie flags (HttpOnly, Secure, SameSite)
- Implement proper session expiration
- Don't leak info in error messages ("user not found" vs "invalid credentials")""",
    ],
    "performance": [
        """## Performance Optimization
- **Measure first, optimize second** - never guess
- Profile before making changes: `cProfile`, `py-spy`, Chrome DevTools
- Focus on hot paths - 80/20 rule applies
- Database queries are usually the bottleneck

```python
import cProfile
cProfile.run('my_function()', sort='cumtime')
```""",
        """## Caching Strategies
- Cache expensive computations and external API calls
- Use appropriate TTL (time-to-live) for different data
- Invalidate cache when underlying data changes
- Consider cache stampede prevention (locks, probabilistic refresh)

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_computation(x):
    return x ** 2
```""",
        """## Database Performance
- Index columns used in WHERE, JOIN, ORDER BY
- Avoid SELECT * - only fetch needed columns
- Use EXPLAIN to understand query plans
- Batch inserts/updates instead of one-by-one
- Consider connection pooling for high concurrency""",
    ],
    "deployment": [
        """## Deployment Checklist
- [ ] Run full test suite
- [ ] Check environment variables are set
- [ ] Database migrations applied
- [ ] Static assets built/collected
- [ ] Health check endpoint working
- [ ] Logging configured
- [ ] Error monitoring set up (Sentry, etc.)
- [ ] Rollback plan ready""",
        """## Docker Best Practices
- Use multi-stage builds to reduce image size
- Don't run as root - create non-root user
- Use .dockerignore to exclude unnecessary files
- Pin base image versions (not :latest)
- Copy requirements first for better layer caching

```dockerfile
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY . .
CMD ["python", "main.py"]
```""",
        """## Environment Management
- Use separate configs for dev/staging/prod
- Never hardcode environment-specific values
- Use python-dotenv or similar for local dev
- Validate required env vars on startup

```python
import os

REQUIRED_VARS = ["DATABASE_URL", "SECRET_KEY", "API_KEY"]

def validate_env():
    missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Missing env vars: {missing}")
```""",
    ],
}


def merge_topics(topic_groups: list[str], shuffle: bool = True) -> str:
    """Merge multiple topic groups into one monolithic block."""
    all_topics = []
    for group in topic_groups:
        all_topics.extend(TOPIC_SEEDS.get(group, []))
    
    if shuffle:
        random.shuffle(all_topics)
    
    return "\n\n".join(all_topics)


def add_redundancy(content: str, redundancy_factor: float = 0.3) -> str:
    """Add redundant/duplicate information with slight variations."""
    lines = content.split("\n")
    result = []
    
    for line in lines:
        result.append(line)
        # Randomly duplicate some lines with variations
        if random.random() < redundancy_factor and line.startswith("- "):
            variations = [
                line.replace("Never", "Don't ever"),
                line.replace("Always", "Make sure to"),
                line + " (important!)",
                "Note: " + line[2:],
            ]
            result.append(random.choice(variations))
    
    return "\n".join(result)


def inconsistent_formatting(content: str) -> str:
    """Make formatting inconsistent (mix header levels, bullet styles)."""
    lines = content.split("\n")
    result = []
    
    for line in lines:
        # Randomly change header levels
        if line.startswith("## ") and random.random() < 0.3:
            line = "### " + line[3:]
        elif line.startswith("### ") and random.random() < 0.3:
            line = "## " + line[4:]
        
        # Randomly change bullet styles
        if line.startswith("- ") and random.random() < 0.2:
            line = "* " + line[2:]
        elif line.startswith("- ") and random.random() < 0.1:
            line = "• " + line[2:]
        
        result.append(line)
    
    return "\n".join(result)


def generate_bloated_block(
    num_topic_groups: int = 3,
    add_redundant: bool = True,
    inconsistent: bool = True,
    seed: Optional[int] = None,
) -> tuple[str, list[str]]:
    """Generate a synthetic bloated block.
    
    Returns:
        Tuple of (bloated_content, list_of_topic_groups_used)
    """
    if seed is not None:
        random.seed(seed)
    
    # Pick random topic groups
    groups = random.sample(list(TOPIC_SEEDS.keys()), min(num_topic_groups, len(TOPIC_SEEDS)))
    
    # Merge them
    content = merge_topics(groups)
    
    # Add bloat
    if add_redundant:
        content = add_redundancy(content)
    if inconsistent:
        content = inconsistent_formatting(content)
    
    return content, groups


def bloat_existing_agent(af: AgentFile, target_block: str = "persona", seed: Optional[int] = None) -> AgentFile:
    """Take an existing agent and bloat one of its blocks."""
    import copy
    
    if seed is not None:
        random.seed(seed)
    
    data = copy.deepcopy(af.data)
    
    for block in data.get("blocks", []):
        if block.get("label") == target_block:
            original = block.get("value", "")
            
            # Generate additional bloat
            extra_content, _ = generate_bloated_block(num_topic_groups=2)
            
            # Combine original + extra
            block["value"] = original + "\n\n---\n\n" + extra_content
            break
    
    return AgentFile(data)


def create_synthetic_agent(
    name: str = "synthetic-bloated-agent",
    num_topic_groups: int = 4,
    seed: Optional[int] = None,
) -> AgentFile:
    """Create a completely synthetic agent with bloated memory."""
    af = AgentFile.create_empty(name=name)
    
    # Generate bloated persona
    persona_content, groups = generate_bloated_block(num_topic_groups=num_topic_groups, seed=seed)
    
    # Add to agent
    data = af.data
    data["blocks"] = [
        {
            "label": "persona",
            "value": persona_content,
            "description": "Accumulated preferences and behaviors",
            "limit": 20000,
            "id": "block-0",
        },
        {
            "label": "human",
            "value": "[TODO: Fill with user info]",
            "description": "Information about the user",
            "limit": 5000,
            "id": "block-1",
        },
    ]
    
    return AgentFile(data)


def create_multi_block_agent(
    name: str = "multi-block-agent",
    seed: Optional[int] = None,
) -> AgentFile:
    """Create agent with multiple bloated blocks (more realistic)."""
    if seed is not None:
        random.seed(seed)
    
    af = AgentFile.create_empty(name=name)
    
    # Generate multiple bloated blocks
    persona_content, _ = generate_bloated_block(num_topic_groups=3, seed=seed)
    project_content, _ = generate_bloated_block(num_topic_groups=2, seed=(seed or 0) + 100)
    
    data = af.data
    data["blocks"] = [
        {
            "label": "persona",
            "value": persona_content,
            "description": "Accumulated preferences and behaviors",
            "limit": 20000,
            "id": "block-0",
        },
        {
            "label": "project",
            "value": project_content,
            "description": "Project-specific knowledge",
            "limit": 20000,
            "id": "block-1",
        },
        {
            "label": "human",
            "value": "[TODO: Fill with user info]",
            "description": "Information about the user",
            "limit": 5000,
            "id": "block-2",
        },
    ]
    
    return AgentFile(data)


def main():
    """Generate synthetic bloated agents for defrag eval."""
    output_dir = Path("defrag_test_cases/synthetic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = []
    idx = 0
    
    # Type 1: Single bloated persona block (10 agents)
    print("Generating single-block agents...")
    for i in range(10):
        num_groups = random.randint(3, 5)
        af = create_synthetic_agent(
            name=f"synthetic-single-{i:02d}",
            num_topic_groups=num_groups,
            seed=42 + i,
        )
        
        path = output_dir / f"synthetic_{idx:02d}_single.af"
        af.save(path)
        
        persona = next((b for b in af.data["blocks"] if b["label"] == "persona"), None)
        if persona:
            size = len(persona["value"])
            topics = len([l for l in persona["value"].split("\n") if l.startswith("## ") or l.startswith("### ")])
            manifest.append({
                "file": path.name,
                "size": size,
                "topics": topics,
                "type": "single_block",
            })
            print(f"  {path.name}: {size} chars, {topics} topics")
        idx += 1
    
    # Type 2: Multiple bloated blocks (10 agents)
    print("\nGenerating multi-block agents...")
    for i in range(10):
        af = create_multi_block_agent(
            name=f"synthetic-multi-{i:02d}",
            seed=100 + i,
        )
        
        path = output_dir / f"synthetic_{idx:02d}_multi.af"
        af.save(path)
        
        total_size = 0
        total_topics = 0
        for b in af.data["blocks"]:
            if b["label"] in ("persona", "project"):
                val = b["value"]
                total_size += len(val)
                total_topics += len([l for l in val.split("\n") if l.startswith("## ") or l.startswith("### ")])
        
        manifest.append({
            "file": path.name,
            "size": total_size,
            "topics": total_topics,
            "type": "multi_block",
        })
        print(f"  {path.name}: {total_size} chars, {total_topics} topics (2 blocks)")
        idx += 1
    
    # Type 3: Extremely bloated (5 agents with all topic groups)
    print("\nGenerating extreme bloat agents...")
    for i in range(5):
        af = AgentFile.create_empty(name=f"synthetic-extreme-{i:02d}")
        
        # Use ALL topic groups
        all_content, _ = generate_bloated_block(
            num_topic_groups=5,  # all groups
            add_redundant=True,
            inconsistent=True,
            seed=200 + i,
        )
        
        af.data["blocks"] = [
            {
                "label": "persona",
                "value": all_content,
                "description": "Everything dumped into one block",
                "limit": 30000,
                "id": "block-0",
            },
        ]
        
        path = output_dir / f"synthetic_{idx:02d}_extreme.af"
        af.save(path)
        
        size = len(all_content)
        topics = len([l for l in all_content.split("\n") if l.startswith("## ") or l.startswith("### ")])
        manifest.append({
            "file": path.name,
            "size": size,
            "topics": topics,
            "type": "extreme",
        })
        print(f"  {path.name}: {size} chars, {topics} topics (extreme)")
        idx += 1
    
    # Save manifest
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nGenerated {len(manifest)} synthetic agents in {output_dir}/")


if __name__ == "__main__":
    main()
