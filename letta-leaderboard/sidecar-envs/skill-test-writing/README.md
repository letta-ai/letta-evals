# Skill Test Writing Evaluation

Measures an agent's ability to write test cases for skills. Given a skill document, the agent must produce tests that distinguish LLMs **with** the skill (should succeed) from those **without** (should struggle).

## Quick Start

```bash
# Letta Code agents
uv run letta-evals run skill_test_code.yaml -o results/code/

# Cloud agents
uv run letta-evals run skill_test_cloud.yaml -o results/cloud/
```

## Evaluation Criteria

Tests are judged on three dimensions (1-10 each):

| Dimension | Good (10) | Bad (1) |
|-----------|-----------|---------|
| **Non-Obviousness** | Tests genuine domain gotchas | Tests trivia (magic strings, arbitrary values) |
| **Clarity** | Grader accepts all valid solutions | Fragile pattern matching |
| **Realism** | Common practitioner task | Contrived scenario |

## Parsing Results for Inspection

```bash
python results/parse_results.py
```

Auto-discovers all `results/*/results.jsonl` and outputs to `parsed/`:

```
parsed/
├── debug/
│   ├── claude-sonnet-4-5/
│   │   ├── 001-agent-development/
│   │   │   ├── meta.json        # score, model, cost, rationale, tokens
│   │   │   ├── input.md         # task prompt
│   │   │   ├── submission.txt   # agent's final output
│   │   │   └── trajectory.jsonl # one {turn, type, content} per line
│   │   └── ...
│   └── ...
└── code/
    └── ...
```

## Directory Structure

```
skill-test-writing/
├── skill_test_code.yaml    # Letta Code agents
├── skill_test_cloud.yaml   # Cloud agents
├── judge_prompt.md         # Grader rubric
├── generate_dataset.py     # Generates data/dataset_all.jsonl from skills
├── results/
│   └── parse_results.py    # Parse results.jsonl for inspection
├── data/dataset_all.jsonl  # 13 skill samples
├── reference_skills/       # 13 skill domains
└── reference_tests/        # ~130 example test cases
```

## Results (2026-01-31)

Gate threshold: 60%. Both evaluations passed.

### Cloud Agents (Letta Cloud)

| Model | Score | Attempted | Status |
|-------|-------|-----------|--------|
| Claude Sonnet 4.5 | 82.3% | 13/13 | ✓ |
| Claude Opus 4.5 | 82.3% | 13/13 | ✓ |
| Claude Haiku 4.5 | 82.3% | 13/13 | ✓ |
| Gemini 3 Flash | 81.5% | 13/13 | ✓ |
| GPT-5.2 | 80.0% | 39/39 | ✓ |
| Claude Opus 4.1 | 80.0% | 13/13 | ✓ |
| GPT-5.1 Codex | 73.1% | 13/13 | ✓ |
| Gemini 3 Pro | 73.1% | 13/13 | ✓ |
| Minimax M2 | 68.5% | 13/13 | ✓ |
| GPT-5.1 Codex Mini | 68.5% | 13/13 | ✓ |
| GLM-4.6 | 64.6% | 13/13 | ✓ |
| Qwen 2.5-72B | 45.4% | 13/13 | ✗ |
| Llama 3.3-70B | 34.6% | 13/13 | ✗ |
| DeepSeek (chat/reasoner/v3.1) | - | 0/39 | error |
| Mistral Large | - | 0/13 | error |

**Summary:** 195/247 attempted (79%), avg 72.2%, cost $7.49

### Code Agents (Letta CLI)

| Model | Score | Attempted | Status |
|-------|-------|-----------|--------|
| Claude Opus 4.5 | 78.5% | 13/13 | ✓ |
| Claude Sonnet 4.5 | 75.4% | 13/13 | ✓ |
| Claude Haiku 4.5 | 75.4% | 13/13 | ✓ |
| GPT-5.2 | 73.1% | 13/13 | ✓ |
| GPT-5.1 Codex | 62.3% | 13/13 | ✓ |
| Gemini 3 Flash | 60.0% | 13/13 | ✓ |
| GPT-5.1 Codex Mini | 57.7% | 13/13 | ✗ |
| GLM-4.6 | 50.0% | 13/13 | ✗ |
| Minimax M2 | 29.2% | 13/13 | ✗ |
| OpenRouter models | - | 0/130 | error |

**Summary:** 117/247 attempted (47%), avg 62.4%, cost $4.83

### Prompt Engineering: "Reason First" Improvement

We discovered that Claude Opus scored higher in Code mode (80.4%) than Cloud mode (71.2%) despite Code having no task-specific instructions. Analysis of trajectories revealed the cause:

**Code mode behavior:**
```
Looking at this skill, the key domain knowledge it provides is:
1. Understanding that .docx files are ZIP archives...
2. Choosing the right workflow (docx-js for creation, redlining for review)...
3. The principle of "minimal, precise edits" for tracked changes...

[YAML test case]
```

**Cloud mode behavior (before fix):**
```yaml
name: docx-internal-structure
prompt: |
  What is the underlying format of a .docx file?
  What XML elements are used for tracked changes?
```

The Code mode system prompt (general-purpose coding assistant) triggered deliberation before output. The Cloud prompt went straight to YAML, sometimes producing trivia-focused tests.

**Fix:** Added explicit "reason first" instructions to the Cloud prompt:

```markdown
## Process

Before writing your test case, reason through the skill document:

1. **What domain knowledge does this skill teach?** Identify core concepts, not file formats.
2. **What workflow decisions does it enable?** Focus on "when to use X vs Y".
3. **What would a practitioner struggle with without this skill?** This is your test target.

Write this analysis first, then output your test case.
```

**Results:** Claude Opus cloud score improved from 71.2% to 81.2% (+10%), eliminating all low-scoring failures (0.3, 0.4 scores disappeared).

### Notes

- OpenRouter models (DeepSeek, Llama, Mistral, Qwen) failed with rate limit/credit errors
- Gemini 3 Pro not available in Letta CLI (only Flash works)
- Claude models lead on both Cloud and Code agents after prompt fix
