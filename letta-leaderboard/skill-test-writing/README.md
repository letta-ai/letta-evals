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
├── skill_test_debug.yaml   # Debug config (fewer models/samples)
├── judge_prompt.md         # Grader rubric
├── generate_dataset.py     # Generates data/dataset.jsonl from skills
├── results/
│   └── parse_results.py    # Parse results.jsonl for inspection
├── data/dataset.jsonl      # 13 skill samples
├── reference_skills/       # 13 skill domains
└── reference_tests/        # ~130 example test cases
```
