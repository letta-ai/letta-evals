# Skill Test Writing Evaluation

Measures an agent's ability to write test cases for skills. Given a skill document, the agent must produce tests that distinguish LLMs **with** the skill (should succeed) from those **without** (should struggle).

## Quick Start

```bash
# Full evaluation
uv run letta-evals run suite.yaml -o results/

# Debug with single model
uv run letta-evals run suite-debug.yaml -o results/
```

## Output Files

When using `-o results/`:

| File | Contents |
|------|----------|
| `results/results.jsonl` | Per-sample results (JSONL) |
| `results/summary.json` | Aggregate metrics |
| `results/header.json` | Suite config snapshot |

```bash
# View results
cat results/summary.json | jq .
cat results/results.jsonl | jq -s '.[] | {id: .sample.id, model: .model_name, score: .grade.score}'
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
python parse_results.py
```

Outputs to `parsed/` (gitignored), organized by model then test case:

```
parsed/
├── claude-haiku-4-5/
│   ├── 000-command-development/
│   │   ├── meta.json        # score, model, cost, rationale, tokens
│   │   ├── input.md         # task prompt
│   │   ├── submission.txt   # agent's final output
│   │   └── trajectory.jsonl # one {turn, type, content} per line
│   └── ...
└── claude-sonnet-4-5/
    └── ...
```

## Directory Structure

```
skill-test-writing/
├── suite.yaml              # Full evaluation config
├── suite-debug.yaml        # Single-model debug config
├── judge_prompt.txt        # Grader rubric
├── custom_extractor.py     # Test file extractor
├── parse_results.py        # Parse results.jsonl for inspection
├── data/dataset.csv        # 13 skill samples
├── reference_skills/       # 13 skill domains (pdf, xlsx, docx, webapp-testing, etc.)
└── reference_tests/        # ~130 example test cases
```
