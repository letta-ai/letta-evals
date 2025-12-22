# Multi-turn Per-Turn Grading Example

This example demonstrates per-turn evaluation in multi-turn conversations, where each turn is graded independently against its own ground truth.

## What This Example Shows

- Using per-turn evaluation where each input message has a corresponding ground truth
- Grading each turn independently using multiple graders (tool-based and LLM-based)
- Calculating proportional scores across turns (e.g., 2/3 turns correct = 0.67 score)
- Accessing per-turn results via `GradeResult.per_turn_grades`

## Key Takeaway

When both `input` and `ground_truth` are lists of the same length, Letta Evals automatically switches to per-turn evaluation mode. Each turn is graded independently, and the final score is the average across all turns.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run the evaluation:
```bash
cd examples/multiturn-per-turn-grading
letta-evals run suite.yaml
```

### Letta Cloud Setup

Set these environment variables:
```bash
export LETTA_API_KEY=your-api-key
export LETTA_PROJECT_ID=your-project-id
```

Update `base_url` in `suite.yaml`:
```yaml
target:
  base_url: https://api.letta.com/
```

Then run the evaluation as above.

## Configuration Details

### Dataset Format

Each sample in `dataset.jsonl` has a multi-turn `input` (list of strings) and a corresponding `ground_truth` (list of strings):

```json
{
  "input": [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?"
  ],
  "ground_truth": ["Paris", "Berlin", "Rome"]
}
```

**Key points:**
- `input`: A list of messages representing a multi-turn conversation
- `ground_truth`: A list of expected answers, one for each turn
- The lists must have the same length
- Each turn is graded independently against its corresponding ground truth

### Suite Configuration

This example uses multiple graders to evaluate each turn:

```yaml
graders:
  correctness:
    kind: tool
    display_name: "Correctness"
    function: contains
    extractor: last_assistant
  quality:
    kind: model_judge
    display_name: "Quality"
    prompt_path: rubric.txt
    model: gpt-4.1-mini
    extractor: last_assistant
gate:
  kind: weighted_average
  weights:
    correctness: 0.6
    quality: 0.4
  aggregation: avg_score
  op: gte
  value: 0.6
```

Each grader independently evaluates all turns, producing its own `per_turn_grades`.

## How Per-Turn Evaluation Works

1. **Detection**: When both `input` and `ground_truth` are lists, per-turn mode is enabled
2. **Extraction**: For each turn, the extractor runs on that turn's trajectory only
3. **Grading**: Each turn's submission is graded against its corresponding ground truth
4. **Scoring**: Final score = average of all turn scores (proportional)

## Result Structure

The `GradeResult` includes per-turn grades as a typed field:

```python
from letta_evals.models import GradeResult, PerTurnGrade

# Access via sample_result.grades["grader_key"]
grade_result = sample_result.grades["correctness"]

# GradeResult structure
GradeResult(
    score=0.67,  # Average across turns (2/3 passed)
    rationale=None,
    per_turn_grades=[
        PerTurnGrade(turn=0, score=1.0, rationale="...", submission="Paris", ground_truth="Paris"),
        PerTurnGrade(turn=1, score=1.0, rationale="...", submission="Berlin", ground_truth="Berlin"),
        PerTurnGrade(turn=2, score=0.0, rationale="...", submission="Madrid", ground_truth="Rome"),
    ],
    metadata={
        "turns_passed": 2,
        "turns_total": 3
    }
)

# Accessing per-turn grades
for grade in sample_result.grades["correctness"].per_turn_grades:
    print(f"Turn {grade.turn}: {grade.score} - {grade.rationale}")

# Multi-grader access
for grader_key, grade_result in sample_result.grades.items():
    print(f"\n{grader_key}: {grade_result.score}")
    if grade_result.per_turn_grades:
        for g in grade_result.per_turn_grades:
            print(f"  Turn {g.turn}: {g.score}")
```

## Comparison with Standard Multi-turn Evaluation

| Feature | Standard Multi-turn | Per-Turn Evaluation |
|---------|---------------------|---------------------|
| `input` | `List[str]` | `List[str]` |
| `ground_truth` | `str` (single) | `List[str]` (one per turn) |
| Evaluation | Final output only | Each turn independently |
| Score | Binary (pass/fail) | Proportional (avg across turns) |
| Use case | Final answer matters | Each step matters |

## When to Use Per-Turn Evaluation

Use per-turn evaluation when:
- Each step in a conversation needs to be correct
- You want to measure partial success (e.g., 2/3 questions answered correctly)
- Testing sequential reasoning where intermediate answers matter
- Evaluating tutoring or Q&A agents across multiple questions

Use standard multi-turn (single ground_truth) when:
- Only the final answer matters
- Earlier turns are just context/setup
- Testing memory updates where only final state matters
