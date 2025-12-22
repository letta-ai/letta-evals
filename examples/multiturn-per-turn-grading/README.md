# Multi-turn Per-Turn Grading Example

This example demonstrates per-turn evaluation in multi-turn conversations, where each turn is graded against its own ground truth.

## What This Example Shows

- Using per-turn evaluation where each input message has a corresponding ground truth
- Grading each turn independently using the same extractor and grader
- Calculating proportional scores across turns (e.g., 2/3 turns correct = 0.67 score)
- Accessing per-turn results in the grade metadata

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

## How Per-Turn Evaluation Works

1. **Detection**: When both `input` and `ground_truth` are lists, per-turn mode is enabled
2. **Extraction**: For each turn, the extractor runs on that turn's trajectory only
3. **Grading**: Each turn's submission is graded against its corresponding ground truth
4. **Scoring**: Final score = average of all turn scores (proportional)

## Result Structure

The `GradeResult` includes per-turn details in metadata:

```python
GradeResult(
    score=0.67,  # 2/3 turns passed
    rationale=None,
    metadata={
        "per_turn_grades": [
            {"turn": 0, "score": 1.0, "rationale": "...", "submission": "Paris", "ground_truth": "Paris"},
            {"turn": 1, "score": 1.0, "rationale": "...", "submission": "Berlin", "ground_truth": "Berlin"},
            {"turn": 2, "score": 0.0, "rationale": "...", "submission": "Madrid", "ground_truth": "Rome"},
        ],
        "turns_passed": 2,
        "turns_total": 3
    }
)
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
