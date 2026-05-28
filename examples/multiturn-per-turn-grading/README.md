# Multi-turn Per-Turn Grading Example

This example demonstrates per-turn grading. When both `input` and `ground_truth` are lists of the same length, each assistant turn is graded against the corresponding ground-truth item and the final score is the average across turns.

```bash
cd examples/multiturn-per-turn-grading
letta-evals run suite.yaml
```

The suite uses the built-in `contains` grader and `last_assistant` extractor. Each turn is isolated for grading, so the first answer is checked against the first ground truth, the second answer against the second ground truth, and so on.
