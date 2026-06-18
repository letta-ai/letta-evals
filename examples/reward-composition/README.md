# Reward Composition Example

This example demonstrates custom reward composers: Python functions that turn all grader outputs for one sample into the canonical training reward stored on `SampleResult.reward`.

## What This Example Shows

- A custom reward that returns the rubric quality score only when the ASCII validator passes
- A custom reward that computes a weighted combination of two grader scores
- Diagnostic grader outputs remain in `grades`; reward metadata stores only derived branch decisions

## Running This Example

```bash
cd examples/reward-composition
letta-evals run suite.logical-and.yaml
letta-evals run suite.weighted-average.yaml
```

## Reward Composer API

The suite points at a decorated Python function:

```yaml
reward:
  kind: custom
  function: rewards.py:quality_if_ascii
```

```python
from letta_evals import RewardOutput, reward_composer


@reward_composer
def quality_if_ascii(ctx):
    quality = ctx.grades["quality"].score
    ascii_only = ctx.grades["ascii_only"].score
    if ascii_only < 1.0:
        return RewardOutput(
            score=0.0,
            metadata={"reason": "ascii_check_failed", "failed_metric": "ascii_only"},
        )
    return RewardOutput(score=quality)
```

The framework validates that the returned score is between 0.0 and 1.0 and persists the composed reward next to the raw grader outputs.
