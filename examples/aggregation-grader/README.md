# Aggregation Grader Example

This example demonstrates how to use the aggregation grader to combine multiple metrics using custom Python code.

## Overview

The aggregation grader allows you to:
- Combine multiple metrics into a single score
- Use custom Python code for flexible aggregation logic
- Apply weights, take min/max, or implement any custom logic

## Configuration

In this example, we have three graders:

1. **contains_check**: Uses the built-in `contains` grader to check if the answer contains the expected output
2. **exact_check**: Uses the built-in `exact_match` grader to check for exact matches
3. **combined_score**: An aggregation grader that combines the above two metrics

### Aggregation Grader Configuration

```yaml
combined_score:
  kind: aggregation
  display_name: "Combined Score"
  function: aggregation.py:weighted_average_aggregate
  depends_on:
    - contains_check
    - exact_check
```

Where `aggregation.py` contains:

```python
from typing import Dict
from letta_evals.decorators import aggregation

@aggregation
def weighted_average_aggregate(metrics: Dict[str, float]) -> float:
    """
    Aggregate multiple metrics with custom weights.

    Args:
        metrics: Dict[str, float] containing scores from dependent graders

    Returns:
        float: Aggregated score between 0.0 and 1.0
    """
    contains_score = metrics.get('contains_check', 0.0)
    exact_score = metrics.get('exact_check', 0.0)

    # Weighted average: 70% contains, 30% exact
    weighted_score = 0.7 * contains_score + 0.3 * exact_score

    return weighted_score
```

**Note**: The `@aggregation` decorator is recommended as it validates your function signature and provides better error messages.

### Key Fields

- **kind**: Must be `aggregation`
- **function**: Path to Python file containing aggregation function (e.g., `aggregation.py:my_function`)
- **depends_on**: List of metric keys (grader names) that this aggregation depends on
- **display_name**: Optional human-friendly name for display

## Aggregation Function

The aggregation function receives a dictionary mapping metric keys to their scores:

```python
from typing import Dict
from letta_evals.decorators import aggregation

@aggregation
def my_aggregate(metrics: Dict[str, float]) -> float:
    # metrics = {'contains_check': 1.0, 'exact_check': 0.5}
    # Return a value between 0.0 and 1.0
    return combined_score
```

You can name the function anything you want, then reference it in your YAML as `filename.py:function_name`.

### Using the @aggregation Decorator

The `@aggregation` decorator is **recommended** because it:
- Validates function signature (must have exactly one parameter named `metrics`)
- Validates return type annotation (must be `float` or `int`)
- Provides clearer error messages if something is wrong
- Follows the same pattern as `@grader` and `@extractor` decorators

The decorator is optional - functions without it will still work, but you won't get the validation benefits.

## Example Aggregation Strategies

### Weighted Average
```python
def aggregate(metrics):
    return 0.7 * metrics['metric1'] + 0.3 * metrics['metric2']
```

### Minimum Score (All must pass)
```python
def aggregate(metrics):
    return min(metrics.values())
```

### Maximum Score (Best metric wins)
```python
def aggregate(metrics):
    return max(metrics.values())
```

### Average Score
```python
def aggregate(metrics):
    return sum(metrics.values()) / len(metrics)
```

### Conditional Logic
```python
def aggregate(metrics):
    # Both must pass at least 0.5, then take average
    if all(score >= 0.5 for score in metrics.values()):
        return sum(metrics.values()) / len(metrics)
    return 0.0
```

### Threshold-based
```python
def aggregate(metrics):
    # Count how many metrics pass threshold
    passed = sum(1 for score in metrics.values() if score >= 0.7)
    return passed / len(metrics)
```

## Gating

The aggregated metric can be used for gating just like any other metric:

```yaml
gate:
  metric_key: combined_score
  op: gte
  value: 0.6
```

## Running the Example

```bash
letta-evals run suite.yaml
```
