# Multi-Grader Gate Example

This example demonstrates **multi-grader gates** that combine multiple evaluation metrics with sophisticated pass/fail logic.

## What This Example Shows

- **Logical gates** with AND/OR operators
- **Weighted average gates** for combining metrics with different importance
- Different **aggregation functions** (avg_score, accuracy, min, max, percentiles)
- Custom **pass thresholds** for accuracy calculations
- Combining rubric grading (LLM judge) with tool grading (deterministic functions)

## Key Takeaway

Multi-grader gates enable sophisticated evaluation criteria that go beyond single metrics. You can:
- Require multiple conditions to pass simultaneously (AND logic)
- Pass if any condition passes (OR logic)
- Combine metrics with weighted importance
- Use different aggregation functions per metric

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Set your OpenAI API key (required for rubric grader):
```bash
export OPENAI_API_KEY=your-key
```

Then run either suite:
```bash
cd examples/multi-grader-gate
letta-evals run suite.logical-and.yaml
letta-evals run suite.weighted-average.yaml
```

### Letta Cloud Setup

```bash
export LETTA_API_KEY=your-api-key
export LETTA_PROJECT_ID=your-project-id
export OPENAI_API_KEY=your-openai-key
```

Update `base_url` in the suite files:
```yaml
target:
  base_url: https://api.letta.com/
```

## Gate Types

### 1. Simple Gate

Gates on a single metric:

```yaml
gate:
  kind: simple
  metric_key: quality
  aggregation: avg_score
  op: gte
  value: 0.6
```

### 2. Logical Gate (`suite.logical-and.yaml`)

Combines multiple conditions with AND/OR logic. All conditions must pass for AND, at least one must pass for OR.

**Example (AND):**
```yaml
gate:
  kind: logical
  operator: and
  conditions:
    - metric_key: quality
      aggregation: avg_score
      op: gte
      value: 0.6
    - metric_key: ascii_only
      aggregation: accuracy
      pass_threshold: 1.0
      op: gte
      value: 60
```

This passes only if:
- **Quality (avg_score) >= 0.6**: Average quality score across all samples >= 0.6
- **ASCII-only (accuracy) >= 60%**: At least 60% of samples have perfect ASCII scores (1.0)

**Example (OR):**
```yaml
gate:
  kind: logical
  operator: or
  conditions:
    - metric_key: exact_match
      aggregation: accuracy
      op: eq
      value: 100
    - metric_key: quality
      aggregation: avg_score
      op: gte
      value: 0.9
```

This passes if EITHER exact matches are 100% OR quality is very high (>= 0.9).

**Nested logical conditions:**

Logical gates can be nested for complex logic like "(A AND B) OR C":

```yaml
gate:
  kind: logical
  operator: or
  conditions:
    - kind: logical
      operator: and
      conditions:
        - metric_key: accuracy
          aggregation: avg_score
          op: gte
          value: 0.8
        - metric_key: coherence
          aggregation: min
          op: gte
          value: 0.6
    - metric_key: fallback_check
      aggregation: accuracy
      op: eq
      value: 100
```

### 3. Weighted Average Gate (`suite.weighted-average.yaml`)

Combines multiple metrics with configurable weights. The weighted average is compared against a threshold.

```yaml
gate:
  kind: weighted_average
  aggregation: avg_score
  weights:
    quality: 0.7
    ascii_only: 0.3
  op: gte
  value: 0.75
```

This computes: `0.7 * avg(quality) + 0.3 * avg(ascii_only)` and passes if the result >= 0.75.

**Key points:**
- Weights are automatically normalized (don't need to sum to 1.0)
- The same aggregation function is applied to all metrics before weighting
- Useful for expressing relative importance of different quality dimensions

## Aggregation Functions

Each condition in a gate can use different aggregation functions:

| Aggregation | Description | Use Case |
|-------------|-------------|----------|
| `avg_score` | Average score across all samples | Overall quality metrics |
| `accuracy` | Percentage of samples passing `pass_threshold` | Binary pass/fail checks |
| `min` | Minimum score across all samples | Worst-case requirements |
| `max` | Maximum score across all samples | Best-case requirements |
| `median` | Median score | Robust to outliers |
| `p50` | 50th percentile (same as median) | Robust to outliers |
| `p95` | 95th percentile | High-end performance |
| `p99` | 99th percentile | Tail behavior |

**Examples:**

```yaml
# average quality must be high
- metric_key: quality
  aggregation: avg_score
  op: gte
  value: 0.8

# worst-case safety must be acceptable
- metric_key: safety
  aggregation: min
  op: gte
  value: 0.5

# at least 90% of samples must pass
- metric_key: format_check
  aggregation: accuracy
  pass_threshold: 1.0
  op: gte
  value: 90

# 95th percentile latency must be acceptable
- metric_key: response_time
  aggregation: p95
  op: lte
  value: 5.0
```

## Pass Threshold for Accuracy

When using `aggregation: accuracy`, the `pass_threshold` parameter defines what score counts as "passing" for a sample:

```yaml
- metric_key: ascii_only
  aggregation: accuracy
  pass_threshold: 0.8  # sample passes if score >= 0.8
  op: gte
  value: 90  # at least 90% of samples must pass
```

**How it works:**
1. Each sample gets a score from the grader (e.g., 0.0 to 1.0)
2. A sample "passes" if its score >= `pass_threshold` (defaults to 1.0)
3. Accuracy = (passed samples / total samples) * 100
4. The gate condition checks if accuracy meets the threshold

**Example:**
- Scores: [1.0, 0.9, 0.85, 0.7, 0.6]
- `pass_threshold: 0.8`
- Passed samples: 3 (1.0, 0.9, 0.85)
- Accuracy: 3/5 = 60%

## Comparison Operators

All gate conditions support these operators:

| Operator | Symbol | Description |
|----------|--------|-------------|
| `gte` | ≥ | Greater than or equal |
| `gt` | > | Greater than |
| `lte` | ≤ | Less than or equal |
| `lt` | < | Less than |
| `eq` | = | Equal |

## Graders in This Example

### Quality (Rubric Grader)

LLM-as-judge using detailed scoring rubric in `rubric.txt`:
- Instruction compliance (0.30)
- Requested item match (0.30)
- Recognizability (0.20)
- Craftsmanship (0.10)
- Effort (0.10)

Scores sum to 1.0 for normalized evaluation.

### ASCII-only (Tool Grader)

Deterministic check using built-in `ascii_printable_only` function:
- Returns 1.0 if all characters are printable ASCII
- Returns 0.0 if non-ASCII characters are found

## Suite Variants

### `suite.logical-and.yaml`

Demonstrates **logical AND gate** requiring:
- High quality (avg_score >= 0.6)
- Valid ASCII in most samples (accuracy >= 60%)

**Use case:** Strict requirements where multiple independent criteria must all pass.

### `suite.weighted-average.yaml`

Demonstrates **weighted average gate** combining:
- Quality (70% weight)
- ASCII validation (30% weight)
- Overall threshold >= 0.75

**Use case:** Balancing multiple quality dimensions with different importance levels.

## Design Guidelines

**When to use Logical Gates:**
- Multiple independent requirements must all pass (AND)
- Any of several alternatives can satisfy the requirement (OR)
- Complex boolean logic (nested conditions)
- Example: "Safety must be perfect AND quality must be good"

**When to use Weighted Average:**
- Trading off between multiple quality dimensions
- Some metrics are more important than others
- Continuous quality assessment with balanced criteria
- Example: "60% quality + 40% style must average to 0.7"

**When to use Simple Gates:**
- Single metric is sufficient
- Clear threshold for pass/fail
- Example: "Average quality must be >= 0.6"

## Common Patterns

### Strict Safety + Flexible Quality
```yaml
gate:
  kind: logical
  operator: and
  conditions:
    - metric_key: safety
      aggregation: accuracy
      op: eq
      value: 100  # all samples must pass safety
    - metric_key: quality
      aggregation: avg_score
      op: gte
      value: 0.6  # quality can be moderate
```

### Balanced Multi-Dimensional Scoring
```yaml
gate:
  kind: weighted_average
  aggregation: avg_score
  weights:
    accuracy: 0.4
    coherence: 0.3
    style: 0.2
    format: 0.1
  op: gte
  value: 0.75
```

### Fallback Criteria
```yaml
gate:
  kind: logical
  operator: or
  conditions:
    - metric_key: exact_match
      aggregation: accuracy
      op: eq
      value: 100  # prefer exact matches
    - metric_key: semantic_similarity
      aggregation: avg_score
      op: gte
      value: 0.9  # but accept high semantic similarity
```

### Worst-Case Guarantees
```yaml
gate:
  kind: logical
  operator: and
  conditions:
    - metric_key: quality
      aggregation: avg_score
      op: gte
      value: 0.8  # average must be high
    - metric_key: quality
      aggregation: min
      op: gte
      value: 0.5  # worst case must be acceptable
```
