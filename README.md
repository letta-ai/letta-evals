# Letta Evals Kit

Evaluation framework for testing Letta AI agents.

<p align="center">
  <img src="docs/assets/evaluation-progress.png" alt="Letta Evals Kit running an evaluation suite with real-time progress tracking" width="800">
</p>

## Project Structure

```
letta_evals/
├── cli.py              # CLI entry point
├── runner/             # Evaluation engine
├── graders/            # Grading implementations
├── datasets/           # Dataset loaders
├── targets/            # Evaluation targets
├── models.py           # Data models
└── visualization/      # Progress display

examples/               # Example suites and datasets
```

## Installation

### Recommended: Using UV

```bash
# install uv if you haven't already
pip install uv

# sync dependencies (run inside your virtual environment)
uv sync --extra dev
```

## Quick Start

1. **Create a test dataset** (`dataset.jsonl`):
```jsonl
{"input": "What's the capital of France?", "ground_truth": "Paris"}
{"input": "Calculate 2+2", "ground_truth": "4"}
```

2. **Write a suite configuration** (`suite.yaml`):
```yaml
name: my-eval-suite
dataset: dataset.jsonl
target:
  kind: agent
  agent_file: my_agent.af  # or use agent_id for existing agents
  base_url: http://localhost:8283
graders:
  quality:
    kind: tool
    function: contains  # or exact_match
    extractor: last_assistant
gate:
  metric_key: quality
  op: gte
  value: 0.75  # require 75% pass threshold
```

3. **Run the evaluation**:
```bash
letta-evals run suite.yaml
```

## Core Concepts

### Suites
YAML configuration files that define your evaluation parameters. Each suite specifies the dataset, target agent, grading method, and pass/fail criteria.

### Datasets
JSONL files where each line contains a test sample with:
- `input`: The prompt to send to the agent
- `ground_truth`: The expected response (for tool graders)
- `metadata`: Optional additional context

### Targets
What you're evaluating. Currently supports:
- **Letta agents**: Via `agent_file` (.af files) or existing `agent_id`

### Graders
How responses are scored (define one or many under `graders`):
- Tool graders: built-ins (`exact_match`, `contains`, etc.) or custom Python functions
- Rubric graders: LLM judges with custom prompts (`prompt_path`, `model`, `provider`)

### Gates
Pass/fail thresholds for your evaluation:
- `metric_key`: Which grader to gate on (key in `graders`)
- `metric`: Aggregate to compare (`avg_score` or `accuracy`)
- `op`: Comparison operator (`gte`, `gt`, `lte`, `lt`, `eq`)
- `value`: Threshold value (decimal for `avg_score`, percent for `accuracy`)

## CLI Commands

```bash
# run an evaluation suite (shows progress by default)
letta-evals run suite.yaml

# save outputs to directory
# header.json for headers, summary.json for results summary, results.jsonl for per-instance results
letta-evals run suite.yaml --output results

# quiet mode (only show pass/fail)
letta-evals run suite.yaml --quiet

# validate suite configuration
letta-evals validate suite.yaml

# list available components
letta-evals list-extractors
letta-evals list-graders
```

## Example: Multi-Metric Suite

- Path: `examples/simple-rubric-grader/suite.two-metrics.yaml`
- Two graders: `quality` (rubric using `rubric.txt`) and `ascii_only` (tool `ascii_printable_only`)
- Gate: `metric_key: quality`, `metric: avg_score`, `op: gte`, `value: 0.6`

Run:

```
letta-evals run examples/simple-rubric-grader/suite.two-metrics.yaml
```

## Configuration

### Suite YAML Structure

```yaml
name: suite-name
description: Optional description
dataset: path/to/dataset.jsonl
max_samples: 100  # optional: limit samples
sample_tags: [tag1, tag2]  # optional: filter by tags

target:
  kind: agent
  agent_file: path/to/agent.af  # or agent_id: existing-id
  base_url: http://localhost:8283

graders:
  my_metric:
    kind: tool  # or rubric
    function: contains  # for tool graders
    extractor: last_assistant  # what to extract from response
    # for rubric graders:
    # prompt_path: path/to/rubric.txt
    # model: gpt-4.1
    # provider: openai

gate:
  metric_key: my_metric
  op: gte  # gte, gt, lte, lt, eq
  value: 0.8
```

### Extractors

Control what part of the agent response gets graded:
- `last_assistant`: Final assistant message
- `all_messages`: All assistant messages
- `tool_calls`: Tool invocations
- `first_assistant`: First assistant message
- Custom JSONPath expressions

### Built-in Grader Functions

- `exact_match`: Exact string matching
- `contains`: Check if response contains expected text
- Custom functions can be registered via Python

## Development

### Linting

```bash
# check for issues
ruff check .

# auto-format code
ruff format .
```

### Testing

```bash
# run tests
pytest

# run specific tests
pytest -k test_name
```
