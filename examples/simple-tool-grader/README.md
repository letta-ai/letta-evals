# Simple Tool Grader Example

This example demonstrates basic tool-based grading using the built-in `contains` function.

## What This Example Shows

- Using tool graders for deterministic, fast evaluation
- Testing agent web-fetching capabilities
- Setting pass/fail gates with threshold values
- Using the `last_assistant` extractor to evaluate final responses

## Key Takeaway

Tool graders like `contains` are ideal when you have clear ground truth answers and need fast, deterministic evaluation. The `contains` function checks if the ground truth appears anywhere in the agent's response.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run the evaluation:
```bash
cd examples/simple-tool-grader
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

Each sample in `dataset.jsonl` has an `input` and `ground_truth`:

```json
{
  "input": "Read `https://www.york.ac.uk/teaching/cws/wws/webpage1.html`. What program is mentioned for writing HTML code? Respond with the program name ONLY in brackets, e.g. {Word}.",
  "ground_truth": "{Notepad}"
}
```

**Key points:**
- `input`: The prompt sent to the agent
- `ground_truth`: The expected answer to check for in the response
- Instructing the agent to format consistently (e.g., `{Notepad}`) makes grading more reliable

### Suite Configuration

The `suite.yaml` file configures the entire evaluation:

```yaml
name: fetch-webpage-test
description: Test if agent can fetch webpage content and extract information
dataset: dataset.jsonl
target:
  kind: agent
  agent_file: test-fetch-webpage-simple-agent.af
  base_url: http://localhost:8283
graders:
  contains_check:
    kind: tool
    function: contains
    extractor: last_assistant
gate:
  metric_key: contains_check
  op: gte
  value: 0.75
```

**Key points:**
- `target.kind: agent` with `agent_file` loads a pre-saved agent
- `graders.contains_check` uses the built-in `contains` function
- `extractor: last_assistant` evaluates the final agent message
- `gate` requires ≥75% pass rate (3+ out of 5 samples must pass)
