# Simple Tool Grader Example

This example demonstrates basic tool-based grading using the built-in `contains` function with two different extractors.

## What This Example Shows

- Using tool graders for deterministic, fast evaluation
- Testing agent web-fetching capabilities
- Setting pass/fail gates with threshold values
- Comparing different extractors: `last_assistant` vs `tool_output`

## Two Suites, Two Evaluation Strategies

This example includes two separate suites that evaluate the same agent differently:

### 1. `last_assistant_suite.yaml` - Evaluating Agent Responses
Uses the `last_assistant` extractor to check if the agent's final response contains the correct answer. This tests whether the agent can successfully fetch webpage content AND communicate the answer properly to the user.

### 2. `tool_output_suite.yaml` - Evaluating Tool Outputs
Uses the `tool_output` extractor to check if the raw output from the `read_webpage_content` tool contains the correct answer. This tests whether the tool is successfully fetching and returning webpage content, independent of what the agent says.

## Key Takeaway

Tool graders like `contains` are ideal when you have clear ground truth answers and need fast, deterministic evaluation. Different extractors let you evaluate different parts of the agent's behavior - you can test tool functionality separately from the agent's ability to process and communicate results.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run either or both evaluations:
```bash
cd examples/simple-tool-grader

# evaluate agent's final responses
letta-evals run last_assistant_suite.yaml

# evaluate tool outputs directly
letta-evals run tool_output_suite.yaml
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

Each sample in `dataset.csv` has an `input` and `ground_truth`:

```csv
input,ground_truth
"Read `https://www.york.ac.uk/teaching/cws/wws/webpage1.html`. What program is mentioned for writing HTML code? Respond with the program name ONLY in brackets, e.g. {Word}.",{Notepad}
```

**Key points:**
- `input`: The prompt sent to the agent
- `ground_truth`: The expected answer to check for in the response
- Instructing the agent to format consistently (e.g., `{Notepad}`) makes grading more reliable

### Suite Configurations

#### `last_assistant_suite.yaml` - Agent Response Evaluation

```yaml
name: fetch-webpage-last-assistant-test
description: Test if agent's final response contains the correct answer from fetched webpage
dataset: dataset.csv
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
- `extractor: last_assistant` evaluates the final agent message
- Tests end-to-end behavior: tool calling + response generation
- `gate` requires ≥75% pass rate (3+ out of 5 samples must pass)

#### `tool_output_suite.yaml` - Tool Output Evaluation

```yaml
name: fetch-webpage-tool-output-test
description: Test if the tool output from read_webpage_content contains the correct answer
dataset: dataset.csv
target:
  kind: agent
  agent_file: test-fetch-webpage-simple-agent.af
  base_url: http://localhost:8283
graders:
  tool_output_check:
    kind: tool
    function: contains
    extractor: tool_output
    extractor_config:
      tool_name: read_webpage_content
gate:
  metric_key: tool_output_check
  op: gte
  value: 0.75
```

**Key points:**
- `extractor: tool_output` with `tool_name: read_webpage_content` evaluates raw tool output
- `extractor_config` specifies which tool's output to extract
- Tests tool functionality independently of agent's response formatting
- Useful for debugging: isolates whether issues are with the tool or the agent's processing
