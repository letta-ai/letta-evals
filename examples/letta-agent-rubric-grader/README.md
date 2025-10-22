# Letta Agent as Judge Example

This example demonstrates using a Letta agent as a judge for rubric-based grading.

## What This Example Shows

- Using a Letta agent as an LLM judge instead of direct API calls
- Defining custom rubric criteria for evaluation
- Validating judge agent tool schemas on initialization
- Testing agent web-fetching capabilities with nuanced evaluation

## How It Works

Instead of calling an LLM API directly (like the standard rubric grader), this approach:
1. Loads a judge agent from a `.af` file
2. Sends the formatted rubric prompt to the judge agent
3. The judge agent evaluates the submission and calls a `submit_grade` tool with score and rationale
4. Extracts the score from the tool call and uses it for grading

## Key Components

### Judge Agent (`judge.af`)
A Letta agent configured with:
- A `submit_grade(score: float, rationale: str)` tool for submitting evaluations
- Access to web fetching if needed for verification
- System instructions for evaluation behavior

### Rubric (`rubric.txt`)
Defines evaluation criteria:
- Correctness: Does the response contain accurate information?
- Format: Is the response properly formatted as requested?
- Completeness: Does it fully address the question?

### Dataset (`dataset.csv`)
Contains test cases where agents must fetch webpage content and answer questions.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run the evaluation:
```bash
cd examples/letta-agent-rubric-grader
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

### Suite Configuration

```yaml
name: fetch-webpage-agent-judge-test
description: Test agent responses using a Letta agent as judge with rubric grading
dataset: dataset.csv
target:
  kind: agent
  agent_file: test-fetch-webpage-simple-agent.af
  base_url: http://localhost:8283
graders:
  agent_judge:
    kind: rubric
    agent_file: judge.af              # Judge agent with submit_grade tool
    prompt_path: rubric.txt           # Rubric criteria for evaluation
    judge_tool_name: submit_grade     # Tool the judge uses to submit scores
    extractor: last_assistant         # Extract agent's final response
gate:
  metric_key: agent_judge
  op: gte
  value: 0.75                         # Pass if avg score ≥ 0.75
```

**Key Configuration Options:**
- `agent_file`: Path to `.af` file containing the judge agent
- `prompt_path`: Path to file containing rubric text (can also use `prompt` for inline rubric)
- `judge_tool_name`: Name of the tool the judge calls to submit scores (default: `submit_grade`)
- `extractor`: How to extract the submission from agent trajectory

### Judge Tool Requirements

The judge agent must have a tool with this signature:
```python
def submit_grade(score: float, rationale: str) -> dict:
    """Submit evaluation with score (0.0-1.0) and rationale."""
    pass
```

The framework validates this on initialization and will fail fast if:
- The specified tool doesn't exist in the agent file
- The tool is missing `score` or `rationale` parameters

### Dataset Format

```csv
input
"Read `https://www.york.ac.uk/teaching/cws/wws/webpage1.html`. What program is mentioned for writing HTML code? Respond with the program name ONLY in brackets, e.g. {Word}."
```

**Note:** Ground truth is optional for agent judges since the judge agent can verify answers independently (e.g., by fetching the webpage itself).
