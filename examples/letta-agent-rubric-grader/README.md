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

### Judge Agent Requirements & Gotchas

#### ✅ Checklist: Will Your Judge Agent Work?

Use this checklist to verify your judge agent is properly configured:

- [ ] **Tool exists**: Agent has a tool with the name specified in `judge_tool_name` (default: `submit_grade`)
- [ ] **Tool parameters**: The tool has BOTH `score: float` and `rationale: str` parameters
- [ ] **Tool is callable**: The tool is not disabled or requires-approval-only
- [ ] **Agent system prompt**: Agent understands it's an evaluator (optional but recommended)
- [ ] **No conflicting tools**: Agent doesn't have other tools that might confuse it into answering questions instead of judging

#### Required Tool Signature

The judge agent **must** have a tool with this exact parameter signature:

```python
def submit_grade(score: float, rationale: str) -> dict:
    """
    Submit an evaluation grade for an agent's response.

    Args:
        score: A float between 0.0 (complete failure) and 1.0 (perfect)
        rationale: Explanation of why this score was given

    Returns:
        dict: Confirmation of grade submission
    """
    return {
        "status": "success",
        "grade": {"score": score, "rationale": rationale}
    }
```

**Important**: The parameter names must be exactly `score` and `rationale`. The framework validates this on initialization.


### Validation on Initialization

The framework validates your judge agent **before** running any evaluations. If validation fails, you'll get a clear error message:

```bash
ValueError: Judge tool 'submit_grade' not found in agent file judge.af.
Available tools: ['fetch_webpage', 'search_documents']
```

This fail-fast approach saves time by catching configuration errors immediately.

### Dataset Format

```csv
input
"Read `https://www.york.ac.uk/teaching/cws/wws/webpage1.html`. What program is mentioned for writing HTML code? Respond with the program name ONLY in brackets, e.g. {Word}."
```

**Note:** Ground truth is optional for agent judges since the judge agent can verify answers independently (e.g., by fetching the webpage itself).
