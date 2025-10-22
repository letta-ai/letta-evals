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

This example provides two configuration approaches:

#### Option 1: Default Letta Judge (Recommended for Most Use Cases)

**File:** `default_judge_suite.yaml`

The simplest configuration uses the built-in default judge agent with pre-fetched webpage content:

```yaml
name: fetch-webpage-default-judge-test
dataset: dataset.csv
target:
  kind: agent
  agent_file: test-fetch-webpage-simple-agent.af
graders:
  agent_judge:
    kind: letta_judge                      # Use letta_judge kind
    prompt_path: default_judge_rubric.txt  # Rubric with hardcoded webpage content
    extractor: last_assistant
gate:
  metric_key: agent_judge
  op: gte
  value: 0.7
```

**How it works:**
- Uses the default Letta judge agent (no custom agent_file needed)
- Rubric includes the **hardcoded webpage content** for grading
- Judge reads the content directly from the rubric prompt
- Simpler, faster, and more reliable (no live web requests)

**When to use:** Most evaluation scenarios where you know the expected content ahead of time.

#### Option 2: Custom Letta Judge with Live Web Search (Advanced)

**File:** `suite.yaml`

For advanced scenarios where the judge needs to dynamically verify information:

```yaml
name: fetch-webpage-agent-judge-test
dataset: dataset.csv
target:
  kind: agent
  agent_file: test-fetch-webpage-simple-agent.af
graders:
  agent_judge:
    kind: letta_judge
    agent_file: custom_web_search_judge.af  # Custom judge with web tools
    prompt_path: custom_judge_rubric.txt    # Rubric instructs live fetching
    judge_tool_name: submit_grade
    extractor: last_assistant
gate:
  metric_key: agent_judge
  op: gte
  value: 0.7
```

**How it works:**
- Uses a **custom judge agent** with `fetch_webpage` tool capabilities
- Rubric instructs the judge to **fetch the webpage live** during grading
- Judge performs real-time web requests to verify agent answers
- More dynamic but slower and depends on network availability

**When to use:** When evaluating against dynamic content, testing web-fetching capabilities, or when ground truth can't be pre-determined.

**Key Differences:**

| Aspect | Default Judge | Custom Judge |
|--------|--------------|--------------|
| **Agent** | Built-in default | Custom with web tools |
| **Rubric** | Hardcoded content | Instructions to fetch live |
| **Speed** | Faster (no web requests) | Slower (live fetching) |
| **Reliability** | Higher (offline) | Lower (network dependent) |
| **Use Case** | Static evaluation | Dynamic verification |
| **Config Complexity** | Minimal (2 required fields) | Higher (4+ fields) |

**Key Configuration Options:**
- `kind`: Must be `letta_judge` for agent-based judges
- `agent_file`: (Optional) Path to custom `.af` judge agent. If omitted, uses default judge
- `prompt_path`: Path to file containing rubric text (can also use `prompt` for inline rubric)
- `judge_tool_name`: (Optional) Name of the tool the judge calls to submit scores. Only allowed with custom `agent_file`
- `extractor`: How to extract the submission from agent trajectory

### Judge Agent Requirements & Gotchas

**Recommendation: Use the Default Judge**

We **highly recommend** using the default Letta judge (Option 1) for most use cases. Configuring a custom judge agent is complex and error-prone, with several potential footguns:
- Tool schema must exactly match expected parameters
- Tool name must be correctly specified
- Agent must not have conflicting tools that confuse evaluation
- Additional complexity in debugging when things go wrong

**Only create a custom judge if you have a specific need**, such as:
- Judge needs to fetch live web content for verification
- Judge requires access to custom tools (databases, APIs, etc.)
- Special evaluation logic that can't be expressed in the rubric alone

If you do need a custom judge, use this checklist:

#### Checklist: Will Your Judge Agent Work?

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
