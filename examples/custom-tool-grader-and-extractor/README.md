# Custom Tool Grader and Extractor Example

This example demonstrates how to write custom graders and extractors using Python decorators.

## What This Example Shows

- Creating custom extractors with the `@extractor` decorator to evaluate agent memory operations
- Creating custom graders with the `@grader` decorator for domain-specific validation
- Inspecting tool calls within agent trajectories (not just final responses)
- Validating internal agent behavior beyond message content

## Key Takeaway

Custom extractors and graders allow you to evaluate any part of the agent's behavior. In this example, we check if the agent correctly stores fruit preferences in memory by extracting `memory_insert` tool calls and validating their arguments.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run the evaluation:
```bash
cd examples/custom-tool-grader-and-extractor
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

## Code Walkthrough

### Custom Extractor (`custom_evaluators.py:memory_insert_extractor`)

The custom extractor inspects the agent's trajectory to find `memory_insert` tool calls:

```python
@extractor
def memory_insert_extractor(trajectory: List[List[LettaMessageUnion]], config: dict) -> str:
    """Extract memory_insert tool calls from trajectory."""
    for turn in trajectory:
        for message in turn:
            if isinstance(message, ToolCallMessage) and message.tool_call.name == "memory_insert":
                return message.tool_call.arguments
    return "{}"
```

**Key points:**
- The `@extractor` decorator registers this function with letta-evals
- `trajectory` is a list of conversation turns, each containing messages
- We look for `ToolCallMessage` with the specific tool name
- Returns the tool call arguments as a JSON string for grading

### Custom Grader (`custom_evaluators.py:grade_fruit_preference`)

The custom grader validates the extracted memory insert arguments:

```python
@grader
def grade_fruit_preference(sample: Sample, submission: str) -> GradeResult:
    """Grade if the fruit preference was correctly stored in memory."""
    try:
        args = json.loads(submission)
    except json.JSONDecodeError:
        return GradeResult(score=0.0, rationale="No valid memory_insert tool call found")

    # check label is user_fruit_preferences
    label = args.get("label", "")
    if label != "user_fruit_preferences":
        return GradeResult(
            score=0.0, rationale=f"Wrong memory block label: expected 'user_fruit_preferences', got '{label}'"
        )

    # check fruit name is in new_str
    fruit = sample.ground_truth.lower()
    new_str = args.get("new_str", "").lower()

    if fruit not in new_str:
        return GradeResult(score=0.0, rationale=f"Fruit '{fruit}' not found in new_str")

    return GradeResult(score=1.0, rationale="Fruit preference correctly stored")
```

**Key points:**
- The `@grader` decorator registers this function with letta-evals
- Takes `sample` (containing ground truth) and `submission` (extractor output)
- Returns `GradeResult` with `score` (0.0-1.0) and `rationale` (explanation)
- Validates both the memory block label and that the fruit name appears in the content

### Suite Configuration

In `suite.yaml`, reference custom functions using the `file:function` syntax:

```yaml
graders:
  preference_check:
    kind: tool
    function: custom_evaluators.py:grade_fruit_preference
    extractor: custom_evaluators.py:memory_insert_extractor
```
