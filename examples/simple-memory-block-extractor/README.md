# Simple Memory Block Extractor Example

This example demonstrates extracting and evaluating agent memory blocks using the `memory_block` extractor.

## What This Example Shows

- Using the `memory_block` extractor to evaluate agent memory contents
- Testing if agents correctly store information in specific memory blocks
- Validating memory persistence and recall capabilities
- Grading based on memory state rather than conversation output

## Key Takeaway

The `memory_block` extractor allows you to evaluate what the agent *remembers* in its memory blocks, not just what it *says*. This is essential for testing memory management, information persistence, and long-term agent state. The extractor automatically retrieves agent state only when needed, keeping evaluation efficient.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run the evaluation:
```bash
cd examples/simple-memory-block-extractor
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
  "input": "Please remember that I like bananas.",
  "ground_truth": "bananas"
}
```

**Key points:**
- `input`: The instruction to store information in memory
- `ground_truth`: The expected value to find in the memory block
- The agent should process the input and update its memory accordingly

### Suite Configuration

The `suite.yaml` file configures the memory block evaluation:

```yaml
name: remember-fruit-preferences-test
description: Test if agent can remember a user's fruit preferences
dataset: dataset.jsonl
target:
  kind: letta_agent
  agent_file: test-fruit-pref-agent.af
  base_url: http://localhost:8283
graders:
  memory_check:
    kind: tool
    function: contains
    extractor: memory_block
    extractor_config:
      block_label: "fruit_preferences"
gate:
  metric_key: memory_check
  op: gte
  value: 0.33
```

**Key points:**
- `extractor: memory_block` extracts content from agent memory blocks
- `extractor_config.block_label` specifies which memory block to read (e.g., `"fruit_preferences"`)
- `function: contains` checks if the ground truth appears in the memory block
- Agent state is automatically retrieved only when this extractor is used

## Memory Block Extractor

The `memory_block` extractor reads from the agent's core memory after the conversation:

```yaml
extractor: memory_block
extractor_config:
  block_label: "fruit_preferences"  # The label of the memory block to extract
```

This extractor:
- Retrieves the agent's final state after all messages are processed
- Extracts the content of the specified memory block by label
- Returns the block's value as a string for grading
- Raises an error if the agent state is unavailable

**Common memory block labels:**
- `human`: Information about the user
- `persona`: Information about the agent's personality
- Custom blocks: Any blocks defined in your agent configuration (e.g., `fruit_preferences`)

## Evaluating Multiple Memory Blocks

To evaluate multiple memory blocks, define multiple graders:

```yaml
graders:
  human_memory:
    kind: tool
    function: contains
    extractor: memory_block
    extractor_config:
      block_label: "human"

  persona_memory:
    kind: tool
    function: contains
    extractor: memory_block
    extractor_config:
      block_label: "persona"
```

Each grader will produce a separate metric, and agent state is retrieved only once per sample.

## When to Use Memory Block Extraction

Use `memory_block` extractor when you want to test:
- **Memory updates**: Did the agent correctly store information?
- **Information persistence**: Does the agent remember facts across turns?
- **Memory management**: Does the agent organize information appropriately?
- **State consistency**: Is the agent's internal state correct?

Use traditional extractors (`last_assistant`, `all_assistant`) when you want to test:
- **Response content**: What the agent says in conversation
- **Tool usage**: What functions the agent calls
- **Communication style**: How the agent phrases responses
