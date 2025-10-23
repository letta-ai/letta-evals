# Multiturn Memory Block Extractor Example

This example demonstrates extracting and evaluating agent memory blocks across multiturn conversations using the `memory_block` extractor.

## What This Example Shows

- Using multiturn conversations to test memory updates and corrections
- Testing if agents correctly update information in specific memory blocks when corrected
- Validating memory persistence and update capabilities across conversation turns
- Grading based on final memory state after multiturn interactions
- Evaluating agent's ability to handle preference changes

## Key Takeaway

The `memory_block` extractor combined with multiturn inputs allows you to evaluate how agents handle memory updates and corrections. This is essential for testing whether agents properly update their memory when users correct themselves, ensuring the latest information is retained. The extractor automatically retrieves the final agent state after all conversation turns are processed.

## Running This Example

### Local Setup (Default)

Start your local Letta server:
```bash
letta server
```

Then run the evaluation:
```bash
cd examples/multiturn-memory-block-extractor
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

Each sample in `dataset.jsonl` has a multiturn `input` (list of strings) and `ground_truth`:

```json
{
  "input": [
    "Please remember that I like bananas.",
    "Actually, sorry, I meant I like apples."
  ],
  "ground_truth": "apples"
}
```

**Key points:**
- `input`: A list of messages representing a multiturn conversation
- First turn: Initial preference statement
- Second turn: Correction to a different preference
- `ground_truth`: The final (corrected) value expected in the memory block
- The agent should process both turns and update its memory to reflect the latest preference

### Suite Configuration

The `suite.yaml` file configures the memory block evaluation:

```yaml
name: remember-fruit-preferences-test
description: Test if agent can remember a user's fruit preferences
dataset: dataset.jsonl
target:
  kind: agent
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
- `extractor: memory_block` extracts content from agent memory blocks after all conversation turns
- `extractor_config.block_label` specifies which memory block to read (e.g., `"fruit_preferences"`)
- `function: contains` checks if the final ground truth (corrected preference) appears in the memory block
- Agent state is automatically retrieved after processing all multiturn messages
- The framework handles multiturn inputs natively - no special configuration needed

## Memory Block Extractor with Multiturn

The `memory_block` extractor reads from the agent's core memory after all conversation turns:

```yaml
extractor: memory_block
extractor_config:
  block_label: "fruit_preferences"  # The label of the memory block to extract
```

This extractor:
- Retrieves the agent's final state after all multiturn messages are processed
- Extracts the content of the specified memory block by label
- Returns the block's value as a string for grading
- Tests whether the agent correctly updated memory in response to the correction
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

## When to Use Memory Block Extraction with Multiturn

Use `memory_block` extractor with multiturn inputs when you want to test:
- **Memory updates and corrections**: Did the agent correctly update stored information when corrected?
- **Information persistence**: Does the agent remember and update facts across turns?
- **Memory management**: Does the agent properly replace or append information?
- **State consistency**: Is the agent's final internal state correct after multiple interactions?
- **Preference changes**: Does the agent track the latest user preferences accurately?

Use traditional extractors (`last_assistant`, `all_assistant`, `last_turn`) when you want to test:
- **Response content**: What the agent says in conversation
- **Tool usage**: What functions the agent calls across turns
- **Communication style**: How the agent phrases responses
- **Per-turn behavior**: Agent responses at specific conversation stages
