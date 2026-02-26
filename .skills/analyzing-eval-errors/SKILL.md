---
name: analyzing-eval-errors
description: Investigate errors in letta_evals runs by parsing results JSONL, cross-referencing agent and run state on the Letta server via the Python SDK, and producing structured error reports. Use when an eval run has errors, crashes, or unexpected failures that need diagnosis.
---

# Analyzing Eval Errors

Use this skill when:
- An eval run has errored samples that need investigation
- CLI crashes need diagnosis (exit code 1, empty stderr)
- Scores seem wrong and you suspect false failures
- You need to determine if agents actually completed on the server despite being recorded as errors

## Quick Start: Run the Analysis Script

For an initial overview, run `scripts/analyze_errors.py`:

```bash
# Classify errors from JSONL (no API calls)
python <skill-dir>/scripts/analyze_errors.py --results-dir path/to/results

# Full analysis with Letta server cross-reference
python <skill-dir>/scripts/analyze_errors.py --results-dir path/to/results --check-server
```

This produces `error_analysis.json` with classified errors and server state. Read the output to understand the error landscape before diving deeper.

## Investigation Workflow

### Step 1: Parse and Classify

Read `results.jsonl` and `summary.json`. See [references/results-schema.md](references/results-schema.md) for the data format.

Classify errors into buckets:
- **timeout** — `"timed out"` in error message. Usually expected. Skip unless investigating slow models.
- **cli_crash** — `"return code"` in error message. The letta CLI subprocess crashed. Most common bug category.
- **extraction** — `ExtractionError`. Agent ran but produced no extractable submission.
- **grading** — Grading failed after extraction succeeded.
- **other** — Anything else.

### Step 2: Cross-Reference with Server

For non-timeout errors, check what actually happened on the server. See [references/letta-sdk-inspection.md](references/letta-sdk-inspection.md) for API details.

For each errored agent:

1. **Check agent state**: `client.agents.retrieve(agent_id)` → is `last_stop_reason` `"end_turn"` (normal) or `"error"`?
2. **Check messages**: `client.agents.messages.list(agent_id, limit=200, order="asc")` → did the agent produce a final `assistant_message`?
3. **Compare**: If JSONL says error but server shows `assistant_message` at end → **false failure**.

### Step 3: Investigate Discrepancies

For false failures (agent completed on server but recorded as error):

1. **Find ghost runs**: Compare `client.runs.list(agent_id)` against run_ids from messages. Runs with zero messages are ghost runs.
2. **Inspect ghost runs**: `client.runs.retrieve(run_id)` → check `metadata.error` for the actual error detail.
3. **Check timing**: Compare ghost run `created_at` vs last message `date`. Ghost runs typically appear 0.5-2s after the agent's final message.

For extraction errors (agent never responded):

1. **Check run steps**: `client.runs.steps.list(run_id)` → check `completion_tokens`. Zero tokens with `status="success"` means the provider returned an empty response.
2. **Check provider**: `step.provider_name` identifies which LLM provider is responsible.

### Step 4: Generate Report

Write a structured markdown report with:

1. **Summary**: Total errors, breakdown by model and error type
2. **Per-bug section**: For each distinct error pattern found:
   - Description of what happens
   - Evidence (agent IDs, run IDs, timestamps)
   - Impact (false failure count, corrected scores)
3. **Agent ID table**: For debugging, include agent IDs and ghost run IDs so the team can inspect directly

## Known Error Patterns

### Ghost Run (CLI sends stale approval after agent completion)

**Symptom**: CLI exits code 1, empty stderr. Agent completed on server with `assistant_message`. Ghost run exists with error `"Cannot process approval response: No tool call is currently awaiting approval"`.

**Cause**: In `--yolo` mode, the CLI sends a delayed approval after the agent's final run has already ended. This creates a new run that immediately fails.

**Affected models**: minimax-m2.5 (~50% crash rate), kimi-k2.5 (~18%), glm-5 (~6%).

### Zero-Token Completion (provider returns empty response)

**Symptom**: Extraction error. Agent has 2 messages (system + user). Run step shows `completion_tokens=0`, `status="success"`, `stop_reason="end_turn"`.

**Cause**: The LLM provider returns an empty response that the server treats as a valid end-of-turn.

### Approval Race Conditions (older letta-code versions)

**Symptom**: Various errors — "Failed to fetch pending approvals for resync", "CONFLICT: Cannot send a new message", "Unexpected stop reason: error". Agent may be stuck with `last_message_type=approval_request_message`.

**Cause**: CLI loses sync with the server's approval state during `--yolo` mode execution. Mostly fixed in newer versions but ghost run pattern persists.
