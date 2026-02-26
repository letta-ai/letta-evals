# Letta SDK Agent & Run Inspection

API reference for inspecting agent state, messages, and runs via the Letta Python SDK.

## Setup

```python
import os
from letta_client import Letta

client = Letta(api_key=os.environ.get("LETTA_API_KEY"))
```

## Agent State

```python
agent = client.agents.retrieve(agent_id="agent-xxxx")
```

Key fields:
- `agent.last_stop_reason` — `"end_turn"` (normal), `"error"`, `"requires_approval"`, etc.
- `agent.last_run_completion` — datetime of last run completion
- `agent.last_run_duration_ms` — duration of last run

**Diagnostic pattern**: If `last_stop_reason == "error"` but messages show a clean `assistant_message` at the end, there's a ghost run.

## Messages

```python
messages = client.agents.messages.list(
    agent_id="agent-xxxx",
    limit=200,        # max per page
    order="asc",      # chronological
)
items = messages.items  # list of message objects
```

Each message has:
- `message_type` — `"system_message"`, `"user_message"`, `"reasoning_message"`, `"assistant_message"`, `"approval_request_message"`, `"approval_response_message"`, `"tool_return_message"`, `"tool_call_message"`
- `run_id` — which run produced this message
- `date` — timestamp

### Common checks

```python
# Did the agent complete?
last_msg = messages.items[-1]
completed = last_msg.message_type == "assistant_message"

# Count message types
from collections import Counter
types = Counter(m.message_type for m in messages.items)

# Check approval balance
approvals_req = types.get("approval_request_message", 0)
approvals_resp = types.get("approval_response_message", 0)
balanced = approvals_req == approvals_resp

# Collect unique run_ids from messages
msg_run_ids = set()
for m in messages.items:
    rid = getattr(m, "run_id", None)
    if rid:
        msg_run_ids.add(rid)
```

## Runs

### List runs for an agent

```python
runs = client.runs.list(agent_id="agent-xxxx", limit=100)
```

### Retrieve a specific run

```python
run = client.runs.retrieve(run_id="run-xxxx")
```

Key fields:
- `run.id` — run ID
- `run.status` — `"completed"`, `"failed"`, `"running"`
- `run.stop_reason` — `"end_turn"`, `"error"`, etc.
- `run.created_at`, `run.completed_at` — timing
- `run.metadata` — dict, may contain `run_type` and `error` details

For failed runs, error details are in metadata:
```python
run.metadata.get("error", {})
# Example:
# {
#     "error_type": "internal_error",
#     "message": "An unknown error occurred with the LLM streaming request.",
#     "detail": "Cannot process approval response: No tool call is currently awaiting approval."
# }
```

### Find ghost runs

Ghost runs are runs that exist for an agent but produced zero messages. They indicate the CLI sent a spurious request after the agent finished.

```python
# Get all run_ids that have messages
msg_run_ids = set()
for m in messages.items:
    rid = getattr(m, "run_id", None)
    if rid:
        msg_run_ids.add(rid)

# Find runs NOT in the message run_ids
runs = client.runs.list(agent_id=agent_id, limit=100)
ghost_runs = [r for r in runs.items if r.id not in msg_run_ids]
```

## Run Steps

Inspect individual LLM calls within a run:

```python
steps = client.runs.steps.list(run_id="run-xxxx", limit=50)
step = steps.items[0]
```

Key fields:
- `step.completion_tokens`, `step.prompt_tokens`, `step.total_tokens`
- `step.status` — `"success"`, `"error"`
- `step.stop_reason` — `"end_turn"`, `"error"`, etc.
- `step.provider_name` — `"zai"`, `"openai"`, `"anthropic"`, etc.
- `step.messages` — messages produced by this step (empty list if none)

**Diagnostic pattern**: A step with `completion_tokens == 0`, `status == "success"`, and `stop_reason == "end_turn"` means the provider returned an empty response that the server treated as valid.

## Run Messages

List messages produced by a specific run:

```python
run_msgs = client.runs.messages.list(run_id="run-xxxx", limit=50)
```
