# User Memory Update

Goal: Test an agent's ability to update its user memory block when the user shares new personal information. The agent must add new facts, remove outdated facts, and preserve unrelated facts.

## Grading

Each scenario provides:

- `initial_memory`: The starting user memory block
- `must_add`: Facts that must appear in the updated memory
- `must_remove`: Facts that must be removed from memory
- `must_preserve`: Unrelated facts that must remain intact

Score is the average of four components: memory was edited, required facts added, outdated facts removed, unrelated facts preserved.

## Files

- `data/dataset_memory.jsonl` - evaluation scenarios
- `custom_grader.py` - deterministic grader checking add/remove/preserve
- `custom_extractor.py` - extracts memory block state after agent run
- `agent_factory.py` - creates the Letta agent under test
- `suite.yaml` - evaluation suite config

## Results

### Per-Model Metrics

| Model | Samples | Attempted | Avg Score (Attempted) | Avg Score (Total) |
|---|---|---|---|---|
| anthropic/claude-haiku-4-5-20251001 | 5 | 5 | 1.00 | 1.00 |
| anthropic/claude-opus-4-5-20251101 | 5 | 5 | 0.90 | 0.90 |
| anthropic/claude-sonnet-4-5-20250929 | 5 | 5 | 0.90 | 0.90 |
| google_ai/gemini-3-flash-preview | 5 | 5 | 0.60 | 0.60 |
| google_ai/gemini-3-pro-preview | 5 | 5 | 0.75 | 0.75 |
| openai/gpt-4.1-nano | 5 | 5 | 0.50 | 0.50 |
| openai/gpt-5.1-codex-mini | 5 | 5 | 0.70 | 0.70 |
| openai/gpt-5.2 | 5 | 5 | 0.85 | 0.85 |
| zai/glm-4.6 | 5 | 5 | 0.95 | 0.95 |
| zai/glm-4.7 | 5 | 5 | 0.90 | 0.90 |

### Per-Model Usage

| Model | Prompt Tokens | Completion Tokens | Cost | Cached Input | Cache Write | Reasoning |
|---|---|---|---|---|---|---|
| anthropic/claude-haiku-4-5-20251001 | 64,613 | 3,980 | $0.0845 | - | 17,328 | - |
| anthropic/claude-opus-4-5-20251101 | 55,431 | 3,082 | $0.3542 | - | 8,613 | - |
| anthropic/claude-sonnet-4-5-20250929 | 63,649 | 3,353 | $0.2412 | 26,138 | 37,317 | - |
| google_ai/gemini-3-flash-preview | 30,263 | 1,333 | $0.0191 | 14,238 | - | 3,333 |
| google_ai/gemini-3-pro-preview | 32,216 | 927 | $0.0756 | - | - | 4,388 |
| openai/gpt-4.1-nano | 19,331 | 639 | $0.0022 | - | - | - |
| openai/gpt-5.1-codex-mini | 32,144 | 2,543 | $0.0131 | 9,600 | - | 1,280 |
| openai/gpt-5.2 | 27,722 | 1,339 | $0.0673 | 17,792 | - | - |
| zai/glm-4.6 | 45,523 | 3,455 | $0.0288 | 34,205 | - | 1,777 |
| zai/glm-4.7 | 38,640 | 3,058 | - | 20,358 | - | 1,313 |
