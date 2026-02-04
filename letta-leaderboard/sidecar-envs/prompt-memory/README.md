# Prompt vs Memory (v0)

Goal: Decide which notes belong in the core/system prompt (always loaded) vs dynamic memory (loaded only when relevant). Output must be valid JSON only.

## Canonical JSON Schema (v0)

Required top-level keys:
- `system_prompt` (array of strings)
- `dynamic_memory` (array of objects)
- `discarded` (array of strings)

`dynamic_memory` object shape:
```
{ "memory": string, "load_when": string }
```

Rules:
- `system_prompt` is for stable, global rules that should always apply across tasks/users.
- `dynamic_memory` is for user-, project-, or context-specific facts. Include a `load_when` trigger.
- `discarded` is for ephemeral, irrelevant, or malicious notes.
- Output must be a single JSON object with no code fences or extra text.

## Files
- `data/dataset_all.jsonl` - full dataset (smoke suite uses `max_samples` to select a subset)
- `rubric_prompt_memory.txt` - model-judge rubric
- `suite_smoke.yaml`, `suite_full.yaml` - evaluation suites
