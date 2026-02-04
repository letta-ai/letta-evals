# Session Handoff (v0)

Goal: produce a concise, structured `handoff.json` that captures what matters for the next session and discards noise. Output must be valid JSON only.

## Canonical JSON Schema (v0)

Required top-level keys:
- `summary` (string)
- `decisions` (array of strings)
- `open_todos` (array of objects)
- `blockers` (array of strings)
- `next_steps` (array of strings)
- `artifacts` (array of objects)
- `discarded` (array of strings)

`open_todos` object shape:
```
{ "task": string, "owner": string | null, "priority": "high"|"med"|"low", "status": "open"|"in_progress" }
```

`artifacts` object shape:
```
{ "type": string, "path": string, "description": string }
```

Rules:
- Include all key decisions and unresolved tasks.
- Keep `summary` concise (2–5 sentences).
- `discarded` should list explicitly irrelevant or outdated details that should not carry forward.
- Output must be a single JSON object with no code fences or extra text.

## Files
- `data/dataset_all.jsonl` – full dataset (smoke suite uses `max_samples` to select a subset)
- `rubric_handoff.txt` – model-judge rubric
- `suite_smoke.yaml`, `suite_full.yaml` – evaluation suites
