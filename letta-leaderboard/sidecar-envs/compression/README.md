# Context Compression (v0)

Goal: Compress a long context into a short summary that preserves task-critical information. Output must be valid JSON only.

## Canonical JSON Schema (v0)

Required top-level keys:
- `summary` (string)
- `must_keep` (array of strings)
- `constraints` (array of strings)
- `open_questions` (array of strings)
- `safe_to_drop` (array of strings)

Rules:
- Keep `summary` concise (2-4 sentences).
- `must_keep` should include key decisions, artifacts, paths, deadlines, and critical facts.
- `constraints` should list hard requirements (deadlines, budgets, performance targets).
- `open_questions` should include unresolved questions that block progress.
- `safe_to_drop` should list irrelevant or superseded details.
- Output must be a single JSON object with no code fences or extra text.

## Files
- `data/dataset_all.jsonl` - full dataset (smoke suite uses `max_samples` to select a subset)
- `rubric_context_compression.txt` - model-judge rubric
- `suite_smoke.yaml`, `suite_full.yaml` - evaluation suites
