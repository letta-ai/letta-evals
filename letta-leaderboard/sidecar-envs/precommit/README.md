# Precommit Over Prose (v0)

Goal: Convert code-enforceable preferences into concrete config/pre-commit changes instead of storing them as prose memory. Output must be valid JSON only.

## Canonical JSON Schema (v0)

Required top-level keys:
- `config_changes` (array of objects)
- `memory_notes` (array of strings)
- `discarded` (array of strings)

`config_changes` object shape:
```
{ "file": string, "change": string }
```

Rules:
- Put enforceable preferences in `config_changes` (linters, formatters, pre-commit hooks, CI checks).
- Put subjective or non-enforceable preferences in `memory_notes`.
- If a preference is already enforced by existing config, list it under `discarded`.
- Output must be a single JSON object with no code fences or extra text.

## Files
- `data/dataset_all.jsonl` - full dataset (smoke suite uses `max_samples` to select a subset)
- `rubric_precommit.txt` - model-judge rubric
- `suite_smoke.yaml`, `suite_full.yaml` - evaluation suites
