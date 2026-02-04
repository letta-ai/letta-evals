# User Model Maintenance (v0)

Goal: Update a canonical `user_model.json` from new conversation facts. Output must be valid JSON only.

## Canonical JSON Schema (v0)

Required top-level keys:
- `user_id` (string)
- `name` (string)
- `current` (object)
- `history` (array)
- `last_updated` (YYYY-MM-DD string)

`current` required keys:
- `employer` (string)
- `role` (string)
- `location` (object with `city`, `country`, `timezone`)
- `preferences` (object with `communication_style` array, `tools` array, `notes` string)

`history` entries (when a current fact is replaced):
```
{
  "field": "employer" | "role" | "location" | "timezone",
  "old_value": "...",
  "new_value": "..."
}
```

Rules:
- If a fact changes, update `current` and add a `history` entry.
- Do NOT keep stale facts in `current`.
- Preferences updates overwrite previous values (no history needed).
- Output must be a single JSON object with no code fences or extra text.

## Files
- `data/dataset_all.jsonl` – full dataset (smoke suite uses `max_samples` to select a subset)
- `rubric_user_model.txt` – model-judge rubric
- `suite_smoke.yaml`, `suite_full.yaml` – evaluation suites
