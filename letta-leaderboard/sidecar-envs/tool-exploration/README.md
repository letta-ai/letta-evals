# Tool Exploration (v0)

Goal: Given reference documentation and an exploration transcript for a tool/API/website, produce a structured report identifying quirks, documentation errors, hidden features, and false alarms. Output must be valid JSON only.

## Canonical JSON Schema (v0)

Required top-level keys:
- `tool_name` (string)
- `category` (one of: "api", "cli", "library", "service")
- `quirks` (array of objects)
- `doc_errors` (array of objects)
- `hidden_features` (array of strings)
- `false_alarms` (array of strings)

`quirks` object shape:
```
{
  "title": string,
  "expected": string (what docs/intuition suggest),
  "actual": string (what really happens),
  "severity": "breaking" | "surprising" | "cosmetic",
  "workaround": string | null
}
```

`doc_errors` object shape:
```
{
  "location": string (where in the docs the error is),
  "correction": string (what it should say)
}
```

Rules:
- Only list genuine quirks (unexpected behaviors that are real, not user mistakes).
- False alarms are things that looked like bugs during exploration but turned out to be user error, expected behavior, or environmental issues.
- Doc errors are cases where the documentation is factually wrong.
- Hidden features are undocumented capabilities discovered during exploration.
- Output must be a single JSON object with no code fences or extra text.

## Files
- `data/dataset_all.jsonl` -- full dataset (smoke suite uses `max_samples` to select a subset)
- `rubric_exploration.txt` -- model-judge rubric
- `suite_smoke.yaml`, `suite_full.yaml` -- evaluation suites
