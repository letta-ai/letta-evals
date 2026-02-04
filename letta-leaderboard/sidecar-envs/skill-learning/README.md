# Skill Learning From Tests (v0)

Goal: given a small set of tests + expected outcomes, produce a `SKILL.md` (or skill update) that would help a model pass those tests. Output must be the SKILL content only.

## Artifact Contract (v0)

- Output a single Markdown document (SKILL.md content) with frontmatter:
  - `name`
  - `description`
- Must include:
  - Overview
  - When to use
  - Workflow / steps
  - Edge cases / pitfalls
  - Example or checklist
- Keep concise; no tools or file writes (response-only).

## Files
- `data/dataset_all.jsonl` – full dataset (smoke suite uses `max_samples` to select a subset)
- `rubric_skill_learning.txt` – model-judge rubric
- `suite_smoke.yaml`, `suite_full.yaml` – evaluation suites
