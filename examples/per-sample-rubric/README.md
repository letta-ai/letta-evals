# Per-sample rubric override example

This example demonstrates how a single suite can use **different rubrics for
different samples**. The grader specifies a default rubric (`prompt_path` in
`suite.yaml`), and individual dataset rows can override it by setting either:

- `rubric` — an inline rubric string in the JSONL row, or
- `rubric_path` — a path to a rubric file (resolved relative to the dataset
  file's directory).

## Precedence

For each sample:

1. If `sample.rubric` is set (inline), it is used.
2. Else if `sample.rubric_path` is set, the loader reads the file and uses it.
3. Else the grader-level `prompt` / `prompt_path` (the suite default) is used.

`rubric` and `rubric_path` are mutually exclusive on a single row.

## Dataset (`dataset.jsonl`)

Three samples:

1. **Cat** — provides an inline `rubric` field tailored to evaluating cat ASCII art.
2. **Dog** — provides `rubric_path: rubrics/dog.txt`, a separate rubric file
   focused on dog-specific features.
3. **Fish** — provides neither, so the suite-level rubric in `rubric.txt` is used.

## Template variables

All rubrics — suite-level, inline, and file-based — are Jinja templates, rendered
and sent verbatim to the judge. These variables are always available:

- `{{ input }}` — the original task given to the agent (`sample.input`)
- `{{ ground_truth }}` — the expected answer (`sample.ground_truth`); empty string if not set
- `{{ submission }}` — the agent's extracted output
- Any key from `sample.rubric_vars` (per-row dict in the dataset)

A rubric that references a variable a sample doesn't provide fails before any
agent runs.

## Running

```
letta-evals run examples/per-sample-rubric/suite.yaml
```
