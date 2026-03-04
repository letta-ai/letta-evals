# Validation Refactor Proposal

## Goals

- Collapse validation into one shared engine.
- Remove any required manual validation step from the normal generation workflow.
- Preserve enough provenance in published datasets to revalidate them later.
- Reduce dependence on regex-heavy natural-language parsing.
- Keep the generator targeting accepted questions, not attempted slots.

## Problem

The current system has too many validation components with overlapping responsibilities:

- `register_question` validates candidate questions before writing them.
- `question_generator.py` runs end-of-run validation.
- `validate_questions.py` exists as a separate validation script.
- `audit_dataset.py` reconstructs benchmark semantics by parsing natural-language questions.

This is happening because the published dataset does not retain enough structured validation metadata. The raw generation artifact stores `verification_query`, but the published benchmark form drops it and keeps only:

- `input`
- `ground_truth`
- `agent_args`

That forces the dataset auditor to infer semantics from English question text.

## Proposal

Introduce a single shared validation system and make it the only source of truth.

### One Validation Engine

Add a shared module, for example:

- `generation/validation_engine.py`

All validation logic should live there and be reused by:

- `register_question`
- `question_generator.py`
- any dataset release/replacement tooling
- any optional debug CLI

The CLI wrappers should be thin entrypoints over the shared engine, not separate implementations.

## Canonical Stored Metadata

Every generated row should preserve validation provenance in both:

- the raw generation artifact
- the published benchmark artifact

Suggested shape:

```json
{
  "input": "Natural language question",
  "ground_truth": "Tammy Roberts",
  "agent_args": {
    "tags": [],
    "extra": {
      "question_type": "multi_entity_comparison",
      "difficulty": "hard",
      "required_files": ["pets.txt", "addresses.txt", "bank_accounts.txt", "vehicles.txt", "people.txt"],
      "validation": {
        "verification_query": "WITH ...",
        "question_program": {
          "template_id": "compare_metric_between_two_populations",
          "population_a": {
            "anchor_type": "pet_name",
            "anchor_value": "Alyssa",
            "relation": "same_state_as_owner",
            "selector": "highest_total_bank_balance"
          },
          "population_b": {
            "anchor_type": "pet_name",
            "anchor_value": "Betty",
            "relation": "same_state_as_owner",
            "selector": "highest_total_bank_balance"
          },
          "comparison_metric": "vehicle_count",
          "comparison_mode": "argmax",
          "tie_policy": "tie"
        },
        "validation_version": 2
      }
    }
  }
}
```

## Why Store Both `verification_query` And `question_program`

`verification_query` and a structured program solve different problems.

`verification_query` helps answer:

- Does the candidate's executable derivation return one row?
- Does that row match the stored ground truth?

`question_program` helps answer:

- Does the stored SQL actually match the intended semantics of the question?
- Is the question truly unique or secretly ambiguous?
- Did the generator introduce an extra tiebreak or answer a narrower question?

Storing only `verification_query` simplifies re-execution, but it does not eliminate semantic mismatch. Storing `question_program` is what removes the need to parse English with regex as the primary benchmark audit path.

## Validation Engine API

Suggested API:

```python
validate_candidate(raw_candidate, db_path) -> ValidationResult
validate_run(raw_rows, parsed_rows, db_path) -> ValidationSummary
validate_dataset(published_rows, db_path) -> ValidationSummary
evaluate_question_program(program, db_path) -> ProgramEvaluation
```

### `ValidationResult`

Should include:

- `status`: `accepted | rejected`
- `issues`: structured issue codes
- `normalized_answer`
- `query_result`
- `program_result_set`
- `program_uniqueness`: `unique | ambiguous | empty`
- `validation_version`

## Candidate Acceptance Rules

A candidate should be accepted only if all of the following pass:

- Basic shape checks pass.
- `verification_query` exists.
- `verification_query` executes successfully.
- `verification_query` returns exactly one row and one value.
- The query result matches `ground_truth`.
- `question_program` exists and passes schema validation.
- `question_program` independently evaluates against the DB.
- `question_program` yields exactly one valid answer.
- That answer matches `ground_truth`.
- The `verification_query` result matches the `question_program` result.

## Question Program Evaluation

For new benchmark rows, semantic validation should be driven by structured programs, not by parsing English.

The evaluation model should be:

- dispatch by `template_id`
- evaluate the structured operands against the DB
- compute the valid answer set
- classify as `unique`, `ambiguous`, or `empty`

This replaces the current benchmark-audit strategy of hard-coded English-pattern matching for newly generated rows.

## Question Text Strategy

Preferred long-term approach:

- the model proposes a structured `question_program`
- the system renders the benchmark question text from that program

Acceptable intermediate approach:

- the model proposes both text and structure
- the system validates that the text is consistent with the structured program

The structured program should be the semantic source of truth. The English question should be treated as the presentation layer.

## Workflow

Normal workflow should be only:

```bash
python3 question_generator.py ...
```

Internally:

1. The model proposes a candidate.
2. `register_question` calls `validate_candidate`.
3. Only accepted rows are written.
4. The generator tops up until it reaches the target accepted count.
5. `question_generator.py` runs `validate_run` before exiting.
6. Any failure returns a non-zero exit code.

No separate manual validation command should be required in normal operation.

## CLI Simplification

- `validate_questions.py` should become an optional debug wrapper over the shared validation engine, or be removed.
- `audit_dataset.py` should stop being an independent logic path and either become a wrapper or be folded into the shared engine.
- The normal release path should not depend on manually invoking multiple scripts.

## File-Level Direction

Add:

- `generation/validation_engine.py`
- `generation/question_program.py`

Update:

- `generation/tools/register_question_tool.py`
- `generation/question_generator.py`
- `generation/validate_questions.py`
- `generation/audit_dataset.py`
- `create_code_dataset.py`

## Migration Plan

### Phase 1

- Preserve `verification_query` in the published dataset.
- Preserve `question_program` in the published dataset.
- Introduce the shared validation engine.
- Keep the old regex-based dataset audit only as a fallback for legacy rows.

### Phase 2

- Require `question_program` for all newly generated benchmark rows.
- Validate new datasets from structure, not English parsing.
- Treat the regex auditor as legacy-only.

### Phase 3

- Backfill or regenerate legacy rows to include `question_program`.
- Remove the legacy English-pattern fallback once all active benchmark rows are structured.

## Backwards Compatibility

For legacy dataset rows without `question_program`:

- if `verification_query` exists, execute it
- mark the row as `legacy_unstructured`
- optionally fall back to the old regex parser

Newly generated benchmark rows should not rely on that fallback.

## Acceptance Criteria

- `register_question` is the primary admission gate for new rows.
- `question_generator.py` is the only command needed in the normal workflow.
- Published datasets retain `verification_query` and `question_program`.
- New benchmark releases do not depend on regex parsing of English text.
- Published datasets can be revalidated directly from stored structured metadata plus the DB.
- Dataset replacement workflows use the same shared engine as generation.

## Recommendation

The cleanest end state is:

- store `verification_query`
- store `question_program`
- make `question_program` the real semantic artifact
- render English question text from structure

Storing only `verification_query` helps, but it does not remove semantic mismatches by itself. The structured program is the piece that actually collapses the validation surface area.
