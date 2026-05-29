# Runner refactor plan

This document tracks the staged refactor of `letta_evals/runner.py` after the
0.19.1 cleanup release. The goal is to reduce runner size and make each
execution concern independently testable without changing suite YAML or runtime
behavior.

## Constraints

- Preserve existing suite YAML compatibility.
- Keep each PR focused and behavior-preserving.
- Prefer moving isolated code first, then simplify orchestration after the file
  is smaller.
- Keep function signatures straightforward; avoid keyword-only parameter blocks
  for newly extracted runner helpers.

## PR sequence

### 1. Extract sandbox dispatch

Move Modal sandbox sample execution out of `Runner` and into
`letta_evals/sandbox/dispatch.py`.

Expected changes:
- Move sandbox environment forwarding, suite/sample upload, in-sandbox CLI
  command construction, result download, and sandbox error result handling.
- Keep `Runner.run_sample` responsible for progress callbacks and dispatch
  selection.
- Retarget sandbox dispatch tests to the new module.

### 2. Extract grading flow

Move grading-specific helpers out of `Runner`.

Candidate functions:
- per-turn grading
- multi-grader sample grading
- target/submission error detection
- rubric variable validation

Expected destination: `letta_evals/grading.py` or a similarly focused module.

### 3. Extract sample result helpers

Move pure result construction helpers out of `Runner`.

Candidate functions:
- usage construction
- error `SampleResult` construction
- primary grade/metric summary extraction
- token/cost/timing assembly helpers

### 4. Extract target execution/cache path

Move target execution and cached trajectory handling into a focused module.

Candidate functions:
- model id normalization
- trajectory cache construction
- Letta Code target construction
- get-or-run trajectory

### 5. Simplify suite orchestration

After sandbox, grading, result construction, and target execution are extracted,
clean up the remaining `Runner.run` / multi-run orchestration logic.

Candidate improvements:
- make the model/sample loop easier to follow
- isolate streaming writes
- isolate per-model summary construction
- keep concurrency behavior unchanged

## Current status

- [x] Extract sandbox dispatch
- [ ] Extract grading flow
- [ ] Extract sample result helpers
- [ ] Extract target execution/cache path
- [ ] Simplify suite orchestration
