# Principal Review Backlog

Date: 2026-02-12

This file groups the earlier 14 findings into implementation workstreams so they are easier to pick up later.

---

## A) Config and Determinism

### A1) Suite-only runtime config (Finding 1)
- Priority: Critical
- Problem: Runtime connection settings currently support multiple override sources (suite, CLI, env), which creates hidden behavior and surprising routing outcomes.
- Proposed change:
1. Remove CLI flags for Letta connection overrides.
2. Remove env fallback reads for Letta connection settings.
3. Resolve connection settings from suite config only.
- Impact: Fully reproducible behavior from suite YAML; fewer environment-dependent failures.

### A2) Single dataset load path (Finding 9)
- Priority: Medium
- Problem: Dataset is loaded multiple times in `run_suite()`/runner paths.
- Proposed change:
1. Load dataset once in `run_suite()`.
2. Reuse the same loaded samples for cache validation, evaluation count, and runner execution.
3. Pass `samples` into runner instead of reloading inside runner.
- Impact: Lower startup latency, lower memory churn, simpler control flow.

### A3) Dynamic object load caching (Finding 10)
- Priority: Medium
- Problem: Dynamic imports for script-based functions can be repeated unnecessarily.
- Proposed change:
1. Cache `load_object()` results by `(file_path, object_name, file_mtime)`.
2. Resolve reusable callables once during setup/init where possible.
- Impact: Lower overhead, fewer repeated module side effects, more predictable behavior.

### A4) `max_samples=0` semantics (Finding 13)
- Priority: Low
- Problem: Falsy checks treat `0` as "unset" instead of "run zero samples."
- Proposed change:
1. Replace checks like `if max_samples and ...` with `if max_samples is not None and ...`.
- Impact: Correct API semantics for edge-case callers.

---

## B) Concurrency and IO Integrity

### B1) Per-sample workspace isolation for `letta_code` (Finding 2)
- Priority: Critical
- Problem: Concurrent runs for the same model can share one working directory and interfere with each other.
- Proposed change:
1. Keep model directory as parent.
2. Create unique per-sample run directory (for example `<model>/<sample_id>-<uuid>`).
3. Execute CLI in that isolated directory.
4. Optionally support `preserve_workdirs` for debugging.
- Impact: Eliminates filesystem cross-contamination and nondeterministic failures.

### B2) Streaming writer safety (Finding 4)
- Priority: High
- Problem: Concurrent append writes are not serialized, and stale `results.jsonl` content can survive between runs.
- Proposed change:
1. Truncate/create `results.jsonl` during writer initialization.
2. Serialize writes using `anyio.Lock` or a single writer task + queue.
3. Keep summary/header writes explicit and atomic.
- Impact: Reliable output artifacts and stable downstream parsing.

### B3) Bounded worker execution model (Finding 8)
- Priority: High
- Problem: One task is scheduled per sample/model combination up front; semaphore limits execution but not task creation.
- Proposed change:
1. Build a job queue of `(sample, model)` jobs.
2. Launch `max_concurrent` worker tasks.
3. Each worker processes queued jobs until completion.
- Impact: Better scalability and memory behavior on large suites.

---

## C) Evaluation Semantics and API Cleanliness

### C1) `sample_tags` contract mismatch (Finding 5)
- Priority: High
- Problem: `sample_tags` is exposed but effectively no-op in loaders.
- Proposed change (preferred short-term):
1. Remove `sample_tags` from suite model, CLI paths, and docs until implemented.
- Proposed change (alternative):
1. Implement true tag-aware filtering by adding `tags` support in dataset schema.
- Impact: Avoids misleading users and silent misconfiguration.

### C2) Metrics attempted/total consistency (Finding 6)
- Priority: High
- Problem: Metric aggregation paths are not consistently explicit about attempted vs total semantics.
- Proposed change:
1. Define `attempted = (error is None)` once.
2. Compute attempted metrics only over attempted results.
3. Compute total metrics using all samples (with errored samples contributing zero).
4. Apply same rule across overall, per-metric, and per-model metrics.
- Impact: Easier-to-trust metrics and cleaner longitudinal comparisons.

### C3) Decorator validation cleanup (Finding 14)
- Priority: Low
- Problem: Redundant condition in `suite_setup` return annotation check.
- Proposed change:
1. Replace duplicate condition with single explicit validation branch.
- Impact: Minor correctness/maintainability improvement.

---

## D) Deferred or Accepted Risk

### D1) Default judge `.af` packaging risk (Finding 3)
- Priority: Critical
- Current decision: Deferred.
- Risk: Runtime failure if package artifacts omit required `.af` resource.
- Minimal hardening later:
1. Include `letta_evals/**/*.af` in build include rules.
2. Add explicit runtime error message if default file missing.

### D2) Core test coverage gap (Finding 7)
- Priority: High
- Current decision: Deferred.
- Risk: Regression risk in runner/metrics/streaming/data loading remains high.
- Suggested later:
1. Add unit tests for runner metric calculations.
2. Add unit tests for loader behavior (`sample_tags`, `max_samples=0`).
3. Add tests for streaming writer/reader correctness.

### D3) Docs drift (Finding 11)
- Priority: Medium
- Current decision: Accepted as fair.
- Scope:
1. Align Python version docs with package requirements.
2. Align license text with project metadata.
3. Fix stale CLI flag/file references in README/examples.

### D4) Dependency shape and pinning (Finding 12)
- Priority: Medium
- Current decision: Keep multi-provider support; other cleanup still useful.
- Suggested later:
1. Relax strict `anyio==...` to compatible range.
2. Move non-core heavy deps to extras where feasible.

---

## Suggested Execution Order (when resumed)

1. A1, B1, B2, C2 (high-value correctness and reproducibility).
2. B3, A2, A3, C1 (scaling and API contract cleanup).
3. A4, C3, then D items as capacity allows.

---

## E) Code Quality Scorecard (Current)

Overall score: 6/10

### E1) Architecture and Modularity: 6/10
- Strengths:
1. Clear domain models and enums.
2. Reasonable abstraction boundaries for targets/graders/progress callbacks.
- Gaps:
1. `runner.py` is too large and mixes orchestration, metrics math, gating, setup, and streaming.
2. Some responsibilities are duplicated between CLI and runner.

### E2) Correctness and Semantics: 5/10
- Strengths:
1. Strong typed schemas for suite/sample/result objects.
2. Structured error model exists.
- Gaps:
1. Behavior contracts are inconsistent in a few important paths (for example feature flags exposed but not implemented).
2. A few edge-case semantics are wrong or surprising (`max_samples=0`, attempted vs total metric definitions).

### E3) Reliability Under Concurrency: 5/10
- Strengths:
1. Concurrency limit via semaphore is present.
- Gaps:
1. Shared workdir risk for concurrent `letta_code` runs.
2. Streaming writes are not strongly serialized.
3. Task creation model is unbounded for large suites.

### E4) Testing and Verification: 3/10
- Strengths:
1. Linting is present and clean.
2. Live e2e path exists.
- Gaps:
1. Almost no deterministic unit tests for core logic.
2. Existing e2e test is secret-dependent and commonly skipped.

### E5) Developer Experience and Operability: 6/10
- Strengths:
1. CLI is straightforward.
2. Progress visualization options are useful.
- Gaps:
1. Docs drift from runtime reality in key places.
2. Some config behavior is non-obvious to operators.

---

## F) Concrete Improvement Plan

### F1) Refactor `runner.py` into focused modules
- Goal: reduce orchestration complexity and make logic testable.
- Concrete change:
1. Extract `metrics.py` (`_calculate_metrics`, `_calculate_run_statistics`).
2. Extract `gates.py` (`_compute_aggregation`, gate evaluation functions).
3. Extract `execution.py` (sample/model scheduling and worker loop).
4. Keep `runner.py` as coordinator only.
- Definition of done:
1. `runner.py` reduced to under ~400 lines.
2. New modules have direct unit tests.

### F2) Establish a deterministic unit test baseline
- Goal: catch regressions without external services.
- Concrete change:
1. Add tests for metric math (attempted/total/per-model/per-metric).
2. Add tests for gate behavior (simple, weighted, logical, accuracy/p95/p99).
3. Add tests for loader semantics (`max_samples`, invalid rows, optional fields).
4. Add tests for streaming writer/reader integrity.
- Definition of done:
1. At least 30 fast unit tests.
2. CI includes unit suite and blocks merge on failures.

### F3) Harden concurrency and file IO
- Goal: reproducible behavior under load.
- Concrete change:
1. Per-sample isolated working directories for `letta_code`.
2. Queue-based writer (single writer task) for `results.jsonl`.
3. Worker-pool execution model instead of one-task-per-job upfront.
- Definition of done:
1. Load test with 500+ evaluations shows no output corruption.
2. Repeated runs produce stable counts and ordering guarantees.

### F4) Define and enforce strict config contracts
- Goal: remove ambiguity and silent no-ops.
- Concrete change:
1. If keeping suite-only config, remove CLI/env override code paths.
2. Remove `sample_tags` until implemented, or implement full contract and docs.
3. Fail fast on unsupported/ignored config keys.
- Definition of done:
1. `validate` command catches all known config mismatches.
2. No known config options are silently ignored.

### F5) Improve error policy and observability
- Goal: preserve debugging signal while keeping robust execution.
- Concrete change:
1. Replace broad `except Exception` with narrower exceptions where practical.
2. Standardize error metadata fields (`category`, `type`, `root_cause`, `context`).
3. Add structured logs around retries, stream resumes, and cache hits.
- Definition of done:
1. Error reports are actionable without re-running with extra debug flags.
2. Retry paths are visible and attributable in logs.

### F6) Align docs with implementation as a release gate
- Goal: reduce user/operator confusion.
- Concrete change:
1. Add docs consistency checklist to release process.
2. Validate README snippets/flags against CLI automatically where possible.
3. Keep compatibility matrix (Python, license, key flags) in one source of truth.
- Definition of done:
1. No stale flags or contradictory requirements in README/examples.

### F7) Dependency governance
- Goal: keep install size and resolver complexity under control.
- Concrete change:
1. Keep multi-provider support as requested.
2. Relax over-strict pins where safe (for example `anyio` range).
3. Move non-core heavy dependencies to extras when they are not runtime-critical.
- Definition of done:
1. Clean install path for core features.
2. Clear extras for advanced/optional features.

---

## G) Suggested Milestones

### Milestone 1 (High confidence, low-risk)
1. A4, C3, docs fixes, loader cleanups, basic unit test scaffold.

### Milestone 2 (Correctness and determinism)
1. A1, B1, B2, C2, C1.

### Milestone 3 (Scale and maintainability)
1. B3, A2, A3, F1 full runner split, broader test coverage.

---

## H) Filesystem Benchmark Validation Follow-up

Date: 2026-03-03

### H1) `filesystem_cloud` audit and generator validation gap
- Priority: Critical
- Problem:
1. `register_question_tool.py` has an indentation bug that prevents the `verification_query` execution/match check from running, so malformed or non-unique verification queries can slip through generation.
2. The filesystem generator/prompt examples encourage SQL patterns that join `addresses` directly when ranking residents, which inflates counts and balances for owners with multiple addresses in the same state.
3. Published dataset files drop `verification_query`, making later audits harder.
- Evidence:
1. `letta-leaderboard/filesystem-agent/generation/tools/register_question_tool.py`
2. `letta-leaderboard/filesystem-agent/generation/prompts/agent_system_prompt.j2`
3. `letta-leaderboard/filesystem-agent/generation/audit_dataset.py`
- Current audit snapshot:
1. `python letta-leaderboard/filesystem-agent/generation/audit_dataset.py`
2. Latest `filesystem_cloud.jsonl` result: 64 correct, 23 ambiguous, 8 wrong, 2 wrong-and-ambiguous, 3 format issues.
3. `filesystem_code.jsonl` inherits the same label problems because it is generated directly from `filesystem_cloud.jsonl`.
- Proposed change:
1. Fix the unreachable `verification_query` execution block in `register_question_tool.py`.
2. Update prompt/examples so state-based ranking uses deduped resident sets instead of raw `JOIN addresses` multiplicity.
3. Keep or export verification metadata for published datasets, or regenerate release datasets from a richer artifact that includes it.
4. Run `audit_dataset.py` as a required release gate for filesystem dataset updates.
