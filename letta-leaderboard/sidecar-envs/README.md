# Sidecar Environments

This directory contains "sidecar" evaluation environments: small, self-contained suites that live alongside the Letta Leaderboard but are not (yet) part of the public leaderboard.

The primary focus is *learned context writing* (e.g., maintaining user models, writing session handoffs, anticipating future queries) and adjacent tasks (e.g., skill learning/test writing) that are useful for post-training and regression testing.

See `letta-leaderboard/sidecar-envs/TODO.md` for the roadmap and design notes.

## Two Modes

Response-only artifact-writing suites:

- The *reward depends only on the model's final response text*.
- The task contract is an artifact emitted in the response (usually strict JSON, sometimes Markdown).
- Graders must only read the extracted final assistant message (no tool calls, no filesystem state, no agent memory/state).

Full Letta Code suites:

- The reward may depend on tools, trajectories, agent state/memory blocks, or filesystem state.
- These suites typically require post-run agent-state retrieval and more complex extractors/graders.

## Quick Start

Run a suite from the repo root:

```bash
uv run letta-evals run letta-leaderboard/sidecar-envs/user-model-maintenance/suite_smoke.yaml -o results/umm-smoke/
uv run letta-evals run letta-leaderboard/sidecar-envs/session-handoff/suite_smoke.yaml -o results/handoff-smoke/
uv run letta-evals run letta-leaderboard/sidecar-envs/skill-learning/suite_smoke.yaml -o results/skill-learning-smoke/
```

Notes:

- Many suites default to Letta Cloud via `target.base_url: https://api.letta.com/`. Ensure your Letta Cloud credentials are set in the environment.
- Rubric grading uses `kind: model_judge` and requires the configured provider API key (e.g., `OPENAI_API_KEY` for an OpenAI judge model).

## Directory Conventions

Most sidecar environments follow this layout:

- `README.md`: task description + artifact contract
- `data/dataset_all.jsonl`: full dataset (smoke suites use `max_samples` to select a subset)
- `rubric_*.txt`: rubric prompt for `model_judge`
- `suite_smoke.yaml`: smoke suite config (small; fast)
- `suite_full.yaml`: larger suite config

Some environments may also include helper scripts, agentfiles, or result-parsing utilities.

## Dataset Format (JSONL)

`letta-evals` datasets use these keys:

- `input` (required): string or list of strings (multi-turn)
- `ground_truth` (optional): string or list (for per-turn grading)
- `agent_args` (optional): dict for target/agent creation
- `rubric_vars` (optional): dict for rubric prompt substitution
- `extra_vars` (optional): dict for custom extractors/graders

Additional top-level keys may exist for other tooling, but `letta-evals` will ignore unknown fields.

## Environments

- `user-model-maintenance/`: update a canonical `user_model.json` from new conversation facts (strict JSON output).
- `session-handoff/`: produce a concise `handoff.json` for the next session (strict JSON output).
- `skill-learning/`: given tests + expected outcomes, produce `SKILL.md` content (Markdown output only).
- `skill-test-writing/`: generate strong tests for skills (includes dataset generation and result parsing utilities).

## letta-evals Reference

### Targets

- `letta_agent`: Creates and runs a Letta agent via the Letta API/SDK.
- `letta_code`: Runs the Letta Code CLI in headless JSON mode.

### Graders

- `model_judge`: LLM rubric grader (RubricGrader).
- `letta_judge`: Agent judge that calls a tool to return a score (AgentJudgeGrader).
- `tool`: Tool grader with built-ins (`contains`, `exact_match`, `ascii_printable_only`).

### Extractors (built-in)

- `last_assistant`, `first_assistant`, `all_assistant`, `last_turn`
- `pattern` (regex)
- `tool_arguments`, `tool_output`
- `after_marker`
- `memory_block` (requires agent_state memory blocks)

### CLI Flows

- Run: `letta-evals run <suite.yaml>`
- Validate: `letta-evals validate <suite.yaml>`
- List: `letta-evals list-extractors`, `letta-evals list-graders`

### Suite YAML Touchpoints

- `dataset`: JSONL/CSV path
- `target`: `kind`, `base_url`, `model_handles` or `model_configs`, `working_dir`, `sandbox`, `timeout`, etc.
- `graders`: one or more graders with `prompt_path` and `extractor`
- `gate`: threshold for pass/fail

### Friction Notes

- CLI may not be on PATH after install; use `uv run` or `python -m letta_evals.cli`.
- Default `base_url` should be cloud (`https://api.letta.com/`) unless running a local server.
