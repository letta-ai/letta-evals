# Letta Evals

Letta Evals is a framework for evaluating [Letta](https://github.com/letta-ai/letta) and Letta Code agents. It lets you define an evaluation suite with a dataset, target, extractors, graders, and a reward contract, then run that suite against one or more model configurations.

<img width="596" src="https://github.com/user-attachments/assets/4471f0b0-8353-48b7-8f52-b51bbf0482cb" alt="Letta Evals running an evaluation suite with real-time progress tracking">

If you are building agentic systems, high-quality evals are one of the fastest ways to understand how model versions, prompts, tools, or agent configuration changes affect your product.

## Requirements

- Python 3.11+
- A running Letta server, either:
  - **Self-hosted**: follow the [Letta installation guide](https://docs.letta.com/guides/ade/desktop#self-hosted-server-mode-recommended), or
  - **Letta Cloud**: create an account at [app.letta.com](https://app.letta.com) and set:
    ```bash
    export LETTA_API_KEY=your-api-key
    export LETTA_PROJECT_ID=your-project-id
    ```
    Then use `base_url: https://api.letta.com/` in your suite YAML, or pass `--base-url https://api.letta.com/` on the CLI.
- Provider API keys for the models you use, such as `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GOOGLE_API_KEY`.

## Installation

For local development or custom eval authoring, clone this repository and install with dev dependencies:

```bash
uv sync --extra dev
```

To run existing evals without editing the repo:

```bash
pip install letta-evals
```

## Quick start

1. Create a dataset (`dataset.jsonl`):

```jsonl
{"input": "What's the capital of France?", "ground_truth": "Paris"}
{"input": "Calculate 2+2", "ground_truth": "4"}
```

2. Create a suite (`suite.yaml`):

```yaml
name: my-eval-suite
dataset: dataset.jsonl

target:
  kind: letta_code
  model_handles:
    - openai/gpt-4.1-mini
  base_url: http://localhost:8283

graders:
  correctness:
    kind: tool
    function: contains
    extractor: last_assistant

reward:
  kind: metric
  metric_key: correctness
```

3. Validate and run:

```bash
letta-evals validate suite.yaml
letta-evals run suite.yaml
```

## Running evals

The core flow is:

**Dataset → Target → Extractor → Grader → Reward → Result**

Common commands:

```bash
# Run an evaluation suite with progress output
letta-evals run suite.yaml

# Save suite.json, summary.json, and per-model JSONL results
letta-evals run suite.yaml --output results/

# Run multiple times for aggregate statistics
letta-evals run suite.yaml --num-runs 5 --output results/

# Re-grade saved trajectories without re-running the target
letta-evals run suite.yaml --cached results/openai-gpt-4.1-mini.jsonl

# Validate suite configuration and list built-ins
letta-evals validate suite.yaml
letta-evals list-extractors
letta-evals list-graders
```

You can also set run defaults in `suite.yaml`:

```yaml
max_concurrent: 5
max_samples: 20
num_runs: 3
output: results/
cleanup: true
```

Relative paths in suite YAML are resolved from the suite file's directory. CLI flags such as `--max-concurrent`, `--output`, `--api-key`, `--base-url`, `--project-id`, and `--num-runs` override suite or environment defaults when provided.

## Writing suites

### Datasets

Datasets can be JSONL or CSV. Each row should provide the user input and, for most graders, a `ground_truth` value:

```json
{"input": "Draw a cat in ASCII", "ground_truth": "cat"}
```

For multi-turn evals, set `input` to a list of user messages. If `ground_truth` is also a list of the same length, supported graders can score each turn independently and average the per-turn scores.

Dataset rows may also include fields such as:

- `extra_vars` for custom graders
- `agent_args` for programmatic agent factories
- `rubric` or `rubric_path` for per-sample model-judge rubric overrides

### Targets

The supported target is `letta_code`, which runs the Letta Code CLI against a Letta server. Important target fields include:

- `base_url`: Letta server URL; defaults to `http://localhost:8283`
- `model_handles`: one or more model handles to evaluate
- `agent_script`: optional `file.py:function_name` agent factory
- `flags`: additional Letta Code CLI flags (including tool restrictions, e.g. `--allowed-tools Bash Read`)
- `permission_mode`: optional Letta Code permission mode
- `letta_command`: Letta Code executable to invoke; defaults to `letta` on `PATH`
- `letta_code_version`: optional expected `@letta-ai/letta-code` CLI version; fails fast if `letta_command --version` does not match
- `disable_autoupdater`: set `DISABLE_AUTOUPDATER=1` for Letta Code subprocesses; automatically enabled when `letta_code_version` is set
- `timeout` and `max_retries`: target execution controls

For reproducible runs, install the desired Letta Code CLI separately and pin it in the target:

```yaml
target:
  kind: letta_code
  model_handles:
    - openai/gpt-4.1-mini
  letta_command: /path/to/letta
  letta_code_version: 0.27.16
```

### Graders and extractors

Suites can use deterministic tool graders or model-judge graders:

```yaml
graders:
  exact:
    kind: tool
    function: exact_match
    extractor: last_assistant
  quality:
    kind: model_judge
    prompt_path: rubric.txt
    model: gpt-5-mini
    provider: openai
    extractor: last_assistant
```

Use `letta-evals list-graders` and `letta-evals list-extractors` for built-ins. You can register custom Python graders, extractors, setup hooks, and agent factories with decorators; see [`examples/custom-tool-grader-and-extractor/`](examples/custom-tool-grader-and-extractor/) and [`examples/programmatic-agent-creation/`](examples/programmatic-agent-creation/).

### Rewards

A reward turns grader outputs into the canonical per-sample scalar stored on `SampleResult.reward`. The framework owns the contract and persistence; suite authors own any custom composition logic.

For simple suites, use one grader directly:

```yaml
reward:
  kind: metric
  metric_key: correctness
```

For suite-specific composition, point at a Python reward composer:

```yaml
reward:
  kind: custom
  function: rewards.py:compose_reward
```

```python
from letta_evals import RewardOutput, reward_composer


@reward_composer
def compose_reward(ctx):
    quality = ctx.grades["quality"].score
    valid = ctx.grades["validity_check"].score
    if valid < 1.0:
        return RewardOutput(score=0.0, metadata={"reason": "validity_check_failed"})
    return RewardOutput(score=quality)
```

`grades` remain the source of truth for raw grader outputs. Reward metadata should only contain derived composer decisions that are not already recoverable from grades, submissions, or the sample.

See [`examples/reward-composition/`](examples/reward-composition/) for complete custom reward examples.

### Setup scripts and agent factories

Use `setup_script: file.py:function_name` for one-time setup before a suite runs. Setup functions may have one of these signatures:

- `() -> None`
- `(client: AsyncLetta) -> None`
- `(client: AsyncLetta, model_handle: str) -> None`

Use `target.agent_script: file.py:function_name` to create or customize an agent per sample. Agent factories receive the Letta client and current `Sample`, and return the agent ID.

## Examples

The [`examples/`](examples/) directory contains working suites:

- [`examples/custom-tool-grader-and-extractor/`](examples/custom-tool-grader-and-extractor/) — custom Python extractor and grader for structured JSON output
- [`examples/letta-code-simple-edit/`](examples/letta-code-simple-edit/) — Letta Code fixes buggy Python files, with a subprocess grader
- [`docs/modal-sandbox.md`](docs/modal-sandbox.md) — per-sample Modal sandbox execution
- [`examples/reward-composition/`](examples/reward-composition/) — custom reward composers across multiple graders
- [`examples/multi-model-simple-rubric-grader/`](examples/multi-model-simple-rubric-grader/) — compare multiple model handles in one suite
- [`examples/multiturn-per-turn-grading/`](examples/multiturn-per-turn-grading/) — score each turn of a multi-turn conversation
- [`examples/per-sample-rubric/`](examples/per-sample-rubric/) — override model-judge rubrics per dataset row
- [`examples/programmatic-agent-creation/`](examples/programmatic-agent-creation/) — create customized agents from Python before each sample

## Modal sandbox execution

Add a suite-level `sandbox` block to run every sample inside a fresh Modal sandbox:

```yaml
sandbox:
  kind: modal
  cpu: 2
  memory_mb: 4096
  timeout_sec: 1800
```

The host runner still owns the sample loop, concurrency, JSONL output, and reward aggregation. Each sample is uploaded to a sandbox along with the suite directory; the target, extractors, graders, and reward composer run in the sandbox; and the final `SampleResult` is returned to the host.

See [`docs/modal-sandbox.md`](docs/modal-sandbox.md) for setup details, networking notes, and common failure modes.

## FAQ

**Can I write evals without Python code?**

Yes. Many suites only need YAML plus JSONL/CSV data and built-in graders such as `contains`, `exact_match`, `regex_match`, or model-judge grading.

**Can I test multiple models?**

Yes. Set `target.model_handles` to a list. Letta Evals runs every sample for every model and writes per-model results.

**Can I run evaluations repeatedly?**

Yes. Use `--num-runs N` or `num_runs: N` to compute aggregate statistics across repeated suite runs.

**Can I reuse trajectories while iterating on graders?**

Yes. Save results with `--output`, then pass a saved JSONL file back with `--cached` to re-grade without re-running the target. `--num-runs > 1` is not supported with cached results because the trajectories would be identical.

**Can I use this in CI/CD?**

Yes. Letta Evals is designed for CI. See [`.github/workflows/e2e-tests.yml`](.github/workflows/e2e-tests.yml) for an example of running suites in GitHub Actions.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for local setup, linting, testing, and PR guidelines.

## License

This project is licensed under the Apache License 2.0. By contributing to this repository, you agree that your contributions are licensed under the repository's license. You must have adequate rights to upload any data used in an eval. Letta reserves the right to use this data in future service improvements to our product.
