# Letta Leaderboard

Evaluation benchmarks for testing Letta agents across different language models. View the leaderboard at https://leaderboard.letta.com/.

## Filesystem Suite

Tests an agent's ability to navigate files and answer questions that require multi-file lookups. The agent has access to 10 structured text files (people, vehicles, pets, bank accounts, etc.) and must use the `open_files` and `grep_files` tools to find and synthesize information.

## Skills Suite

Tests an agent's ability to complete tasks that require external skills. The agent must discover and load relevant skills from a skill library to satisfy all completion criteria.
The benchmark has three suites:
- `suite_baseline.yaml`: Agent does not have access to skills
- `suite_skill_use.yaml`: Agent is given the right skill and only has to load and use it (no selection required)
- `suite_skill_select_use.yaml`: Agent has to select the right skill, load and use it -> shown on leaderboard

## Adding New Models

To add a new model to the leaderboard:

1. Add the model handle to the benchmark config in either [filesystem-agent/filesystem.yaml](filesystem-agent/filesystem.yaml) or [skills-suite/suite_skill_select_use.yaml](skills-suite/suite_skill_select_use.yaml):

```yaml
target:
  model_handles:
    - provider/model-name
```

2. Run the evaluation suite:
Filesystem (Cloud) Suite:
```bash
letta-evals run letta-leaderboard/filesystem-agent/filesystem_cloud.yaml \
  --output letta-leaderboard/filesystem-agent/code-results/{provider}-{model-name}
```

Filesystem (Code) Suite:
```bash
letta-evals run letta-leaderboard/filesystem-agent/filesystem_code.yaml \
  --output letta-leaderboard/filesystem-agent/code-results/{provider}-{model-name}
```

Skills Suite:
```bash
letta-evals run letta-leaderboard/skills-suite/suite_skill_select_use.yaml \
  --output letta-leaderboard/skills-suite/results/{provider}-{model-name}
```

3. Generate the updated leaderboard:

The `generate_leaderboard_results.py` script reads evaluation results from the unified `summary.json` produced by `letta-evals` and merges them into the leaderboard YAML file. `summary.json` has the same shape for single-run and multi-run; the script mean-aggregates across runs automatically.

**Basic usage:**
```bash
python3 generate_leaderboard_results.py \
  filesystem-agent/results/{provider1}-{model-name-1} \
  --leaderboard leaderboard_filesystem_results.yaml \
  [--output leaderboard_filesystem_results_updated.yaml]  # generate new results file
```

The script will:
- Read `summary.json` from each results directory
- Extract per-model `reward` (0–1, reported ×100) and `usage.cost`
- Automatically detect the correct metric key from the existing leaderboard
- Add new models or update existing ones while preserving all data
- Handle null/missing cost data gracefully
- Normalize model names with provider prefixes (e.g., `gpt-4` → `openai/gpt-4`)

**File format:**
- `summary.json`: Top-level fields `{ suite, models: [...] }`
- Each entry in `models` has `{ model, n_total, n_attempted, reward, per_metric, usage, timing, ... }` (multi-run also has `reward_std` and `per_metric_std`)

4. Update leaderboard site
- Add new models and any analysis / commentary to [updates.md](updates.md).
- In case of a new provider, add their logo to [leaderboard_site/src/icons](leaderboard_site/src/icons).

Results will be added to the leaderboard YAML file and automatically updated on the website. To preview changes locally, see [README.md](leaderboard_site/README.md) for instructions on running the leaderboard site.