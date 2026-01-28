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

1. Create a model config in [../letta_evals/llm_model_configs](../letta_evals/llm_model_configs).
2. Add the model to the benchmark config in either [filesystem-agent/filesystem.yaml](filesystem-agent/filesystem.yaml) or [skills-suite/suite_skill_select_use.yaml](skills-suite/suite_skill_select_use.yaml):

```yaml
target:
  model_configs:
    - provider-model-name
```

3. Run the evaluation suite:
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

4. Generate the updated leaderboard:

The `generate_leaderboard_results.py` script reads evaluation results and merges them into the leaderboard YAML file. The script looks for `aggregate_stats.json` (computed from multiple runs) or falls back to `summary.json` (from a single run) in the specified directories.

**Basic usage:**
```bash
python3 generate_leaderboard_results.py \
  filesystem-agent/results/{provider1}-{model-name-1} \
  --leaderboard leaderboard_filesystem_results.yaml \
  [--output leaderboard_filesystem_results_updated.yaml]  # generate new results file
```

The script will:
- Read `aggregate_stats.json` if available, otherwise fall back to `summary.json`
- Extract model performance metrics (scores and costs) from the JSON files
- Automatically detect the correct metric key from the existing leaderboard
- Add new models or update existing ones while preserving all data
- Handle null/missing cost data gracefully
- Normalize model names with provider prefixes (e.g., `gpt-4` → `openai/gpt-4`)

**File format requirements:**
- `aggregate_stats.json`: Contains `num_runs`, `individual_run_metrics`, and `per_model` data
- `summary.json`: Contains `metrics` and `per_model` data
- Both formats include `model_name`, `avg_score_attempted`, and optional `cost` information

5. Update leaderboard site
- Add new models and any analysis / commentary to [updates.md](../leaderboard_site/src/_includes/updates.md).
- In case of a new provider, add their logo to [leaderboard_site/src/icons](../leaderboard_site/src/icons).

Results will be added to the leaderboard YAML file and automatically updated on the website. To preview changes locally, see [README.md](../leaderboard_site/README.md) for instructions on running the leaderboard site.