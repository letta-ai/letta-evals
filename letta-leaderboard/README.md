# Letta Leaderboard

Evaluation benchmarks for testing Letta agents across different language models. View the leaderboard at https://leaderboard.letta.com/.

## Filesystem Suite

Tests an agent's ability to navigate files and answer questions that require multi-file lookups. The agent has access to 10 structured text files (people, vehicles, pets, bank accounts, etc.) and must use the `open_files` and `grep_files` tools to find and synthesize information.

## Skills Suite

Tests an agent's ability to complete tasks that require external skills. The agent must discover and load relevant skills from a skill library to satisfy all completion criteria.
The benchmark has three suites:
- `suite_baseline.yaml`: Agent does not have access to skills
- `suite_skill_use.yaml`: Agent is given the right skill and only has to load and use it (no selection required)
- `suite_skill_select_use.yaml`: Agent has to select the right skill, load and use it

## Adding New Models

To add a new model to the leaderboard:

1. Create a model config in [../letta_evals/llm_model_configs](../letta_evals/llm_model_configs).
2. Add the model to the benchmark config in [filesystem-agent/filesystem.yaml](filesystem-agent/filesystem.yaml):

```yaml
target:
  model_configs:
    - provider-model-name
```

3. Run the evaluation suite:
Filesystem Suite:
```bash
letta-evals run letta-leaderboard/filesystem-agent/filesystem.yaml \
  --output letta-leaderboard/filesystem-agent/results/{provider}-{model-name}
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
  filesystem-agent/results/filesystem-{provider1}-{model-name-1} \
  filesystem-agent/results/filesystem-{provider2}-{model-name-2} \
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

5. In case of a new provider, add their logo to [../leaderboard_site/src/icons](../leaderboard_site/src/icons).

Results will be added to the leaderboard YAML file and automatically updated on the website. To preview changes locally, see [../leaderboard_site/README.md](../leaderboard_site/README.md) for instructions on running the leaderboard site.

## Updating Other Benchmark Leaderboards

The same `generate_leaderboard_results.py` script works for all benchmarks (Filesystem, Skills Suite, etc.). Simply provide the appropriate results directory and leaderboard file:

**For Skills Suite:**
```bash
python generate_leaderboard_results.py \
  filesystem-agent/results/filesystem-gpt-51-system \
  filesystem-agent/results/filesystem-opus-45 \
  filesystem-agent/results/filesystem-gemini3-pro \
  --leaderboard leaderboard_filesystem_results.yaml
```

**For any custom benchmark:**
```bash
python3 generate_leaderboard_results.py \
  path/to/results/directory \
  --leaderboard path/to/leaderboard.yaml
```

The script automatically detects the benchmark type from the existing leaderboard file and uses the correct metric keys.

## Troubleshooting

**Script can't find aggregate_stats.json or summary.json:**
- Ensure you're pointing to the correct results directory
- Check that the evaluation completed successfully and generated output files
- The script will log which file it's looking for and whether it was found

**Model results not updating:**
- Verify the model names match exactly (including provider prefix)
- Check that the results directory contains valid JSON data
- Review the script logs for any parsing errors

**Leaderboard site issues:**
- See [../leaderboard_site/README.md](../leaderboard_site/README.md) for troubleshooting the website