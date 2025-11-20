# Letta Leaderboard

Evaluation benchmarks for testing Letta agents across different language models. View the leaderboard at https://leaderboard.letta.com/.

## Filesystem Agent Task

Tests an agent's ability to navigate files and answer questions that require multi-file lookups. The agent has access to 10 structured text files (people, vehicles, pets, bank accounts, etc.) and must use the `open_files` and `grep_files` tools to find and synthesize information.

## Adding New Models

To add a new model to the leaderboard:

1. Create a model config in [../letta_evals/llm_model_configs](../letta_evals/llm_model_configs).
2. Add the model to the benchmark config in [filesystem-agent/filesystem.yaml](filesystem-agent/filesystem.yaml):

```yaml
target:
  model_configs:
    - provider-model-name
```

3. Run the evaluation:
```bash
letta-evals run letta-leaderboard/filesystem-agent/filesystem.yaml \
  --output letta-leaderboard/filesystem-agent/results/filesystem-{provider}-{model-name}
```

Note: The `filesystem-agent/results/` directory uses Git LFS (Large File Storage) to manage result files. Ensure you have [Git LFS installed](https://git-lfs.com/) and configured before committing results. To pull existing result files, run:
```bash
git lfs pull
```

4. Generate the updated leaderboard:

The `generate_leaderboard_results.py` script reads evaluation results and merges them into the leaderboard YAML file. The script looks for `aggregate_stats.json` (computed from multiple runs) or falls back to `summary.json` (from a single run) in the specified directories.

**Basic usage:**
```bash
python3 generate_leaderboard_results.py \
  filesystem-agent/results/filesystem-{provider}-{model-name} \
  --leaderboard leaderboard_filesystem_results.yaml
```

**Update multiple models at once:**
```bash
python3 generate_leaderboard_results.py \
  filesystem-agent/results/filesystem-model1 \
  filesystem-agent/results/filesystem-model2 \
  filesystem-agent/results/filesystem-model3 \
  --leaderboard leaderboard_filesystem_results.yaml
```

**Save to a different file:**
```bash
python3 generate_leaderboard_results.py \
  filesystem-agent/results/filesystem-{provider}-{model-name} \
  --leaderboard leaderboard_filesystem_results.yaml \
  --output leaderboard_filesystem_results_updated.yaml
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
python3 generate_leaderboard_results.py \
  skills-suite/results/skills-{provider}-{model-name} \
  --leaderboard leaderboard_skill_results.yaml
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