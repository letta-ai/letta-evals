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
```bash
python generate_leaderboard_results.py
```

5. In case of a new provider, add their logo to [leaderboard_site/src/icons](leaderboard_site/src/icons).

Results will be added to [leaderboard_results.yaml](leaderboard_results.yaml) and automatically updated.