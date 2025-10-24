"""
Leaderboard Results Generator

Processes evaluation results from filesystem benchmarks and generates
a YAML leaderboard with model performance metrics and costs.
"""

import json
import logging
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Model pricing configuration (costs per million tokens)
MODEL_COSTS = {
    "anthropic/claude-opus-4-1-20250805": {
        "prompt_tokens": 15,
        "completion_tokens": 75,
    },
    "anthropic/claude-sonnet-4-5-20250929": {
        "prompt_tokens": 3,
        "completion_tokens": 6,
    },
    "anthropic/claude-haiku-4-5-20251001": {
        "prompt_tokens": 1,
        "completion_tokens": 5,
    },
    "openai/gpt-5-2025-08-07": {
        "prompt_tokens": 1.25,
        "completion_tokens": 10,
    },
    "openai/gpt-5-mini-2025-08-07": {
        "prompt_tokens": 0.25,
        "completion_tokens": 2,
    },
    "openai/gpt-5-nano-2025-08-07": {
        "prompt_tokens": 0.05,
        "completion_tokens": 0.4,
    },
    "openai/gpt-4.1-2025-04-14": {
        "prompt_tokens": 2,
        "completion_tokens": 8,
    },
    "openai/gpt-4.1-mini-2025-04-14": {
        "prompt_tokens": 0.4,
        "completion_tokens": 1.6,
    },
    "openai/gpt-4.1-nano-2025-04-14": {
        "prompt_tokens": 0.10,
        "completion_tokens": 0.4,
    },
    "deepseek/deepseek-chat-v3.1": {
        "prompt_tokens": 0.27,
        "completion_tokens": 1,
    },
    "moonshotai/kimi-k2-0905": {
        "prompt_tokens": 0.39,
        "completion_tokens": 1.9,
    },
    "z-ai/glm-4.6": {
        "prompt_tokens": 0.5,
        "completion_tokens": 1.75,
    },
    "openai/gpt-oss-120b": {
        "prompt_tokens": 0.15,
        "completion_tokens": 0.6,
    },
    "openai/gpt-oss-20b": {
        "prompt_tokens": 0.05,
        "completion_tokens": 0.2,
    },
}

EXCLUDED_MODELS = {"moonshotai/Kimi-K2-Instruct-0905"}


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model names by adding provider prefixes.

    Args:
        model_name: Raw model name from results

    Returns:
        Normalized model name with provider prefix
    """
    if model_name.startswith("claude"):
        return f"anthropic/{model_name}"
    if model_name.startswith("gpt"):
        return f"openai/{model_name}"
    return model_name


def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate the cost for a model's token usage.

    Args:
        model_name: Name of the model
        prompt_tokens: Number of prompt tokens used
        completion_tokens: Number of completion tokens used

    Returns:
        Total cost in dollars

    Raises:
        KeyError: If model_name is not in MODEL_COSTS
    """
    model_costs = MODEL_COSTS[model_name]
    prompt_cost = model_costs["prompt_tokens"] * prompt_tokens / 1_000_000
    completion_cost = model_costs["completion_tokens"] * completion_tokens / 1_000_000
    return prompt_cost + completion_cost


def find_result_files(pattern: str = "results/filesystem-*/**/results.jsonl") -> List[Path]:
    """
    Find all result files matching the given pattern.

    Args:
        pattern: Glob pattern for finding result files

    Returns:
        Sorted list of Path objects for result files
    """
    paths = sorted(glob(pattern, recursive=True))
    logger.info(f"Found {len(paths)} result files")
    return [Path(p) for p in paths]


def parse_result_entry(row: Dict, line_num: int, file_path: Path) -> Tuple[str, float, float, bool]:
    """
    Parse a single result entry from a JSONL file.

    Args:
        row: Parsed JSON row from results file
        line_num: Line number in file (for error reporting)
        file_path: Path to the file being processed

    Returns:
        Tuple of (model_name, score, cost, has_error)
    """
    try:
        model_name = row["result"]["model_name"]

        # Skip excluded models
        if model_name in EXCLUDED_MODELS:
            return None, None, None, False

        model_name = normalize_model_name(model_name)
        score = row["result"]["grade"]["score"]

        # Extract token usage
        try:
            prompt_tokens = row["result"]["agent_usage"][0]["prompt_tokens"]
            completion_tokens = row["result"]["agent_usage"][0]["completion_tokens"]
            cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
            has_error = False
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Missing token data in {file_path}:{line_num} - {e}")
            cost = 0.0
            has_error = True

        return model_name, score, cost, has_error

    except (KeyError, TypeError) as e:
        logger.error(f"Error parsing entry in {file_path}:{line_num} - {e}")
        return None, None, None, True


def load_results(result_files: List[Path]) -> Tuple[Dict[str, List], int, int]:
    """
    Load and parse all result files.

    Args:
        result_files: List of paths to result files

    Returns:
        Tuple of (results_dict, total_samples, error_count)
    """
    results = defaultdict(list)
    num_total = 0
    num_errors = 0

    for file_path in result_files:
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, "r") as f:
                for line_num, line in enumerate(f, start=1):
                    num_total += 1
                    try:
                        row = json.loads(line)
                        # print(f"Row: {row['result'].keys()}")
                        model_name, score, cost, has_error = parse_result_entry(row, line_num, file_path)
                        # print(f"Model name: {model_name}, Score: {score}, Cost: {cost}, Has error: {has_error}")

                        if model_name is None:
                            continue

                        if model_name in [
                            "anthropic/claude-sonnet-4-5-20250929",
                            "anthropic/claude-opus-4-1-20250805",
                        ] and "answerable-6" not in str(file_path):
                            continue

                        if has_error:
                            num_errors += 1

                        results["model_name"].append(model_name)
                        results["score"].append(score)
                        results["cost"].append(cost)

                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in {file_path}:{line_num} - {e}")
                        num_errors += 1

        except IOError as e:
            logger.error(f"Error reading file {file_path} - {e}")
            continue

    results_df = pd.DataFrame(results)
    print("=" * 40)
    print("MODEL COUNTS")
    print("=" * 40)
    print(results_df["model_name"].value_counts())
    print("=" * 40)
    return results, num_total, num_errors


def aggregate_model_stats(results: Dict[str, List]) -> Dict[str, Dict[str, List]]:
    """
    Aggregate results by model.

    Args:
        results: Dictionary with model_name, score, and cost lists

    Returns:
        Dictionary mapping model names to their scores and costs
    """
    model_stats = defaultdict(lambda: {"scores": [], "costs": []})

    for model, score, cost in zip(results["model_name"], results["score"], results["cost"]):
        model_stats[model]["scores"].append(score)
        model_stats[model]["costs"].append(cost)

    return model_stats


def format_leaderboard_output(model_stats: Dict[str, Dict[str, List]]) -> List[Dict]:
    """
    Format aggregated model statistics for YAML output.

    Args:
        model_stats: Dictionary of model statistics

    Returns:
        List of dictionaries formatted for YAML output
    """
    yaml_output = []

    for model_name in sorted(model_stats.keys()):
        scores = model_stats[model_name]["scores"]
        costs = model_stats[model_name]["costs"]

        avg_score = sum(scores) * 100 / len(scores)
        total_cost = sum(costs) * 100 / len(costs)

        yaml_output.append(
            {
                "model": model_name,
                "average": round(avg_score, 2),
                "total_cost": round(total_cost, 2),
                "leaderboard_filesystem_100": round(avg_score, 2),
            }
        )

    return yaml_output


def write_yaml_output(data: List[Dict], output_path: str = "leaderboard_results.yaml") -> None:
    """
    Write leaderboard data to a YAML file.

    Args:
        data: List of model result dictionaries
        output_path: Path to output YAML file
    """
    try:
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Results written to {output_path}")
    except IOError as e:
        logger.error(f"Error writing to {output_path} - {e}")
        raise


def print_summary(result_files: List[Path], yaml_output: List[Dict], num_total: int, num_errors: int) -> None:
    """
    Print a summary of the leaderboard generation.

    Args:
        result_files: List of result files
        yaml_output: Generated leaderboard data
        num_total: Total number of samples processed
        num_errors: Number of errors encountered
    """
    print("=" * 40)
    print("LEADERBOARD RESULTS SUMMARY")
    print("=" * 40)
    print(f"Total files:  {len(result_files)}")
    print(f"Total models:  {len(yaml_output)}")
    print(f"Total samples: {num_total}")
    print(f"Total errors:  {num_errors}")
    print("=" * 40 + "\n")


def main() -> None:
    """Main execution function."""
    result_files = find_result_files()
    if not result_files:
        logger.error("No result files found")
        return

    results, num_total, num_errors = load_results(result_files)
    model_stats = aggregate_model_stats(results)
    yaml_output = format_leaderboard_output(model_stats)
    write_yaml_output(yaml_output)
    print_summary(result_files, yaml_output, num_total, num_errors)


if __name__ == "__main__":
    main()
