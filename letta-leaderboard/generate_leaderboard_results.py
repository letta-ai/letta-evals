"""
Leaderboard Results Generator

Processes evaluation results from aggregate_stats.json or summary.json files
and merges them with an existing leaderboard YAML file.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_model_name(model_name: str) -> str:
    """
    Normalize model names by adding provider prefixes if not already present.

    Args:
        model_name: Raw model name from results

    Returns:
        Normalized model name with provider prefix
    """
    if "/" in model_name:
        return model_name
    if model_name.startswith("claude"):
        return f"anthropic/{model_name}"
    if model_name.startswith("gpt"):
        return f"openai/{model_name}"
    if model_name.startswith("gemini"):
        return f"google/{model_name}"
    if model_name.startswith("mistral-"):
        return f"mistralai/{model_name}"
    if model_name.startswith("deepseek"):
        return f"deepseek/{model_name}"
    return model_name


def read_stats_file(directory: Path) -> Optional[Dict]:
    """
    Read aggregate_stats.json or fallback to summary.json from a directory.

    Args:
        directory: Path to directory containing stats files

    Returns:
        Dictionary containing the stats data, or None if neither file exists
    """
    aggregate_stats_path = directory / "aggregate_stats.json"
    summary_path = directory / "summary.json"

    if aggregate_stats_path.exists():
        logger.info(f"Reading aggregate_stats.json from {directory}")
        try:
            with open(aggregate_stats_path, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading {aggregate_stats_path}: {e}")
            return None

    elif summary_path.exists():
        logger.info(f"Reading summary.json from {directory}")
        try:
            with open(summary_path, "r") as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Error reading {summary_path}: {e}")
            return None

    else:
        logger.warning(f"Neither aggregate_stats.json nor summary.json found in {directory}")
        return None


def extract_model_results_from_aggregate(stats: Dict) -> List[Dict]:
    """
    Extract model results from aggregate_stats.json.

    Args:
        stats: Parsed aggregate_stats.json data

    Returns:
        List of model result dictionaries with name, score, cost, and individual metrics
    """
    model_results = []

    # Check if this is aggregate stats (has num_runs and individual_run_metrics)
    if "num_runs" in stats and "individual_run_metrics" in stats:
        # Aggregate model results across all runs
        model_data = {}

        for run_metrics in stats["individual_run_metrics"]:
            if "per_model" not in run_metrics:
                continue

            for model_info in run_metrics["per_model"]:
                model_name = normalize_model_name(model_info["model_name"])

                if model_name not in model_data:
                    model_data[model_name] = {
                        "scores": [],
                        "costs": [],
                        "metrics": {},
                    }

                # Score is already in percentage (e.g., 77.0)
                score = model_info.get("avg_score_attempted", 0) * 100
                cost_data = model_info.get("cost")
                cost = cost_data.get("total_cost", 0) if cost_data else 0

                model_data[model_name]["scores"].append(score)
                model_data[model_name]["costs"].append(cost)

                # Extract individual metrics (e.g., task_completion, skill_use)
                metrics_data = model_info.get("metrics", {})
                for metric_name, metric_value in metrics_data.items():
                    if metric_name not in model_data[model_name]["metrics"]:
                        model_data[model_name]["metrics"][metric_name] = []
                    model_data[model_name]["metrics"][metric_name].append(metric_value)

        # Compute averages
        for model_name, data in model_data.items():
            avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
            avg_cost = sum(data["costs"]) / len(data["costs"]) if data["costs"] else 0

            # Average individual metrics
            avg_metrics = {}
            for metric_name, metric_values in data["metrics"].items():
                avg_metrics[metric_name] = sum(metric_values) / len(metric_values) if metric_values else 0

            model_results.append(
                {
                    "model_name": model_name,
                    "score": avg_score,
                    "cost": avg_cost,
                    "metrics": avg_metrics,
                }
            )

    return model_results


def extract_model_results_from_summary(stats: Dict) -> List[Dict]:
    """
    Extract model results from summary.json.

    Args:
        stats: Parsed summary.json data

    Returns:
        List of model result dictionaries with name, score, cost, and individual metrics
    """
    model_results = []

    if "metrics" in stats and "per_model" in stats["metrics"]:
        for model_info in stats["metrics"]["per_model"]:
            model_name = normalize_model_name(model_info["model_name"])

            # Score is already in percentage (e.g., 77.0)
            score = model_info.get("avg_score_attempted", 0) * 100
            cost_data = model_info.get("cost")
            cost = cost_data.get("total_cost", 0) if cost_data else 0

            # Extract individual metrics (e.g., task_completion, skill_use)
            metrics_data = model_info.get("metrics", {})

            model_results.append(
                {
                    "model_name": model_name,
                    "score": score,
                    "cost": cost,
                    "metrics": metrics_data,
                }
            )

    return model_results


def load_leaderboard_yaml(yaml_path: Path) -> Dict:
    """
    Load existing leaderboard YAML file.

    Args:
        yaml_path: Path to the leaderboard YAML file

    Returns:
        Dictionary containing the leaderboard data
    """
    if not yaml_path.exists():
        logger.warning(f"Leaderboard file {yaml_path} does not exist. Creating new leaderboard.")
        return {"benchmark_name": "Benchmark", "metrics": {}, "results": []}

    try:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
            if data is None:
                data = {"benchmark_name": "Benchmark", "metrics": {}, "results": []}
            return data
    except (IOError, yaml.YAMLError) as e:
        logger.error(f"Error reading {yaml_path}: {e}")
        raise


def merge_results(existing_results: List[Dict], new_results: List[Dict], metric_key: str) -> List[Dict]:
    """
    Merge new results with existing leaderboard results.
    Handles both single-metric and multi-metric benchmarks.

    Args:
        existing_results: List of existing model results from leaderboard
        new_results: List of new model results to add
        metric_key: The metric key to use for single-metric benchmarks (e.g., 'leaderboard_filesystem_100')

    Returns:
        Merged list of model results
    """
    # Create a dictionary for easy lookup and update
    results_dict = {}

    # Check if existing results include total_cost field
    include_cost = False
    if existing_results:
        include_cost = "total_cost" in existing_results[0]

    # Add existing results
    for result in existing_results:
        model_name = result["model"]
        results_dict[model_name] = result.copy()

    # Merge/add new results
    for new_result in new_results:
        model_name = new_result["model_name"]
        cost = round(new_result["cost"], 2)

        # Check if this is a multi-metric benchmark (2+ individual metrics)
        individual_metrics = new_result.get("metrics", {})

        if individual_metrics and len(individual_metrics) > 1:
            # Multi-metric benchmark: populate each metric individually
            rounded_metrics = {k: round(v, 2) for k, v in individual_metrics.items()}

            # Calculate average across all metrics
            if rounded_metrics:
                avg_score = round(sum(rounded_metrics.values()) / len(rounded_metrics), 2)
            else:
                avg_score = round(new_result["score"], 2)

            if model_name in results_dict:
                # Update existing entry
                results_dict[model_name]["average"] = avg_score
                if include_cost:
                    results_dict[model_name]["total_cost"] = cost
                # Update all individual metrics
                for metric_name, metric_value in rounded_metrics.items():
                    results_dict[model_name][metric_name] = metric_value
                logger.info(f"Updated results for {model_name} (multi-metric)")
            else:
                # Add new entry
                entry = {
                    "model": model_name,
                    "average": avg_score,
                }
                if include_cost:
                    entry["total_cost"] = cost
                # Add all individual metrics
                entry.update(rounded_metrics)
                results_dict[model_name] = entry
                logger.info(f"Added new results for {model_name} (multi-metric)")
        else:
            # Single-metric benchmark: use the provided metric_key
            score = round(new_result["score"], 2)

            if model_name in results_dict:
                # Update existing entry
                results_dict[model_name]["average"] = score
                if include_cost:
                    results_dict[model_name]["total_cost"] = cost
                results_dict[model_name][metric_key] = score
                logger.info(f"Updated results for {model_name} (single-metric)")
            else:
                # Add new entry
                entry = {
                    "model": model_name,
                    "average": score,
                    metric_key: score,
                }
                if include_cost:
                    entry["total_cost"] = cost
                results_dict[model_name] = entry
                logger.info(f"Added new results for {model_name} (single-metric)")

    # Convert back to list and sort by model name
    return sorted(results_dict.values(), key=lambda x: x["model"])


def write_leaderboard_yaml(data: Dict, output_path: Path) -> None:
    """
    Write leaderboard data to a YAML file.

    Args:
        data: Dictionary containing leaderboard data
        output_path: Path to output YAML file
    """
    try:
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Updated leaderboard written to {output_path}")
    except IOError as e:
        logger.error(f"Error writing to {output_path}: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Generate or update leaderboard results from aggregate_stats.json or summary.json files"
    )
    parser.add_argument(
        "directories", nargs="+", type=Path, help="Directories containing aggregate_stats.json or summary.json files"
    )
    parser.add_argument(
        "--leaderboard", "-l", type=Path, required=True, help="Path to the leaderboard YAML file (input and output)"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="Path to output YAML file (defaults to same as --leaderboard)"
    )

    return parser.parse_args()


def extract_benchmark_name_from_path(directory: Path) -> str:
    """
    Extract benchmark name from directory path.

    Args:
        directory: Path to the results directory

    Returns:
        Benchmark name (e.g., 'filesystem', 'skills')
    """
    # Look for common benchmark patterns in the path
    path_str = str(directory)
    if "filesystem" in path_str.lower():
        return "filesystem"
    elif "skill" in path_str.lower():
        return "skills"
    elif "memory" in path_str.lower():
        return "memory"

    # Default to the directory name
    return directory.name


def main() -> None:
    """Main execution function."""
    args = parse_arguments()

    # Load existing leaderboard
    leaderboard_data = load_leaderboard_yaml(args.leaderboard)

    # Extract metric key from existing results or generate from benchmark name
    existing_results = leaderboard_data.get("results", [])
    metric_key = None

    # Try to find the metric key from existing results
    if existing_results:
        # Look for a key that starts with "leaderboard_" and ends with "_100"
        for key in existing_results[0].keys():
            if key.startswith("leaderboard_") and key.endswith("_100"):
                metric_key = key
                break

    # If no metric key found, try to extract from metrics section
    if not metric_key:
        metrics = leaderboard_data.get("metrics", {})
        for key in metrics.keys():
            if key.startswith("leaderboard_") and key.endswith("_100"):
                metric_key = key
                break

    # If still no metric key, generate one from benchmark name or directory
    if not metric_key:
        benchmark_name = leaderboard_data.get("benchmark_name", "benchmark")
        if benchmark_name == "Benchmark" and args.directories:
            benchmark_name = extract_benchmark_name_from_path(args.directories[0])
        # Normalize benchmark name to lowercase without spaces
        normalized_name = benchmark_name.lower().replace(" ", "_").replace("-", "_")
        metric_key = f"leaderboard_{normalized_name}_100"

    logger.info(f"Using metric key: {metric_key}")

    # Collect all model results from all directories
    all_new_results = []

    for directory in args.directories:
        if not directory.exists():
            logger.warning(f"Directory {directory} does not exist, skipping")
            continue

        # Read stats file
        stats = read_stats_file(directory)
        if stats is None:
            continue

        # Extract model results based on file type
        if "num_runs" in stats:
            model_results = extract_model_results_from_aggregate(stats)
        elif "metrics" in stats:
            model_results = extract_model_results_from_summary(stats)
        else:
            logger.warning(f"Unknown stats format in {directory}, skipping")
            continue

        all_new_results.extend(model_results)
        logger.info(f"Extracted {len(model_results)} model results from {directory}")

    if not all_new_results:
        logger.error("No results found in any of the specified directories")
        return

    # Merge results
    merged_results = merge_results(existing_results, all_new_results, metric_key)

    # Update leaderboard data
    leaderboard_data["results"] = merged_results

    # Write output
    output_path = args.output or args.leaderboard
    write_leaderboard_yaml(leaderboard_data, output_path)

    # Print summary
    print("\n" + "=" * 80)
    print("LEADERBOARD UPDATE SUMMARY")
    print("=" * 80)
    print(f"Processed directories: {len(args.directories)}")
    print(f"New/updated models: {len(all_new_results)}")
    print(f"Total models in leaderboard: {len(merged_results)}")
    print(f"Output written to: {output_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
