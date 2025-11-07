#!/usr/bin/env python3
"""
Script to collect results from summary.json files and create plots for task
completion and skill use scores across different settings.

Usage:
    python analyze_results.py [results_directory]

    If no directory is specified, defaults to 'results_old'
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def collect_results(results_dir="results_old"):
    """Collect all summary.json files and extract metrics by model."""
    # Structure: results[model][setting] = {"task_completion": X, "skill_use": Y}
    results = defaultdict(lambda: defaultdict(dict))

    # Find all summary.json files
    results_path = Path(results_dir)
    summary_files = list(results_path.glob("*/summary.json"))

    print(f"Found {len(summary_files)} summary.json files\n")

    for summary_file in summary_files:
        dir_name = summary_file.parent.name

        # Determine setting (baseline, oracle, or selection)
        if "baseline" in dir_name:
            setting = "baseline"
        elif "oracle" in dir_name:
            setting = "oracle"
        elif "selection" in dir_name:
            setting = "selection"
        else:
            print(f"Unknown setting for {dir_name}, skipping")
            continue

        # Load and extract metrics from per_model array
        with open(summary_file, "r") as f:
            data = json.load(f)

        per_model = data.get("metrics", {}).get("per_model", [])

        print(f"{dir_name} ({setting}):")
        for model_data in per_model:
            model_name = model_data.get("model_name", "unknown")
            metrics = model_data.get("metrics", {})
            task_completion = metrics.get("task_completion", 0)
            skill_use = metrics.get("skill_use", 0)

            results[model_name][setting] = {"task_completion": task_completion, "skill_use": skill_use}

            print(f"  - {model_name}: task_completion={task_completion:.2f}, skill_use={skill_use:.2f}")
        print()

    return results


def create_plot(results, metric_name, ylabel, title, filename):
    """Create a grouped bar plot with models on x-axis and settings as bars."""
    settings = ["baseline", "oracle", "selection"]

    # Get all models and sort them
    models = sorted(results.keys())

    # Simplify model names for display
    def simplify_model_name(name):
        if "claude-sonnet-4-5" in name:
            return "Sonnet 4.5"
        elif "claude-haiku-4-5" in name:
            return "Haiku 4.5"
        elif "claude-opus-4" in name:
            return "Opus 4"
        elif "gpt-4o-mini" in name:
            return "GPT-4o-mini"
        elif "gpt-4o" in name:
            return "GPT-4o"
        elif "gemini-2.0" in name:
            return "Gemini 2.0"
        elif "gemini-1.5-pro" in name:
            return "Gemini 1.5 Pro"
        elif "gemini-1.5-flash" in name:
            return "Gemini 1.5 Flash"
        elif "deepseek" in name:
            return "DeepSeek"
        return name

    model_display_names = [simplify_model_name(m) for m in models]

    # Setting colors
    colors = {
        "baseline": "#3498db",  # Blue
        "oracle": "#e74c3c",  # Red
        "selection": "#2ecc71",  # Green
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(models))
    width = 0.25
    multiplier = 0

    for setting in settings:
        values = []
        for model in models:
            if setting in results[model]:
                values.append(results[model][setting][metric_name])
            else:
                values.append(0)

        offset = width * multiplier
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=setting.capitalize(),
            color=colors.get(setting, "#999999"),
            edgecolor="black",
            linewidth=1,
            alpha=0.85,
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        multiplier += 1

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_display_names, fontsize=10, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()


def print_summary_table(results):
    """Print a summary table of all results by model."""
    print("\n" + "=" * 110)
    print("SUMMARY TABLE (by Model)")
    print("=" * 110)

    settings = ["baseline", "oracle", "selection"]

    print(f"{'Model':<50} {'Setting':<15} {'Task Completion':<20} {'Skill Use':<15}")
    print("-" * 110)

    for model in sorted(results.keys()):
        for setting in settings:
            if setting in results[model]:
                task_comp = results[model][setting].get("task_completion", 0)
                skill_use = results[model][setting].get("skill_use", 0)
                print(f"{model:<50} {setting:<15} {task_comp:<20.2f} {skill_use:<15.2f}")
            else:
                print(f"{model:<50} {setting:<15} {'N/A':<20} {'N/A':<15}")
        print()

    print("=" * 110)


def create_aggregated_plot(results, filename):
    """Create a grouped bar plot with mean of task completion and skill use."""
    settings = ["baseline", "oracle", "selection"]

    # Get all models and sort them
    models = sorted(results.keys())

    # Simplify model names for display
    def simplify_model_name(name):
        if "claude-sonnet-4-5" in name:
            return "Sonnet 4.5"
        elif "claude-haiku-4-5" in name:
            return "Haiku 4.5"
        elif "claude-opus-4" in name:
            return "Opus 4"
        elif "gpt-4o-mini" in name:
            return "GPT-4o-mini"
        elif "gpt-4o" in name:
            return "GPT-4o"
        elif "gemini-2.0" in name:
            return "Gemini 2.0"
        elif "gemini-1.5-pro" in name:
            return "Gemini 1.5 Pro"
        elif "gemini-1.5-flash" in name:
            return "Gemini 1.5 Flash"
        elif "deepseek" in name:
            return "DeepSeek"
        return name

    model_display_names = [simplify_model_name(m) for m in models]

    # Setting colors
    colors = {
        "baseline": "#3498db",  # Blue
        "oracle": "#e74c3c",  # Red
        "selection": "#2ecc71",  # Green
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 7))

    x = np.arange(len(models))
    width = 0.25
    multiplier = 0

    for setting in settings:
        values = []
        for model in models:
            if setting in results[model]:
                task_comp = results[model][setting]["task_completion"]
                skill_use = results[model][setting]["skill_use"]
                # Calculate mean of both metrics
                mean_score = (task_comp + skill_use) / 2
                values.append(mean_score)
            else:
                values.append(0)

        offset = width * multiplier
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=setting.capitalize(),
            color=colors.get(setting, "#999999"),
            edgecolor="black",
            linewidth=1,
            alpha=0.85,
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

        multiplier += 1

    ax.set_xlabel("Model", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score (%)", fontsize=14, fontweight="bold")
    ax.set_title("Average Score (Task Completion + Skill Use) Across Settings", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_display_names, fontsize=10, rotation=45, ha="right")
    ax.legend(loc="upper left", fontsize=12, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()


def create_setting_aggregation_plot(results, filename):
    """Create a bar plot with 3 bars showing average scores across all models for each setting."""
    settings = ["baseline", "oracle", "selection"]

    # Calculate average scores across all models for each setting
    setting_averages = {}

    for setting in settings:
        task_comp_scores = []
        skill_use_scores = []

        # Collect all scores for this setting across all models
        for model in results.keys():
            if setting in results[model]:
                task_comp_scores.append(results[model][setting]["task_completion"])
                skill_use_scores.append(results[model][setting]["skill_use"])

        # Calculate average of both metrics combined
        if task_comp_scores and skill_use_scores:
            avg_task_comp = np.mean(task_comp_scores)
            # avg_skill_use = np.mean(skill_use_scores)
            # Average of task completion and skill use
            # setting_averages[setting] = (avg_task_comp + avg_skill_use) / 2
            setting_averages[setting] = avg_task_comp
        else:
            setting_averages[setting] = 0

    # Setting colors
    colors = {
        "baseline": "#3498db",  # Blue
        "oracle": "#e74c3c",  # Red
        "selection": "#2ecc71",  # Green
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    x = np.arange(len(settings))
    values = [setting_averages[s] for s in settings]
    bar_colors = [colors[s] for s in settings]

    bars = ax.bar(x, values, width=0.6, color=bar_colors, edgecolor="black", linewidth=2, alpha=0.85)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    ax.set_xlabel("Setting", fontsize=14, fontweight="bold")
    ax.set_ylabel("Average Score (%)", fontsize=14, fontweight="bold")
    ax.set_title("Average Score Across All Models by Setting", fontsize=16, fontweight="bold", pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in settings], fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, max(values) * 1.15)  # Add some space at the top for labels

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {filename}")
    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze results from summary.json files and create plots.")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default="results_old",
        help="Directory containing result folders with summary.json files (default: results_old)",
    )
    parser.add_argument("--output-dir", default="plots", help="Directory to save plots (default: plots)")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}\n")

    print(f"Collecting results from {args.results_dir}...\n")
    results = collect_results(args.results_dir)

    # Print summary table
    print_summary_table(results)

    # Create plots
    print("\nCreating plots...")

    create_plot(
        results,
        metric_name="task_completion",
        ylabel="Task Completion Score (%)",
        title="Task Completion Scores Across Settings",
        filename=output_dir / "task_completion_plot.png",
    )

    create_plot(
        results,
        metric_name="skill_use",
        ylabel="Skill Use Score (%)",
        title="Skill Use Scores Across Settings",
        filename=output_dir / "skill_use_plot.png",
    )

    create_aggregated_plot(results, filename=output_dir / "aggregated_score_plot.png")

    create_setting_aggregation_plot(results, filename=output_dir / "setting_aggregation_plot.png")

    print("\nAnalysis complete!")
    print(f"All plots saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
