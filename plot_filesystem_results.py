#!/usr/bin/env python3
"""
Script to plot results from filesystem evaluation runs.
Plots average scores across 3 runs with standard deviation error bars.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Use the ggplot style
plt.style.use("ggplot")
# Define the result directories
result_dirs = [
    "leaderboard_10172025/filesystem-results-oai-anthropic-no-answerable/run_1",
    "leaderboard_10172025/filesystem-results-oai-anthropic-no-answerable/run_2",
    "leaderboard_10172025/filesystem-results-oai-anthropic-no-answerable/run_3",
]




def load_results(base_path):
    """Load results from all three runs."""
    all_results = {}
    all_attempted = {}

    for result_dir in result_dirs:
        summary_path = Path(base_path) / result_dir / "summary.json"

        with open(summary_path, "r") as f:
            data = json.load(f)

        # Extract per-model results
        for model_data in data["metrics"]["per_model"]:
            model_name = model_data["model_name"]
            avg_score_total = model_data["avg_score_total"]
            total_attempted = model_data["total_attempted"]

            if model_name not in all_results:
                all_results[model_name] = []
                all_attempted[model_name] = []

            all_results[model_name].append(avg_score_total)
            all_attempted[model_name].append(total_attempted)

    return all_results, all_attempted


def plot_results(results, output_file="filesystem_results_comparison.png"):
    """Create a bar plot with error bars."""
    # Filter out models with insufficient data
    filtered_results = {
        model: scores
        for model, scores in results.items()
        if len(scores) == 3 and model != "moonshotai/Kimi-K2-Instruct-0905"
    }

    # Calculate averages and standard deviations
    models = list(filtered_results.keys())
    averages = [np.mean(filtered_results[model]) for model in models]
    stds = [np.std(filtered_results[model]) for model in models]

    # Sort by average score (descending)
    sorted_indices = np.argsort(averages)[::-1]
    models = [models[i] for i in sorted_indices]
    averages = [averages[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bar positions
    x_pos = np.arange(len(models))

    # Define colors for different model families
    colors = []
    for model in models:
        if "claude" in model.lower():
            colors.append("#E67E22")  # Orange for Claude
        elif "gpt-5" in model.lower():
            colors.append("#3498DB")  # Blue for GPT-5
        elif "gpt-4" in model.lower():
            colors.append("#9B59B6")  # Purple for GPT-4
        else:
            colors.append("#95A5A6")  # Gray for others

    # Create bars with error bars
    bars = ax.bar(x_pos, averages, yerr=stds, align="center", alpha=0.8, ecolor="black", capsize=5, color=colors)

    # Customize the plot
    ax.set_ylabel("Average Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Filesystem Eval", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, max(averages) * 1.2)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for i, (avg, std) in enumerate(zip(averages, stds)):
        ax.text(i, avg + std + 0.02, f"{avg:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add legend for model families
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#E67E22", alpha=0.8, label="Claude"),
        Patch(facecolor="#3498DB", alpha=0.8, label="GPT-5"),
        Patch(facecolor="#9B59B6", alpha=0.8, label="GPT-4"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Also print statistics
    print("\n" + "=" * 70)
    print("Model Performance Statistics (Across 3 Runs)")
    print("=" * 70)
    print(f"{'Model':<40} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    for model, avg, std in zip(models, averages, stds):
        print(f"{model:<40} {avg:.4f}    {std:.4f}")
    print("=" * 70)

    plt.show()


def plot_attempted(attempted, output_file="filesystem_attempted_comparison.png"):
    """Create a bar plot showing number of attempts with error bars."""
    # Filter out models with insufficient data
    filtered_attempted = {
        model: attempts
        for model, attempts in attempted.items()
        if len(attempts) == 3 and model != "moonshotai/Kimi-K2-Instruct-0905"
    }

    # Calculate averages and standard deviations
    models = list(filtered_attempted.keys())
    averages = [np.mean(filtered_attempted[model]) for model in models]
    stds = [np.std(filtered_attempted[model]) for model in models]

    # Sort by average number attempted (descending)
    sorted_indices = np.argsort(averages)[::-1]
    models = [models[i] for i in sorted_indices]
    averages = [averages[i] for i in sorted_indices]
    stds = [stds[i] for i in sorted_indices]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bar positions
    x_pos = np.arange(len(models))

    # Define colors for different model families
    colors = []
    for model in models:
        if "claude" in model.lower():
            colors.append("#E67E22")  # Orange for Claude
        elif "gpt-5" in model.lower():
            colors.append("#3498DB")  # Blue for GPT-5
        elif "gpt-4" in model.lower():
            colors.append("#9B59B6")  # Purple for GPT-4
        else:
            colors.append("#95A5A6")  # Gray for others

    # Create bars with error bars
    bars = ax.bar(x_pos, averages, yerr=stds, align="center", alpha=0.8, ecolor="black", capsize=5, color=colors)

    # Customize the plot
    ax.set_ylabel("Number Attempted (out of 100)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_title("Filesystem Eval - Number of Tasks Attempted", fontsize=14, fontweight="bold", pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 105)  # Max is 100 tasks

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Add value labels on top of bars
    for i, (avg, std) in enumerate(zip(averages, stds)):
        ax.text(i, avg + std + 1, f"{avg:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Add legend for model families
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#E67E22", alpha=0.8, label="Claude"),
        Patch(facecolor="#3498DB", alpha=0.8, label="GPT-5"),
        Patch(facecolor="#9B59B6", alpha=0.8, label="GPT-4"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_file}")

    # Also print statistics
    print("\n" + "=" * 70)
    print("Number Attempted Statistics (Across 3 Runs)")
    print("=" * 70)
    print(f"{'Model':<40} {'Mean':<10} {'Std':<10}")
    print("-" * 70)
    for model, avg, std in zip(models, averages, stds):
        print(f"{model:<40} {avg:.2f}      {std:.2f}")
    print("=" * 70)

    plt.show()


def main():
    # Get the base path (script directory)
    base_path = Path(__file__).parent

    # Load results
    print("Loading results from 3 runs...")
    results, attempted = load_results(base_path)

    print(f"Found results for {len(results)} models")

    # Plot results
    plot_results(results)
    
    # Plot attempted
    plot_attempted(attempted)


if __name__ == "__main__":
    main()
