import sys
from pathlib import Path
from typing import Optional

import anyio
import typer
import yaml
from rich.console import Console
from rich.table import Table

from letta_evals.datasets.loader import load_dataset
from letta_evals.models import SuiteSpec
from letta_evals.runner import run_suite
from letta_evals.visualization.factory import ProgressStyle

app = typer.Typer(help="Letta Evals - Evaluation framework for Letta AI agents")
console = Console()


@app.command()
def run(
    suite_path: Path = typer.Argument(..., help="Path to suite YAML file"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Stream header, summary, and per-instance results to directory"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    display: Optional[str] = typer.Option(
        None, "--display", help="Display style: 'rich' (default), 'simple', or 'none'"
    ),
    max_concurrent: int = typer.Option(15, "--max-concurrent", help="Maximum concurrent evaluations"),
    cached: Optional[Path] = typer.Option(
        None, "--cached", "-c", help="Path to cached results (JSONL) for re-grading trajectories"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Letta API key override. If not provided, uses LETTA_API_KEY from environment",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="Letta base URL override. If omitted and an API key is set, defaults to Letta Cloud",
    ),
    project_id: Optional[str] = typer.Option(
        None,
        "--project-id",
        help="Letta project ID override. If not provided, uses LETTA_PROJECT_ID from environment or suite config",
    ),
    num_runs: Optional[int] = typer.Option(
        None,
        "--num-runs",
        help="Number of times to run the evaluation suite. Overrides suite config if provided.",
    ),
):
    """Run an evaluation suite."""

    # auto-detect if we should disable fancy output based on terminal capabilities
    import os

    no_fancy = not console.is_terminal or os.getenv("NO_COLOR") is not None

    if not suite_path.exists():
        console.print(f"[red]Error: Suite file not found: {suite_path}[/red]")
        raise typer.Exit(1)

    try:
        with open(suite_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_path.parent)

        samples = list(load_dataset(suite.dataset, max_samples=suite.max_samples, sample_tags=suite.sample_tags))
        num_samples = len(samples)

        # calculate total evaluations (samples × models)
        if suite.target.model_configs:
            num_models = len(suite.target.model_configs)
        elif suite.target.model_handles:
            num_models = len(suite.target.model_handles)
        else:
            num_models = 1
        total_evaluations = num_samples * num_models
    except Exception as e:
        console.print(f"[red]Error loading suite: {e}[/red]")
        raise typer.Exit(1)

    if not quiet and not no_fancy:
        console.print(f"[cyan]Loading suite: {suite.name}[/cyan]")
        if num_models > 1:
            console.print(
                f"[cyan]Total evaluations: {total_evaluations} ({num_samples} samples × {num_models} models)[/cyan]"
            )
        else:
            console.print(f"[cyan]Total samples: {num_samples}[/cyan]")
        console.print(f"[cyan]Max concurrent: {max_concurrent}[/cyan]")

        if cached:
            console.print(f"[yellow]Using cached trajectories from: {cached}[/yellow]")
            console.print(
                f"[yellow]Re-grading {total_evaluations} trajectories with updated grader configuration[/yellow]"
            )

    async def run_with_progress():
        # Choose built-in progress style for CLI
        if display:
            display_lower = display.lower()
            if display_lower == "none":
                style = ProgressStyle.NONE
            elif display_lower == "simple":
                style = ProgressStyle.SIMPLE
            elif display_lower == "rich":
                style = ProgressStyle.RICH
            else:
                console.print(f"[red]Error: Invalid display style '{display}'. Use 'rich', 'simple', or 'none'.[/red]")
                raise typer.Exit(1)
        elif quiet:
            style = ProgressStyle.NONE
        elif no_fancy:
            style = ProgressStyle.SIMPLE
        else:
            style = ProgressStyle.RICH

        if not quiet:
            console.print(f"Running evaluation suite: {suite.name}")
            if cached:
                console.print(f"[yellow]Re-grading {total_evaluations} cached trajectories...[/yellow]")
            else:
                console.print(f"Evaluating {total_evaluations} samples...")

        return await run_suite(
            suite_path,
            max_concurrent=max_concurrent,
            progress_style=style,
            cached_results_path=cached,
            output_path=output,
            letta_api_key=api_key,
            letta_base_url=base_url,
            letta_project_id=project_id,
            num_runs=num_runs,
        )

    try:
        result = anyio.run(run_with_progress)  # type: ignore[arg-type]

        if not quiet:
            # Display aggregate statistics if multiple runs
            if result.run_statistics is not None:
                display_aggregate_statistics(result.run_statistics)

        if output and not quiet:
            if result.run_statistics is not None:
                # Multiple runs - output to subdirectories
                num_runs_actual = result.run_statistics.num_runs
                console.print(
                    f"[green]Individual run results saved to {output}/run_1/ through {output}/run_{num_runs_actual}/[/green]"
                )
                console.print(f"[green]Aggregate statistics saved to {output}/aggregate_stats.json[/green]")
            else:
                # Single run - output to main directory
                console.print(f"[green]Results streamed to {output}/results.jsonl (JSONL)[/green]")
                console.print(f"[green]Summary saved to {output}/summary.json[/green]")
                console.print(f"[green]Header saved to {output}/header.json[/green]")

        if result.gates_passed:
            if not quiet:
                console.print("[green]✓ All gates passed[/green]")
            sys.exit(0)
        else:
            if not quiet:
                console.print("[red]✗ Some gates failed[/red]")
            sys.exit(1)

    except Exception as e:
        console.print(f"[red]Error running suite: {e}[/red]")
        if not quiet:
            import traceback

            traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def validate(suite_path: Path = typer.Argument(..., help="Path to suite YAML file")):
    """Validate a suite configuration without running it."""

    if not suite_path.exists():
        console.print(f"[red]Error: Suite file not found: {suite_path}[/red]")
        raise typer.Exit(1)

    try:
        with open(suite_path, "r") as f:
            yaml_data = yaml.safe_load(f)

        suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_path.parent)
        console.print(f"[green]✓ Suite '{suite.name}' is valid[/green]")

        console.print("\n[bold]Configuration:[/bold]")
        console.print(f"  Dataset: {suite.dataset}")
        console.print(f"  Target: {suite.target.kind.value}")
        if suite.graders:
            console.print("  Graders:")
            for key, gspec in suite.graders.items():
                label = gspec.display_name or key
                console.print(f"    - {label}: {gspec.kind.value}")
        if suite.gate:
            metric_key = suite.gate.metric_key or "<default>"
            console.print(
                f"  Gate: metric_key={metric_key} aggregate={suite.gate.metric.value} {suite.gate.op.value} {suite.gate.value}"
            )

    except Exception as e:
        console.print(f"[red]Invalid suite configuration: {e}[/red]")
        raise typer.Exit(1)


@app.command("list-extractors")
def list_extractors():
    """List available submission extractors."""

    from letta_evals.decorators import EXTRACTOR_REGISTRY

    table = Table(title="Available Extractors")
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="white")

    descriptions = {
        "last_assistant": "Extract the last assistant message",
        "first_assistant": "Extract the first assistant message",
        "all_assistant": "Concatenate all assistant messages",
        "last_turn": "Extract assistant messages from last turn",
        "pattern": "Extract using regex pattern",
        "json": "Extract JSON field from response",
        "tool_output": "Extract specific tool output",
        "after_marker": "Extract content after marker",
    }

    for name in sorted(EXTRACTOR_REGISTRY.keys()):
        desc = descriptions.get(name, "")
        table.add_row(name, desc)

    console.print(table)


@app.command("list-graders")
def list_graders():
    """List available built-in grader functions."""

    from letta_evals.graders.tool import GRADER_REGISTRY

    table = Table(title="Built-in Graders")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="yellow")

    for name in sorted(GRADER_REGISTRY.keys()):
        table.add_row(name, "tool")

    console.print(table)
    console.print("\n[dim]You can also use 'model_judge' or 'letta_judge' graders with custom prompts[/dim]")


def display_aggregate_statistics(run_statistics):
    """Display aggregate statistics across multiple runs."""
    from letta_evals.models import RunStatistics

    stats: RunStatistics = run_statistics

    console.print(f"\n[bold]Aggregate Statistics (across {stats.num_runs} runs):[/bold]")
    console.print("=" * 50)

    console.print("\n[bold]Run Summary:[/bold]")
    console.print(f"  Total runs: {stats.num_runs}")
    console.print(f"  Runs passed: {stats.runs_passed}")
    console.print(f"  Runs failed: {stats.num_runs - stats.runs_passed}")
    pass_rate = (stats.runs_passed / stats.num_runs * 100.0) if stats.num_runs > 0 else 0.0
    console.print(f"  Pass rate: {pass_rate:.1f}%")

    console.print("\n[bold]Average Score (Attempted):[/bold]")
    console.print(f"  Mean: {stats.mean_avg_score_attempted:.4f}")
    console.print(f"  Std Dev: {stats.std_avg_score_attempted:.4f}")

    console.print("\n[bold]Average Score (Total):[/bold]")
    console.print(f"  Mean: {stats.mean_avg_score_total:.4f}")
    console.print(f"  Std Dev: {stats.std_avg_score_total:.4f}")

    if stats.mean_scores:
        console.print("\n[bold]Per-Metric Statistics:[/bold]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Mean Score", style="white")
        table.add_column("Std Dev", style="white")

        for metric_key in stats.mean_scores.keys():
            mean = stats.mean_scores[metric_key]
            std = stats.std_scores.get(metric_key, 0.0)
            table.add_row(metric_key, f"{mean:.4f}", f"{std:.4f}")

        console.print(table)


if __name__ == "__main__":
    app()
