import sys
from pathlib import Path
from typing import Optional

import anyio
import typer
import yaml
from rich.console import Console
from rich.table import Table

from letta_evals.datasets.loader import load_jsonl
from letta_evals.models import RunnerResult, SuiteSpec
from letta_evals.runner import run_suite
from letta_evals.types import GraderKind
from letta_evals.visualization.progress import DisplayMode, EvalProgress

app = typer.Typer(help="Letta Evals - Evaluation framework for Letta AI agents")
console = Console()


@app.command()
def run(
    suite_path: Path = typer.Argument(..., help="Path to suite YAML file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save results to JSON file"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
    max_concurrent: int = typer.Option(15, "--max-concurrent", help="Maximum concurrent evaluations"),
    cached: Optional[Path] = typer.Option(
        None, "--cached", "-c", help="Path to cached results.json for re-grading trajectories"
    ),
):
    """Run an evaluation suite."""

    # auto-detect if we should disable fancy output based on terminal capabilities
    import os

    no_fancy = not console.is_terminal or os.getenv("NO_COLOR") is not None

    # verbose is now the default unless --quiet is specified
    verbose = not quiet

    if not suite_path.exists():
        console.print(f"[red]Error: Suite file not found: {suite_path}[/red]")
        raise typer.Exit(1)

    try:
        with open(suite_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        suite = SuiteSpec.from_yaml(yaml_data, base_dir=suite_path.parent)

        samples = list(load_jsonl(suite.dataset, max_samples=suite.max_samples, sample_tags=suite.sample_tags))
        num_samples = len(samples)

        # calculate total evaluations (samples × models)
        num_models = len(suite.target.model_configs) if suite.target.model_configs else 1
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
        if no_fancy or quiet:
            if not quiet:
                console.print(f"Running evaluation suite: {suite.name}")
                if cached:
                    console.print(f"[yellow]Re-grading {total_evaluations} cached trajectories...[/yellow]")
                else:
                    console.print(f"Evaluating {total_evaluations} samples...")
            return await run_suite(suite_path, max_concurrent=max_concurrent, cached_results_path=cached)
        else:
            rubric_model = None
            if suite.grader.kind == GraderKind.RUBRIC and hasattr(suite.grader, "model"):
                rubric_model = suite.grader.model

            progress = EvalProgress(
                suite_name=suite.name,
                total_samples=total_evaluations,
                target_kind=suite.target.kind.value,
                grader_kind=suite.grader.kind.value,
                rubric_model=rubric_model,
                max_concurrent=max_concurrent,
                display_mode=DisplayMode.DETAILED,
                console=console,
                show_samples=True,
                cached_mode=(cached is not None),
            )

            await progress.start()
            try:
                result = await run_suite(
                    suite_path, max_concurrent=max_concurrent, progress_callback=progress, cached_results_path=cached
                )
                return result
            finally:
                progress.stop()

    try:
        result = anyio.run(run_with_progress)  # type: ignore[arg-type]

        if not quiet:
            display_results(result, verbose, cached_mode=(cached is not None))

        if output:
            save_results(result, output)
            if not quiet:
                console.print(f"[green]Results saved to {output}[/green]")

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
        if verbose:
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
        console.print(f"  Grader: {suite.grader.kind.value}")
        if suite.gate:
            console.print(f"  Gate: {suite.gate.op.value} {suite.gate.value}")

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
    console.print("\n[dim]You can also use 'rubric' graders with custom prompts[/dim]")


def display_results(result: RunnerResult, verbose: bool = False, cached_mode: bool = False):
    console.print(f"\n[bold]Evaluation Results: {result.suite}[/bold]")
    if cached_mode:
        console.print("[dim]Note: Results re-graded from cached trajectories[/dim]")
    console.print("=" * 50)

    metrics = result.metrics
    console.print("\n[bold]Overall Metrics:[/bold]")
    console.print(f"  Total samples: {metrics.total}")
    console.print(f"  Total attempted: {metrics.total_attempted}")
    console.print(f"  Average score: {metrics.avg_score:.2f}")
    console.print(f"  Accuracy: {metrics.accuracy:.1f}%")

    # show per-model metrics if available
    if metrics.per_model:
        console.print("\n[bold]Per-Model Metrics:[/bold]")
        model_table = Table()
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Samples", style="white")
        model_table.add_column("Attempted", style="white")
        model_table.add_column("Avg Score", style="white")
        model_table.add_column("Accuracy", style="white")
        model_table.add_column("Passed", style="green")
        model_table.add_column("Failed", style="red")

        for model_metrics in metrics.per_model:
            model_table.add_row(
                model_metrics.model_name,
                str(model_metrics.total),
                str(model_metrics.total_attempted),
                f"{model_metrics.avg_score:.2f}",
                f"{model_metrics.accuracy:.1f}%",
                str(model_metrics.passed_samples),
                str(model_metrics.failed_samples),
            )

        console.print(model_table)

    gate = result.config["gate"]
    gate_op = gate["op"]
    gate_value = gate["value"]

    op_symbols = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "eq": "="}
    op_symbol = op_symbols.get(gate_op, gate_op)

    status = "[green]PASSED[/green]" if result.gates_passed else "[red]FAILED[/red]"
    console.print(
        f"\n[bold]Gate:[/bold] avg_score {op_symbol} {gate_value:.2f} → {status} (actual: {metrics.avg_score:.2f})"
    )

    if verbose:
        console.print("\n[bold]Sample Results:[/bold]")
        table = Table()
        table.add_column("Sample", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("Passed", style="white")
        table.add_column("Score", style="white")
        table.add_column("Rationale", style="dim")

        from letta_evals.models import GateSpec

        gate_spec = GateSpec(**result.config["gate"])

        for i, sample_result in enumerate(result.results):
            score_val = sample_result.grade.score
            passed = "✓" if gate_spec.check_score(score_val) else "✗"
            score = f"{score_val:.2f}"
            rationale = sample_result.grade.rationale or ""
            if len(rationale) > 50:
                rationale = rationale[:47] + "..."

            table.add_row(
                f"Sample {sample_result.sample.id + 1}", sample_result.model_name or "-", passed, score, rationale
            )

        console.print(table)


def save_results(result: RunnerResult, output_path: Path):
    result_json = result.model_dump_json(indent=2)

    with open(output_path, "w") as f:
        f.write(result_json)


if __name__ == "__main__":
    app()
