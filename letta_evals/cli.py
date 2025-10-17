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
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Stream header, summary, and per-instance results to directory"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Minimal output"),
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
        if no_fancy or quiet:
            if not quiet:
                console.print(f"Running evaluation suite: {suite.name}")
                if cached:
                    console.print(f"[yellow]Re-grading {total_evaluations} cached trajectories...[/yellow]")
                else:
                    console.print(f"Evaluating {total_evaluations} samples...")
            return await run_suite(
                suite_path,
                max_concurrent=max_concurrent,
                cached_results_path=cached,
                output_path=output,
                letta_api_key=api_key,
                letta_base_url=base_url,
                letta_project_id=project_id,
            )
        else:
            rubric_model = None
            if suite.graders:
                # choose a rubric model if any grader is rubric
                for key, gspec in suite.graders.items():
                    if gspec.kind == GraderKind.RUBRIC and hasattr(gspec, "model"):
                        rubric_model = gspec.model
                        break

            # Build metric labels mapping for live progress (key -> display)
            metric_labels = None
            if suite.graders:
                metric_labels = {key: (gspec.display_name or key) for key, gspec in suite.graders.items()}
            # Determine grader kind label for progress
            if suite.graders and len(suite.graders) == 1:
                only_kind = next(iter(suite.graders.values())).kind.value
                grader_kind_label = only_kind
            else:
                grader_kind_label = "multi"

            progress = EvalProgress(
                suite_name=suite.name,
                total_samples=total_evaluations,
                target_kind=suite.target.kind.value,
                grader_kind=grader_kind_label,
                rubric_model=rubric_model,
                max_concurrent=max_concurrent,
                display_mode=DisplayMode.DETAILED,
                console=console,
                show_samples=True,
                cached_mode=(cached is not None),
                metric_labels=metric_labels,
            )

            await progress.start()
            try:
                result = await run_suite(
                    suite_path,
                    max_concurrent=max_concurrent,
                    progress_callback=progress,
                    cached_results_path=cached,
                    output_path=output,
                    letta_api_key=api_key,
                    letta_base_url=base_url,
                    letta_project_id=project_id,
                )
                return result
            finally:
                progress.stop()

    try:
        result = anyio.run(run_with_progress)  # type: ignore[arg-type]

        if not quiet:
            display_results(result, verbose, cached_mode=(cached is not None))

        if output and not quiet:
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
    errors = metrics.total - metrics.total_attempted
    errors_pct = (errors / metrics.total * 100.0) if metrics.total > 0 else 0.0
    console.print(f"  Errored: {errors_pct:.1f}% ({errors}/{metrics.total})")
    console.print(f"  Average score (gate metric): {metrics.avg_score:.2f}")
    console.print(f"  Passed attempts (gate metric): {metrics.passed_attempts}")
    console.print(f"  Failed attempts (gate metric): {metrics.failed_attempts}")

    # Print per-metric aggregates if available
    if hasattr(metrics, "by_metric") and metrics.by_metric:
        console.print("\n[bold]Metrics by Metric:[/bold]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Avg Score", style="white")
        # Build key->label mapping from config
        label_map = {}
        if "graders" in result.config and isinstance(result.config["graders"], dict):
            for key, gspec in result.config["graders"].items():
                label_map[key] = gspec.get("display_name") or key

        for key, agg in metrics.by_metric.items():
            label = label_map.get(key, key)
            table.add_row(label, f"{agg.avg_score:.2f}")
        console.print(table)

    # show per-model metrics if available
    if metrics.per_model:
        console.print("\n[bold]Per-Model Metrics:[/bold]")
        model_table = Table()
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Samples", style="white")
        model_table.add_column("Attempted", style="white")
        model_table.add_column("Avg Score", style="white")
        model_table.add_column("Passed", style="green")
        model_table.add_column("Failed", style="red")

        for model_metrics in metrics.per_model:
            model_table.add_row(
                model_metrics.model_name,
                str(model_metrics.total),
                str(model_metrics.total_attempted),
                f"{model_metrics.avg_score:.2f}",
                str(model_metrics.passed_samples),
                str(model_metrics.failed_samples),
            )

        console.print(model_table)

    gate = result.config["gate"]
    gate_op = gate["op"]
    gate_value = gate["value"]
    gate_metric = gate.get("metric", "avg_score")
    gate_metric_key = gate.get("metric_key")

    op_symbols = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "eq": "="}
    op_symbol = op_symbols.get(gate_op, gate_op)

    status = "[green]PASSED[/green]" if result.gates_passed else "[red]FAILED[/red]"
    actual = metrics.avg_score if gate_metric == "avg_score" else metrics.accuracy
    suffix = "%" if gate_metric == "accuracy" else ""
    # Prefer display name for gate metric key
    display_label = None
    if gate_metric_key and "graders" in result.config and isinstance(result.config["graders"], dict):
        gspec = result.config["graders"].get(gate_metric_key)
        if gspec:
            display_label = gspec.get("display_name")
    metric_key_suffix = f" on '{display_label or gate_metric_key}'" if gate_metric_key else ""
    console.print(
        f"\n[bold]Gate:{metric_key_suffix}[/bold] {gate_metric} {op_symbol} {gate_value:.2f}{suffix} → {status} (actual: {actual:.2f}{suffix})"
    )

    if verbose:
        console.print("\n[bold]Sample Results:[/bold]")
        table = Table()
        table.add_column("Sample", style="cyan")
        table.add_column("Model", style="yellow")
        table.add_column("Passed", style="white")

        # Determine available metrics and display labels
        metric_keys = []
        metric_labels = {}
        if "graders" in result.config and isinstance(result.config["graders"], dict):
            for k, gspec in result.config["graders"].items():
                metric_keys.append(k)
                metric_labels[k] = gspec.get("display_name") or k

        # Add two sub-columns per metric: score + rationale
        for mk in metric_keys:
            lbl = metric_labels.get(mk, mk)
            table.add_column(f"{lbl} score", style="white")
            table.add_column(f"{lbl} rationale", style="dim")

        from letta_evals.models import GateSpec

        gate_spec = GateSpec(**result.config["gate"])

        for sample_result in result.results:
            score_val = sample_result.grade.score
            passed = "✓" if gate_spec.check_sample(score_val) else "✗"

            # Build per-metric cells in config order
            cells = []
            for mk in metric_keys:
                g = sample_result.grades.get(mk) if sample_result.grades else None
                if g is None:
                    cells.extend(["-", ""])
                else:
                    try:
                        s_val = float(getattr(g, "score", None))
                        r_text = getattr(g, "rationale", None) or ""
                    except Exception:
                        # dict fallback
                        try:
                            s_val = float(g.get("score"))  # type: ignore[attr-defined]
                            r_text = g.get("rationale", "")  # type: ignore[attr-defined]
                        except Exception:
                            s_val = None
                            r_text = ""
                    score_cell = f"{s_val:.2f}" if s_val is not None else "-"
                    if r_text and len(r_text) > 50:
                        r_text = r_text[:47] + "..."
                    cells.extend([score_cell, r_text])

            table.add_row(f"Sample {sample_result.sample.id + 1}", sample_result.model_name or "-", passed, *cells)

        console.print(table)


if __name__ == "__main__":
    app()
