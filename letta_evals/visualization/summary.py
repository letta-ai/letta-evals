from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from letta_evals.constants import MAX_SAMPLES_DISPLAY
from letta_evals.models import (
    SampleResult,
    SuiteSpec,
    Summary,
)


def print_basic_overall_metrics(console: Console, summary: Summary) -> None:
    """Print a compact overall summary, one line per model."""
    console.print("\n[bold]Overall Metrics:[/bold]")
    for ms in summary.models:
        n_errors = ms.n_total - ms.n_attempted
        errors_pct = (n_errors / ms.n_total * 100.0) if ms.n_total > 0 else 0.0
        console.print(f"  [cyan]{ms.model}[/]")
        console.print(f"    Total samples: {ms.n_total}")
        console.print(f"    Total attempted: {ms.n_attempted}")
        console.print(f"    Errored: {errors_pct:.1f}% ({n_errors}/{ms.n_total})")
        console.print(f"    Reward: {ms.reward:.2f}")


def get_metric_labels(suite: SuiteSpec) -> Dict[str, str]:
    """Build a metric_key -> display label mapping from a SuiteSpec."""
    metric_labels: Dict[str, str] = {}
    if suite.graders:
        for key, gspec in suite.graders.items():
            metric_labels[key] = gspec.display_name or key
    return metric_labels


def format_reward_description(
    suite: SuiteSpec, *, prefer_display_label: bool = False, quote_metric_label: bool = False
) -> str:
    """Format the reward description for summaries and validation output."""
    reward = suite.reward
    kind = getattr(getattr(reward, "kind", None), "value", getattr(reward, "kind", None))
    if kind == "metric":
        metric_key = getattr(reward, "metric_key", "")
        metric_label = get_metric_labels(suite).get(metric_key, metric_key) if prefer_display_label else metric_key
        if quote_metric_label and metric_label:
            metric_label = f"'{metric_label}'"
        return f"metric {metric_label}".strip()
    if kind == "custom":
        return f"custom {getattr(reward, 'function', '')}".strip()
    return f"unknown reward kind: {kind}"


def print_truncated_samples_notice(console: Console, total_samples: int, displayed_samples: int) -> None:
    if total_samples > displayed_samples:
        console.print(f"[dim]Showing first {displayed_samples} of {total_samples} samples[/dim]")


def print_remaining_samples_notice(console: Console, total_samples: int, displayed_samples: int) -> None:
    if total_samples > displayed_samples:
        console.print(
            f"[dim]... and {total_samples - displayed_samples} more samples (see output file for complete results)[/dim]"
        )


def extract_score_and_rationale(grade: Any) -> tuple[Optional[float], str]:
    """Coerce score and rationale from pydantic objects or dict-like grades."""
    try:
        score = float(getattr(grade, "score", None))
        rationale = getattr(grade, "rationale", None) or ""
        return score, rationale
    except Exception:
        try:
            score = float(grade.get("score"))
            rationale = grade.get("rationale", "") or ""
            return score, rationale
        except Exception:
            return None, ""


# Display rows pair each SampleResult with the model_id it ran under.
DisplayRow = Tuple[str, SampleResult]


def _sample_id_sort_key(sample_id):
    return (0, sample_id) if isinstance(sample_id, int) else (1, str(sample_id))


def get_displayed_sample_results(result: Any) -> tuple[int, List[DisplayRow]]:
    """Flatten the per-model results into a sorted list for display.

    Returns ``(total_samples, displayed_rows[:MAX])`` where each row is a
    ``(model_id, SampleResult)`` tuple. Falls back to ``[]`` if the result
    has no runs (e.g. interrupted before any sample completed).
    """
    runs = getattr(result, "runs", None) or {}
    rows: List[DisplayRow] = []
    for model_id, model_run in runs.items():
        for sample_result in model_run.results:
            rows.append((model_id, sample_result))

    rows.sort(key=lambda row: (row[0], _sample_id_sort_key(row[1].sample_id)))
    return len(rows), rows[:MAX_SAMPLES_DISPLAY]


def _format_reward(sample_result: SampleResult) -> str:
    return f"{sample_result.reward.score:.2f}" if sample_result.reward is not None else "-"


def build_simple_sample_results_table(suite: SuiteSpec, displayed_rows: List[DisplayRow]) -> Table:
    """Build the simple summary sample-results table."""
    table = Table(show_header=True)
    table.add_column("Sample", style="cyan")
    table.add_column("Agent ID", style="dim cyan")
    table.add_column("Model", style="yellow")
    table.add_column("Reward", style="white")

    metric_labels = get_metric_labels(suite)
    metric_keys = list(metric_labels.keys())

    for metric_key in metric_keys:
        table.add_column(f"{metric_labels[metric_key]} score", style="white")

    for model_id, sample_result in displayed_rows:
        cells = [_format_reward(sample_result)]
        for metric_key in metric_keys:
            grade = sample_result.grades.get(metric_key) if sample_result.grades else None
            if grade is None:
                cells.append("-")
            else:
                score, _ = extract_score_and_rationale(grade)
                cells.append(f"{score:.2f}" if score is not None else "-")

        table.add_row(
            f"Sample {sample_result.sample_id}",
            sample_result.agent_id or "-",
            model_id or "-",
            *cells,
        )

    return table


def build_rich_sample_results_table(suite: SuiteSpec, displayed_rows: List[DisplayRow]) -> Table:
    """Build the richer final sample-results table with rationales."""
    table = Table(show_header=True, header_style="bold cyan", border_style="blue", box=ROUNDED)
    table.add_column("Sample", style="cyan", no_wrap=True)
    table.add_column("Agent ID", style="dim cyan", no_wrap=False)
    table.add_column("Model", style="yellow", no_wrap=True)
    table.add_column("Reward", style="white", no_wrap=True)

    metric_labels = get_metric_labels(suite)
    metric_keys = list(metric_labels.keys())

    for metric_key in metric_keys:
        label = metric_labels[metric_key]
        table.add_column(f"{label} score", style="white", no_wrap=True)
        table.add_column(f"{label} rationale", style="dim", no_wrap=False)

    for model_id, sample_result in displayed_rows:
        cells = [_format_reward(sample_result)]
        for metric_key in metric_keys:
            grade = sample_result.grades.get(metric_key) if sample_result.grades else None
            if grade is None:
                cells.extend(["-", ""])
            else:
                score, rationale = extract_score_and_rationale(grade)
                cells.extend([f"{score:.2f}" if score is not None else "-", rationale])

        table.add_row(
            f"Sample {sample_result.sample_id}",
            sample_result.agent_id or "-",
            model_id or "-",
            *cells,
        )

    return table
