from __future__ import annotations

from typing import Any, Dict, Optional

from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table

from letta_evals.constants import MAX_SAMPLES_DISPLAY


def print_basic_overall_metrics(console: Console, metrics: Any) -> None:
    """Print the shared overall metrics summary used by built-in progress views."""
    console.print("\n[bold]Overall Metrics:[/bold]")
    console.print(f"  Total samples: {metrics.total}")
    console.print(f"  Total attempted: {metrics.total_attempted}")
    errors = metrics.total - metrics.total_attempted
    errors_pct = (errors / metrics.total * 100.0) if metrics.total > 0 else 0.0
    console.print(f"  Errored: {errors_pct:.1f}% ({errors}/{metrics.total})")
    console.print(f"  Average score (attempted): {metrics.avg_score_attempted:.2f}")
    console.print(f"  Average score (total): {metrics.avg_score_total:.2f}")


def get_metric_labels(config: Dict[str, Any]) -> Dict[str, str]:
    """Build a metric_key -> display label mapping from the runner config."""
    metric_labels: Dict[str, str] = {}
    if "graders" in config and isinstance(config["graders"], dict):
        for key, gspec in config["graders"].items():
            metric_labels[key] = gspec.get("display_name") or key
    return metric_labels


def format_gate_description(
    config: Dict[str, Any],
    *,
    prefer_display_label: bool = False,
    quote_metric_label: bool = False,
    default_metric_label: str = "",
    fixed_decimal_value: bool = False,
) -> str:
    """Format the gate description for final summaries."""
    gate = config["gate"]
    gate_kind = gate.get("kind", "simple")
    op_symbols = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "eq": "="}

    if gate_kind == "simple":
        gate_op = gate["op"]
        gate_value = gate["value"]
        gate_aggregation = gate.get("aggregation", "avg_score")
        gate_metric_key = gate.get("metric_key")
        op_symbol = op_symbols.get(gate_op, gate_op)

        metric_label = gate_metric_key or default_metric_label
        if prefer_display_label and gate_metric_key:
            metric_label = get_metric_labels(config).get(gate_metric_key, gate_metric_key)

        if quote_metric_label and metric_label:
            metric_label = f"'{metric_label}'"

        gate_value_text = f"{gate_value:.2f}" if fixed_decimal_value else f"{gate_value}"
        return f"{metric_label} {gate_aggregation} {op_symbol} {gate_value_text}".strip()

    if gate_kind == "weighted_average":
        weights = gate.get("weights", {})
        gate_op = gate["op"]
        gate_value = gate["value"]
        gate_aggregation = gate.get("aggregation", "avg_score")
        op_symbol = op_symbols.get(gate_op, gate_op)
        weight_strs = [f"{k}({w})" for k, w in weights.items()]
        return f"weighted_average[{', '.join(weight_strs)}] {gate_aggregation} {op_symbol} {gate_value}"

    if gate_kind == "logical":
        operator = gate.get("operator", "and").upper()
        num_conditions = len(gate.get("conditions", []))
        return f"logical {operator} with {num_conditions} conditions"

    return f"unknown gate kind: {gate_kind}"


def print_truncated_samples_notice(console: Console, total_samples: int, displayed_samples: int) -> None:
    """Print the shared truncated sample-results notice when needed."""
    if total_samples > displayed_samples:
        console.print(f"[dim]Showing first {displayed_samples} of {total_samples} samples[/dim]")


def print_remaining_samples_notice(console: Console, total_samples: int, displayed_samples: int) -> None:
    """Print the shared tail notice after the sample results table when needed."""
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


def get_displayed_sample_results(result: Any) -> tuple[int, list[Any]]:
    """Sort and truncate sample results for final reporting."""
    total_samples = len(result.results)
    sorted_results = sorted(
        result.results, key=lambda sample_result: (sample_result.model_name or "", sample_result.sample.id)
    )
    return total_samples, sorted_results[:MAX_SAMPLES_DISPLAY]


def build_simple_sample_results_table(config: Dict[str, Any], displayed_results: list[Any]) -> Table:
    """Build the simple summary sample-results table."""
    table = Table(show_header=True)
    table.add_column("Sample", style="cyan")
    table.add_column("Agent ID", style="dim cyan")
    table.add_column("Model", style="yellow")

    metric_labels = get_metric_labels(config)
    metric_keys = list(metric_labels.keys())

    for metric_key in metric_keys:
        table.add_column(f"{metric_labels[metric_key]} score", style="white")

    for sample_result in displayed_results:
        cells = []
        for metric_key in metric_keys:
            grade = sample_result.grades.get(metric_key) if sample_result.grades else None
            if grade is None:
                cells.append("-")
            else:
                score, _ = extract_score_and_rationale(grade)
                cells.append(f"{score:.2f}" if score is not None else "-")

        table.add_row(
            f"Sample {sample_result.sample.id + 1}",
            sample_result.agent_id or "-",
            sample_result.model_name or "-",
            *cells,
        )

    return table


def build_rich_sample_results_table(config: Dict[str, Any], displayed_results: list[Any]) -> Table:
    """Build the richer final sample-results table with rationales."""
    table = Table(show_header=True, header_style="bold cyan", border_style="blue", box=ROUNDED)
    table.add_column("Sample", style="cyan", no_wrap=True)
    table.add_column("Agent ID", style="dim cyan", no_wrap=False)
    table.add_column("Model", style="yellow", no_wrap=True)

    metric_labels = get_metric_labels(config)
    metric_keys = list(metric_labels.keys())

    for metric_key in metric_keys:
        label = metric_labels[metric_key]
        table.add_column(f"{label} score", style="white", no_wrap=True)
        table.add_column(f"{label} rationale", style="dim", no_wrap=False)

    for sample_result in displayed_results:
        cells = []
        for metric_key in metric_keys:
            grade = sample_result.grades.get(metric_key) if sample_result.grades else None
            if grade is None:
                cells.extend(["-", ""])
            else:
                score, rationale = extract_score_and_rationale(grade)
                cells.extend([f"{score:.2f}" if score is not None else "-", rationale])

        table.add_row(
            f"Sample {sample_result.sample.id + 1}",
            sample_result.agent_id or "-",
            sample_result.model_name or "-",
            *cells,
        )

    return table
