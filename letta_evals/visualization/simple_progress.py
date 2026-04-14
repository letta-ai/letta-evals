from __future__ import annotations

from typing import Dict, Optional

from rich.console import Console

from letta_evals.visualization.base import ProgressCallback
from letta_evals.visualization.summary import (
    build_simple_sample_results_table,
    format_gate_description,
    get_displayed_sample_results,
    print_basic_overall_metrics,
    print_remaining_samples_notice,
    print_truncated_samples_notice,
)


class SimpleProgress(ProgressCallback):
    """Clean hierarchical progress callback for CI and non-interactive terminals.

    Uses visual hierarchy with indentation and simple unicode symbols to make
    evaluation progress easy to scan in logs.
    """

    def __init__(self, suite_name: str, total_samples: int, console: Optional[Console] = None):
        self.suite_name = suite_name
        self.total_samples = total_samples
        self.console = console or Console()

    async def start(self) -> None:
        self.console.print("━" * 60)
        self.console.print(f"[bold cyan]Suite:[/] {self.suite_name}")
        self.console.print(f"[bold cyan]Samples:[/] {self.total_samples}")
        self.console.print("━" * 60)
        self.console.print()

    def stop(self) -> None:
        self.console.print()
        self.console.print("━" * 60)
        self.console.print("[bold cyan]Suite completed[/]")
        self.console.print("━" * 60)

    def reset(self) -> None:
        """Reset state for a new run."""

    async def sample_started(
        self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> None:
        model_text = f" [dim]({model_name})[/]" if model_name else ""
        agent_text = f" [dim]agent={agent_id}[/]" if agent_id else ""
        self.console.print(f"[bold cyan]▸ Sample [{sample_id}]{model_text}{agent_text}[/]")

    async def agent_created(
        self, sample_id: int, agent_id: str, model_name: Optional[str] = None, from_cache: bool = False
    ) -> None:
        prefix = self._format_prefix(sample_id, agent_id, model_name)
        cache_text = " [dim](cached)[/]" if from_cache else ""
        self.console.print(f"{prefix} [dim]•[/] Agent created{cache_text}")

    async def message_sending(
        self,
        sample_id: int,
        message_num: int,
        total_messages: int,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        prefix = self._format_prefix(sample_id, agent_id, model_name)
        self.console.print(f"{prefix} [dim]•[/] Sending messages {message_num}/{total_messages}")

    async def grading_started(
        self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None
    ) -> None:
        prefix = self._format_prefix(sample_id, agent_id, model_name)
        self.console.print(f"{prefix} [dim]•[/] Grading...")

    async def sample_completed(
        self,
        sample_id: int,
        agent_id: Optional[str] = None,
        score: Optional[float] = None,
        target_cost: Optional[float] = None,
        model_name: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
        rationale: Optional[str] = None,
        metric_rationales: Optional[Dict[str, str]] = None,
    ) -> None:
        prefix = self._format_prefix(sample_id, agent_id, model_name)
        status = "[bold cyan]✓ DONE[/]"
        parts = [f"{prefix} {status}"]

        if score is not None:
            parts.append(f"score={score:.2f}")

        if metric_scores:
            metric_bits = ", ".join(f"{k}={v:.2f}" for k, v in metric_scores.items())
            parts.append(metric_bits)

        self.console.print("  ".join(parts))

    async def sample_error(
        self,
        sample_id: int,
        error: str,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
        target_cost: Optional[float] = None,
    ) -> None:
        prefix = self._format_prefix(sample_id, agent_id, model_name)
        self.console.print(f"{prefix} [bold yellow]⚠ ERROR[/]: {error}")

    async def suite_completed(self, result):
        """Display summary results after evaluation completes"""
        self.console.print()
        self.console.print("[bold]Evaluation Results:[/bold]")
        self.console.print("=" * 50)

        print_basic_overall_metrics(self.console, result.metrics)

        # gate status
        status = "[green]PASSED[/green]" if result.gates_passed else "[red]FAILED[/red]"
        gate_desc = format_gate_description(result.config, fixed_decimal_value=True)

        self.console.print(f"\n[bold]Gate:[/bold] {gate_desc} → {status}")

        # sample results table
        self.console.print("\n[bold]Sample Results:[/bold]")
        total_samples, displayed_results = get_displayed_sample_results(result)
        print_truncated_samples_notice(self.console, total_samples, len(displayed_results))
        self.console.print(build_simple_sample_results_table(result.config, displayed_results))
        print_remaining_samples_notice(self.console, total_samples, len(displayed_results))

    def _format_prefix(self, sample_id: int, agent_id: Optional[str], model_name: Optional[str]) -> str:
        """format a compact prefix for substeps to show which sample they belong to."""
        parts = [f"[dim]\\[[/][cyan]{sample_id}[/][dim]][/]"]
        if model_name:
            parts.append(f"[dim]({model_name})[/]")
        if agent_id:
            parts.append(f"[dim]agent={agent_id}[/]")
        return "".join(parts)
