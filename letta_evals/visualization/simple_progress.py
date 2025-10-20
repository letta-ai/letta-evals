from __future__ import annotations

from typing import Dict, Optional

from rich.console import Console

from letta_evals.visualization.base import ProgressCallback


class SimpleProgress(ProgressCallback):
    """Clean hierarchical progress callback for CI and non-interactive terminals.

    Uses visual hierarchy with indentation and simple unicode symbols to make
    evaluation progress easy to scan in logs.
    """

    def __init__(self, suite_name: str, total_samples: int, console: Optional[Console] = None):
        self.suite_name = suite_name
        self.total_samples = total_samples
        self.console = console or Console()
        self._current_sample = None

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

    async def sample_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        # track current sample to avoid printing header multiple times
        self._current_sample = (sample_id, model_name)
        model_text = f" [dim]({model_name})[/]" if model_name else ""
        self.console.print(f"[bold cyan]▸ Sample [{sample_id}]{model_text}[/]")

    async def agent_loading(self, sample_id: int, model_name: Optional[str] = None, from_cache: bool = False) -> None:
        prefix = self._format_prefix(sample_id, model_name)
        cache_text = " [dim](cached)[/]" if from_cache else ""
        self.console.print(f"{prefix} [dim]•[/] Loading agent{cache_text}")

    async def message_sending(
        self, sample_id: int, message_num: int, total_messages: int, model_name: Optional[str] = None
    ) -> None:
        prefix = self._format_prefix(sample_id, model_name)
        self.console.print(f"{prefix} [dim]•[/] Sending messages {message_num}/{total_messages}")

    async def grading_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        prefix = self._format_prefix(sample_id, model_name)
        self.console.print(f"{prefix} [dim]•[/] Grading...")

    async def sample_completed(
        self,
        sample_id: int,
        passed: bool,
        score: Optional[float] = None,
        model_name: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
        metric_pass: Optional[Dict[str, bool]] = None,
        rationale: Optional[str] = None,
        metric_rationales: Optional[Dict[str, str]] = None,
    ) -> None:
        prefix = self._format_prefix(sample_id, model_name)
        status = "[bold green]✓ PASS[/]" if passed else "[bold red]✗ FAIL[/]"
        parts = [f"{prefix} {status}"]

        if score is not None:
            parts.append(f"score={score:.2f}")

        if metric_scores:
            metric_bits = ", ".join(f"{k}={v:.2f}" for k, v in metric_scores.items())
            parts.append(metric_bits)

        self.console.print("  ".join(parts))

    async def sample_error(self, sample_id: int, error: str, model_name: Optional[str] = None) -> None:
        prefix = self._format_prefix(sample_id, model_name)
        self.console.print(f"{prefix} [bold yellow]⚠ ERROR[/]: {error}")

    def _format_prefix(self, sample_id: int, model_name: Optional[str]) -> str:
        """format a compact prefix for substeps to show which sample they belong to."""
        if model_name:
            return f"[dim]\\[[/][cyan]{sample_id}[/][dim]]({model_name})[/]"
        return f"[dim]\\[[/][cyan]{sample_id}[/][dim]][/]"
