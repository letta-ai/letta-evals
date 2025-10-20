from __future__ import annotations

from typing import Dict, Optional

from rich.console import Console

from letta_evals.visualization.base import ProgressCallback


class SimpleProgress(ProgressCallback):
    """Minimal, line-oriented progress callback.

    Prints concise, readable updates without rich Live layout. Suitable for
    non-interactive terminals or logs.
    """

    def __init__(self, suite_name: str, total_samples: int, console: Optional[Console] = None):
        self.suite_name = suite_name
        self.total_samples = total_samples
        self.console = console or Console()

    async def start(self) -> None:
        self.console.print(f"[cyan]Starting suite:[/] {self.suite_name}  ([dim]{self.total_samples} samples[/dim])")

    def stop(self) -> None:
        self.console.print("[cyan]Suite completed[/]")

    async def sample_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        prefix = self._prefix(sample_id, model_name)
        self.console.print(f"{prefix} [bold]started[/]")

    async def agent_loading(self, sample_id: int, model_name: Optional[str] = None, from_cache: bool = False) -> None:
        prefix = self._prefix(sample_id, model_name)
        cache_text = " (cached)" if from_cache else ""
        self.console.print(f"{prefix} loading agent{cache_text}")

    async def message_sending(
        self, sample_id: int, message_num: int, total_messages: int, model_name: Optional[str] = None
    ) -> None:
        prefix = self._prefix(sample_id, model_name)
        self.console.print(f"{prefix} sending messages {message_num}/{total_messages}")

    async def grading_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        prefix = self._prefix(sample_id, model_name)
        self.console.print(f"{prefix} grading...")

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
        prefix = self._prefix(sample_id, model_name)
        status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        parts = [f"{prefix} {status}"]
        if score is not None:
            parts.append(f"score={score:.2f}")
        if metric_scores:
            metric_bits = ", ".join(f"{k}={v:.2f}" for k, v in metric_scores.items())
            parts.append(metric_bits)
        self.console.print("  ".join(parts))

    async def sample_error(self, sample_id: int, error: str, model_name: Optional[str] = None) -> None:
        prefix = self._prefix(sample_id, model_name)
        self.console.print(f"{prefix} [yellow]ERROR[/]: {error}")

    def _prefix(self, sample_id: int, model_name: Optional[str]) -> str:
        if model_name:
            return f"[{sample_id}]({model_name})"
        return f"[{sample_id}]"
