from __future__ import annotations

import time
from typing import Dict, List, Optional

from rich.align import Align
from rich.box import MINIMAL_DOUBLE_HEAD, ROUNDED
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table
from rich.text import Text

from letta_evals.types import GraderKind
from letta_evals.utils import build_turn_symbols, calculate_turn_average
from letta_evals.visualization.reducer import ProgressRuntimeState
from letta_evals.visualization.state import (
    SampleProgress,
    SampleState,
    get_last_update_key,
    is_active_state,
    is_completed_state,
)


class RichProgressRenderer:
    """Render the live Rich layout for evaluation progress."""

    def __init__(
        self,
        *,
        console: Console,
        suite_name: str,
        target_kind: str,
        grader_kind: str,
        rubric_model: Optional[str],
        max_concurrent: int,
        metric_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        self.console = console
        self.suite_name = suite_name
        self.target_kind = target_kind
        self.grader_kind = grader_kind
        self.rubric_model = rubric_model
        self.max_concurrent = max_concurrent
        self.metric_labels = metric_labels or {}

    def render(self, runtime_state: ProgressRuntimeState, main_progress: Progress) -> Layout:
        """Render the complete live progress display."""
        layout = Layout()
        active_panel_size, completed_panel_size = self._detail_layout_budget(runtime_state)
        active_row_limit = self._panel_row_limit(active_panel_size)
        completed_row_limit = self._panel_row_limit(completed_panel_size)

        details_layout = Layout()
        details_layout.split_column(
            Layout(self._create_active_view(runtime_state, limit=active_row_limit), size=active_panel_size),
            Layout(self._create_completed_view(runtime_state, limit=completed_row_limit), size=completed_panel_size),
        )

        layout.split_column(
            Layout(self._create_header_panel(), size=4),
            Layout(self._create_progress_with_metrics(runtime_state, main_progress), size=5),
            Layout(details_layout),
        )

        return layout

    def select_active_rows(
        self, runtime_state: ProgressRuntimeState, limit: Optional[int] = None
    ) -> List[SampleProgress]:
        """Select only currently active rows for the live top panel."""
        rows = [sample for sample in runtime_state.samples.values() if is_active_state(sample.state)]
        rows.sort(key=lambda sample: (sample.model_name or "", sample.sample_id))
        if limit is None:
            return rows
        return rows[:limit]

    def select_completed_rows(
        self, runtime_state: ProgressRuntimeState, limit: Optional[int] = None
    ) -> List[SampleProgress]:
        """Select the most recent completed or errored rows for the bottom panel."""
        rows = [sample for sample in runtime_state.samples.values() if is_completed_state(sample.state)]
        rows.sort(key=lambda sample: (-get_last_update_key(sample), sample.model_name or "", sample.sample_id))
        if limit is None:
            return rows
        return rows[:limit]

    def _get_state_icon(self, state: SampleState) -> Text:
        icons = {
            SampleState.QUEUED: ("⋯", "dim"),
            SampleState.LOADING_AGENT: ("⊙", "yellow"),
            SampleState.SENDING_MESSAGES: ("⚡", "cyan"),
            SampleState.GRADING: ("🔍", "magenta"),
            SampleState.GRADING_TURNS: ("🔍", "magenta"),
            SampleState.COMPLETED: ("✓", "green"),
            SampleState.FAILED: ("✗", "red"),
            SampleState.ERROR: ("⚠", "red"),
        }
        icon, style = icons.get(state, ("?", "white"))
        return Text(icon, style=style)

    def _get_state_text(self, sample: SampleProgress) -> Text:
        icon = self._get_state_icon(sample.state)

        if sample.state == SampleState.SENDING_MESSAGES and sample.total_messages > 0:
            text = Text()
            text.append(icon)
            text.append(f" sending [{sample.messages_sent}/{sample.total_messages}]")
            return text

        if sample.state == SampleState.GRADING_TURNS and sample.total_turns > 0:
            text = Text()
            text.append(icon)
            text.append(f" grading [{sample.turns_graded}/{sample.total_turns}]")
            return text

        if sample.state == SampleState.COMPLETED:
            text = Text()
            text.append(Text("✓", style="green"))
            text.append(" completed")
            return text

        text = Text()
        text.append(icon)
        text.append(f" {sample.state.value}")
        return text

    def _create_header_panel(self) -> Panel:
        header_title = Text(f"🧪 Evaluation: {self.suite_name}", style="bold white")
        try:
            header_title.apply_gradient("#00D1FF", "#7C3AED")
        except Exception:
            pass

        subtitle = Text()
        subtitle.append(f"Target: {self.target_kind}  •  ", style="dim")
        subtitle.append(f"Grader: {self.grader_kind}  •  ", style="dim")
        subtitle.append(f"Concurrent: {self.max_concurrent}", style="dim")

        rows: List[Text] = [Align.center(header_title), Align.center(subtitle)]
        if self.metric_labels:
            metrics_line = Text("Metrics: ", style="dim")
            metrics_line.append(", ".join(self.metric_labels.values()), style="white")
            rows.append(Align.center(metrics_line))

        return Panel(
            Group(*rows),
            border_style="cyan",
            box=MINIMAL_DOUBLE_HEAD,
            padding=(0, 1),
        )

    def _create_progress_with_metrics(self, runtime_state: ProgressRuntimeState, main_progress: Progress) -> Panel:
        completed = runtime_state.completed_count + runtime_state.error_count

        if completed == 0:
            errors_text = "Errored: N/A"
        else:
            errors_pct = (runtime_state.error_count / completed * 100.0) if completed > 0 else 0.0
            errors_text = f"Errored: {errors_pct:.1f}%"

        chips = Text()
        chips.append(f"  {errors_text}", style="bold white")

        if runtime_state.metric_totals:
            chips.append("   ")
            first = True
            keys = list(self.metric_labels.keys()) if self.metric_labels else list(runtime_state.metric_totals.keys())
            for key in keys:
                if key not in runtime_state.metric_totals:
                    continue
                total = runtime_state.metric_totals[key]
                count = runtime_state.metric_counts.get(key, 0)
                if count == 0:
                    continue
                avg = total / count
                label = self.metric_labels.get(key, key)
                if not first:
                    chips.append("   ")
                chips.append(f"{label}: {avg:.2f}", style="bold white")
                first = False

        chips.append("   ")
        chips.append(f"✓ {runtime_state.completed_count}", style="green")
        if runtime_state.error_count:
            chips.append("   ")
            chips.append(f"⚠ {runtime_state.error_count}", style="yellow")

        return Panel(Group(main_progress, Text(""), chips), box=ROUNDED, border_style="blue", padding=(0, 1))

    def _detail_layout_budget(self, runtime_state: ProgressRuntimeState) -> tuple[int, int]:
        available_lines = max(12, self.console.height - 10)
        has_completed = any(is_completed_state(sample.state) for sample in runtime_state.samples.values())

        if has_completed:
            completed_panel_size = min(11, max(6, available_lines // 3))
        else:
            completed_panel_size = 5

        active_panel_size = max(7, available_lines - completed_panel_size)
        return active_panel_size, completed_panel_size

    def _panel_row_limit(self, panel_size: int) -> int:
        return max(1, panel_size - 5)

    def _create_samples_table(self, rows: List[SampleProgress], title: str, empty_message: str):
        if not rows:
            return Panel(
                Text(empty_message, style="dim"),
                title=title,
                border_style="blue",
                box=ROUNDED,
                padding=(0, 1),
            )

        table = Table(
            title=f"{title}  (♻ means cached)",
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            box=ROUNDED,
            expand=True,
        )

        table.add_column("#", style="cyan", width=5)
        table.add_column("Agent ID", style="dim cyan", no_wrap=False)
        table.add_column("Model", style="yellow", width=27)
        if self.grader_kind == GraderKind.MODEL_JUDGE.value and self.rubric_model:
            table.add_column("Rubric Model", style="magenta", width=27)
        table.add_column("Status", width=20)

        metric_keys = list(self.metric_labels.keys())
        if metric_keys:
            for metric_key in metric_keys:
                label = self.metric_labels.get(metric_key, metric_key)
                table.add_column(f"{label} Score", width=10, justify="right")
                table.add_column(f"{label} Rationale", width=45, justify="left")
        else:
            table.add_column("Score", width=10, justify="right")
            table.add_column("Rationale", width=45, justify="left")
        table.add_column("Time", width=8, justify="right")
        table.add_column("Details", justify="left")

        for sample in rows:
            if sample.start_time and sample.end_time:
                duration = sample.end_time - sample.start_time
                time_text = f"{duration:.1f}s"
            elif sample.start_time:
                duration = time.time() - sample.start_time
                time_text = f"{duration:.1f}s"
            else:
                time_text = "-"

            cells: List[str] = []
            if sample.state == SampleState.GRADING_TURNS and sample.turn_scores:
                if self.metric_labels:
                    for metric_key in metric_keys:
                        grader_scores = sample.turn_scores.get(metric_key)
                        if grader_scores:
                            score_cell = f"{calculate_turn_average(grader_scores):.2f}"
                            rationale = build_turn_symbols(grader_scores)
                        else:
                            score_cell = "-"
                            rationale = ""
                        cells.extend([score_cell, rationale])
                else:
                    first_grader = next(iter(sample.turn_scores.values()), None)
                    if first_grader:
                        score_cell = f"{calculate_turn_average(first_grader):.2f}"
                        rationale = build_turn_symbols(first_grader)
                    else:
                        score_cell = "-"
                        rationale = ""
                    cells.extend([score_cell, rationale])
            elif self.metric_labels:
                for metric_key in metric_keys:
                    score_value = None
                    rationale = ""
                    if sample.metric_scores and metric_key in sample.metric_scores:
                        score_value = sample.metric_scores.get(metric_key)
                    if sample.metric_rationales and metric_key in sample.metric_rationales:
                        rationale = sample.metric_rationales.get(metric_key) or ""
                    score_cell = (
                        f"{score_value:.2f}" if isinstance(score_value, (int, float)) and score_value is not None else "-"
                    )
                    if rationale and len(rationale) > 50:
                        rationale = rationale[:47] + "..."
                    cells.extend([score_cell, rationale])
            else:
                score_cell = f"{sample.score:.2f}" if sample.score is not None else "-"
                rationale = sample.rationale or ""
                if rationale and len(rationale) > 50:
                    rationale = rationale[:47] + "..."
                cells.extend([score_cell, rationale])

            if sample.state == SampleState.SENDING_MESSAGES and sample.total_messages > 0:
                progress = sample.messages_sent / sample.total_messages
                bar_width = max(10, min(30, max(10, self.console.width // 6)))
                filled = int(progress * bar_width)
                bar = "▰" * filled + "▱" * (bar_width - filled)
                details = f"{bar}  msg {sample.messages_sent}/{sample.total_messages}"
            elif sample.state == SampleState.GRADING_TURNS and sample.total_turns > 0:
                progress = sample.turns_graded / sample.total_turns
                bar_width = max(10, min(30, max(10, self.console.width // 6)))
                filled = int(progress * bar_width)
                bar = "▰" * filled + "▱" * (bar_width - filled)
                details = f"{bar}  turn {sample.turns_graded}/{sample.total_turns}"
            elif sample.state == SampleState.LOADING_AGENT:
                details = "Loading from cache…" if sample.from_cache else "Loading agent…"
            elif sample.state == SampleState.GRADING:
                details = "Grading response…"
            elif sample.state == SampleState.COMPLETED:
                details = "[green]✓ Completed[/green]"
            elif sample.state == SampleState.ERROR:
                details = f"[red]Error: {sample.error[:25]}…[/red]" if sample.error else "[red]Error[/red]"
            elif sample.state == SampleState.QUEUED:
                details = "[dim]Waiting…[/dim]"
            else:
                details = ""

            sample_num = str(sample.sample_id + 1)
            if sample.from_cache:
                sample_num = f"{sample_num} ♻"

            row_data = [
                sample_num,
                sample.agent_id or "-",
                sample.model_name or "-",
            ]
            if self.grader_kind == GraderKind.MODEL_JUDGE.value and self.rubric_model:
                row_data.append(self.rubric_model)
            row_data.extend([self._get_state_text(sample), *cells, time_text, details])
            table.add_row(*row_data)

        return table

    def _create_active_view(self, runtime_state: ProgressRuntimeState, limit: int):
        rows = self.select_active_rows(runtime_state, limit=limit)
        title = f"Active Samples · showing {len(rows)}"
        return self._create_samples_table(rows, title, "No active samples")

    def _create_completed_view(self, runtime_state: ProgressRuntimeState, limit: int):
        rows = self.select_completed_rows(runtime_state, limit=limit)
        title = f"Recent Completed · showing {len(rows)}"
        return self._create_samples_table(rows, title, "No completed samples yet")
