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

    HEADER_PANEL_SIZE = 4
    STATUS_PANEL_SIZE = 5
    SAMPLE_NUM_WIDTH = 3
    AGENT_ID_WIDTH = 14
    MODEL_WIDTH = 14
    STATUS_WIDTH = 11
    TIME_WIDTH = 6

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

        details_layout = Layout(name="details")
        details_layout.split_column(
            Layout(self._create_active_view(runtime_state, limit=active_row_limit), name="active", ratio=1),
            Layout(self._create_completed_view(runtime_state, limit=completed_row_limit), name="completed", ratio=1),
        )

        layout.split_column(
            Layout(self._create_header_panel(), name="header", size=self.HEADER_PANEL_SIZE),
            Layout(
                self._create_progress_with_metrics(runtime_state, main_progress),
                name="status",
                size=self.STATUS_PANEL_SIZE,
            ),
            details_layout,
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
        header_title = Text(
            f"🧪 Evaluation: {self.suite_name}",
            style="bold white",
            no_wrap=True,
            overflow="ellipsis",
        )
        try:
            header_title.apply_gradient("#00D1FF", "#7C3AED")
        except Exception:
            pass

        subtitle = Text(no_wrap=True, overflow="ellipsis")
        subtitle.append(f"Target: {self.target_kind}  •  ", style="dim")
        subtitle.append(f"Grader: {self.grader_kind}  •  ", style="dim")
        if self.rubric_model:
            subtitle.append("Rubric: ", style="dim")
            subtitle.append(self.rubric_model, style="magenta")
            subtitle.append("  •  ", style="dim")
        subtitle.append(f"Concurrent: {self.max_concurrent}", style="dim")

        rows = [Align.center(header_title), Align.center(subtitle)]
        if self.metric_labels:
            metrics_line = Text("Metrics: ", style="dim", no_wrap=True, overflow="ellipsis")
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
            errors_text = "Errors: N/A"
        else:
            errors_pct = (runtime_state.error_count / completed * 100.0) if completed > 0 else 0.0
            errors_text = f"Errors: {errors_pct:.1f}%"

        chips = Text()
        chips.append(f"  {errors_text}", style="bold white")

        chips.append("   ")
        chips.append(f"Target cost: ${runtime_state.total_target_cost:.4f}", style="bold white")

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
        del runtime_state
        available_lines = max(12, self.console.height - (self.HEADER_PANEL_SIZE + self.STATUS_PANEL_SIZE + 1))
        panel_size = max(6, available_lines // 2)
        return panel_size, panel_size

    def _panel_row_limit(self, panel_size: int) -> int:
        return max(1, panel_size - 5)

    def _add_fixed_width_column(
        self,
        table: Table,
        header: str,
        *,
        width: int,
        style: str,
        justify: str = "left",
        ratio: Optional[int] = None,
        allow_expand: bool = False,
    ) -> None:
        max_width = None if allow_expand else width
        table.add_column(
            header,
            style=style,
            justify=justify,
            width=width,
            min_width=width,
            max_width=max_width,
            ratio=ratio,
            no_wrap=True,
            overflow="ellipsis",
        )

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

        self._add_fixed_width_column(table, "#", width=self.SAMPLE_NUM_WIDTH, style="cyan")
        self._add_fixed_width_column(
            table,
            "Agent ID",
            width=self.AGENT_ID_WIDTH,
            style="dim cyan",
            ratio=2,
            allow_expand=True,
        )
        self._add_fixed_width_column(
            table,
            "Model",
            width=self.MODEL_WIDTH,
            style="yellow",
            ratio=2,
            allow_expand=True,
        )
        self._add_fixed_width_column(table, "Status", width=self.STATUS_WIDTH, style="white")

        metric_keys = list(self.metric_labels.keys())
        metric_count = len(metric_keys) if metric_keys else 1
        if metric_keys:
            for metric_key in metric_keys:
                label = self.metric_labels.get(metric_key, metric_key)
                score_width = max(8, min(10, len(label) + 2))
                rationale_width = max(
                    16 if metric_count > 1 else 18, min(24 if metric_count == 1 else 18, len(label) + 12)
                )
                self._add_fixed_width_column(
                    table,
                    f"{label} Score",
                    width=score_width,
                    style="white",
                    justify="right",
                )
                self._add_fixed_width_column(
                    table,
                    f"{label} Rationale",
                    width=rationale_width,
                    style="dim",
                    ratio=3,
                    allow_expand=True,
                )
        else:
            self._add_fixed_width_column(table, "Score", width=8, style="white", justify="right")
            self._add_fixed_width_column(table, "Rationale", width=18, style="dim", ratio=3, allow_expand=True)
        self._add_fixed_width_column(table, "Time", width=self.TIME_WIDTH, style="white", justify="right")
        self._add_fixed_width_column(
            table,
            "Details",
            width=14 if metric_count == 1 else 10,
            style="white",
            ratio=2,
            allow_expand=True,
        )

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
                        f"{score_value:.2f}"
                        if isinstance(score_value, (int, float)) and score_value is not None
                        else "-"
                    )
                    cells.extend([score_cell, rationale])
            else:
                score_cell = f"{sample.score:.2f}" if sample.score is not None else "-"
                rationale = sample.rationale or ""
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
                details = Text("Loading from cache..." if sample.from_cache else "Loading agent...", style="white")
            elif sample.state == SampleState.GRADING:
                details = Text("Grading response...", style="white")
            elif sample.state == SampleState.COMPLETED:
                details = Text("✓ Completed", style="green")
            elif sample.state == SampleState.ERROR:
                details = Text(f"Error: {sample.error}" if sample.error else "Error", style="red")
            elif sample.state == SampleState.QUEUED:
                details = Text("Waiting...", style="dim")
            else:
                details = ""

            sample_num = str(sample.sample_id)
            if sample.from_cache:
                sample_num = f"{sample_num} ♻"

            row_data = [
                sample_num,
                sample.agent_id or "-",
                sample.model_name or "-",
            ]
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
