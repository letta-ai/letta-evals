from __future__ import annotations

import time
from typing import Dict, List, Optional

from rich.align import Align
from rich.box import MINIMAL_DOUBLE_HEAD, ROUNDED
from rich.console import Console, Group, RenderableType
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

    ACTIVE_STATE_ORDER = {
        SampleState.LOADING_AGENT: 0,
        SampleState.SENDING_MESSAGES: 1,
        SampleState.GRADING: 2,
        SampleState.GRADING_TURNS: 3,
    }

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
        self._detail_sizes: Optional[tuple[int, int]] = None

    def render(self, runtime_state: ProgressRuntimeState, main_progress: Progress) -> Layout:
        """Render the complete live progress display."""
        layout = Layout()
        active_panel_size, completed_panel_size = self._detail_layout_budget()
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

    def build_visible_snapshot(self, runtime_state: ProgressRuntimeState) -> tuple:
        """Return a stable snapshot of visible content for refresh diffing."""
        active_panel_size, completed_panel_size = self._detail_layout_budget()
        active_row_limit = self._panel_row_limit(active_panel_size)
        completed_row_limit = self._panel_row_limit(completed_panel_size)
        metric_keys = self._metric_keys(runtime_state)

        visible_metric_totals = []
        for key in metric_keys:
            if key not in runtime_state.metric_totals:
                continue
            visible_metric_totals.append(
                (
                    key,
                    round(runtime_state.metric_totals[key], 6),
                    runtime_state.metric_counts.get(key, 0),
                )
            )

        return (
            active_panel_size,
            completed_panel_size,
            runtime_state.completed_count,
            runtime_state.error_count,
            tuple(visible_metric_totals),
            tuple(
                self._row_snapshot(sample, metric_keys)
                for sample in self.select_active_rows(runtime_state, limit=active_row_limit)
            ),
            tuple(
                self._row_snapshot(sample, metric_keys)
                for sample in self.select_completed_rows(runtime_state, limit=completed_row_limit)
            ),
        )

    def select_active_rows(
        self, runtime_state: ProgressRuntimeState, limit: Optional[int] = None
    ) -> List[SampleProgress]:
        """Select active rows with stable ordering inside each phase bucket."""
        rows = [sample for sample in runtime_state.samples.values() if is_active_state(sample.state)]
        rows.sort(key=self._active_sort_key)
        if limit is None:
            return rows
        return rows[:limit]

    def select_completed_rows(
        self, runtime_state: ProgressRuntimeState, limit: Optional[int] = None
    ) -> List[SampleProgress]:
        """Select completed rows in completion order instead of hot update order."""
        rows = [sample for sample in runtime_state.samples.values() if is_completed_state(sample.state)]
        rows.sort(key=self._completed_sort_key)
        if limit is None:
            return rows
        return rows[:limit]

    def _active_sort_key(self, sample: SampleProgress) -> tuple:
        state_rank = self.ACTIVE_STATE_ORDER.get(sample.state, len(self.ACTIVE_STATE_ORDER))
        sequence = sample.active_sort_sequence or 10**9
        return (state_rank, sequence, sample.model_name or "", sample.sample_id)

    def _completed_sort_key(self, sample: SampleProgress) -> tuple:
        completion_rank = sample.completion_sequence if sample.completion_sequence is not None else 0
        fallback = get_last_update_key(sample) if sample.completion_sequence is None else 0.0
        return (-completion_rank, -fallback, sample.model_name or "", sample.sample_id)

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
        return Text(icon, style=style, end="")

    def _get_state_text(self, sample: SampleProgress) -> Text:
        icon = self._get_state_icon(sample.state)

        if sample.state == SampleState.SENDING_MESSAGES and sample.total_messages > 0:
            text = Text(overflow="ellipsis", no_wrap=True, end="")
            text.append(icon)
            text.append(f" sending [{sample.messages_sent}/{sample.total_messages}]")
            return text

        if sample.state == SampleState.GRADING_TURNS and sample.total_turns > 0:
            text = Text(overflow="ellipsis", no_wrap=True, end="")
            text.append(icon)
            text.append(f" grading [{sample.turns_graded}/{sample.total_turns}]")
            return text

        if sample.state == SampleState.COMPLETED:
            text = Text(overflow="ellipsis", no_wrap=True, end="")
            text.append(Text("✓", style="green", end=""))
            text.append(" completed")
            return text

        text = Text(overflow="ellipsis", no_wrap=True, end="")
        text.append(icon)
        text.append(f" {sample.state.value}")
        return text

    def _create_header_panel(self) -> Panel:
        header_title = Text(f"🧪 Evaluation: {self.suite_name}", style="bold white", end="")
        try:
            header_title.apply_gradient("#00D1FF", "#7C3AED")
        except Exception:
            pass

        subtitle = Text(end="")
        subtitle.append(f"Target: {self.target_kind}  •  ", style="dim")
        subtitle.append(f"Grader: {self.grader_kind}  •  ", style="dim")
        subtitle.append(f"Concurrent: {self.max_concurrent}", style="dim")

        rows: List[RenderableType] = [Align.center(header_title), Align.center(subtitle)]
        if self.metric_labels:
            metrics_line = Text("Metrics: ", style="dim", end="")
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

        chips = Text(end="")
        chips.append(f"  {errors_text}", style="bold white")

        if runtime_state.metric_totals:
            chips.append("   ")
            first = True
            for key in self._metric_keys(runtime_state):
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

        return Panel(Group(main_progress, Text("", end=""), chips), box=ROUNDED, border_style="blue", padding=(0, 1))

    def _detail_layout_budget(self) -> tuple[int, int]:
        if self._detail_sizes is None:
            available_lines = max(12, self.console.height - 10)
            completed_panel_size = min(11, max(6, available_lines // 3))
            active_panel_size = max(7, available_lines - completed_panel_size)
            self._detail_sizes = (active_panel_size, completed_panel_size)
        return self._detail_sizes

    def _panel_row_limit(self, panel_size: int) -> int:
        return max(1, panel_size - 5)

    def _create_samples_table(
        self, rows: List[SampleProgress], title: str, row_limit: int, empty_message: Optional[str] = None
    ) -> Table:
        metric_keys = self._metric_keys_from_rows(rows)
        table = Table(
            title=title,
            show_header=True,
            header_style="bold cyan",
            border_style="blue",
            box=ROUNDED,
            expand=True,
        )

        table.add_column("#", style="cyan", width=5, no_wrap=True)
        table.add_column("Agent ID", style="dim cyan", max_width=24, no_wrap=True, overflow="ellipsis")
        table.add_column("Model", style="yellow", max_width=27, no_wrap=True, overflow="ellipsis")
        if self.grader_kind == GraderKind.MODEL_JUDGE.value and self.rubric_model:
            table.add_column("Rubric Model", style="magenta", max_width=27, no_wrap=True, overflow="ellipsis")
        table.add_column("Status", max_width=20, no_wrap=True, overflow="ellipsis")

        if metric_keys:
            for metric_key in metric_keys:
                label = self.metric_labels.get(metric_key, metric_key)
                table.add_column(f"{label} Score", width=10, justify="right", no_wrap=True)
                table.add_column(
                    f"{label} Rationale",
                    max_width=45,
                    no_wrap=True,
                    overflow="ellipsis",
                )
        else:
            table.add_column("Score", width=10, justify="right", no_wrap=True)
            table.add_column("Rationale", max_width=45, no_wrap=True, overflow="ellipsis")
        table.add_column("Time", width=8, justify="right", no_wrap=True)
        table.add_column("Details", ratio=1, min_width=16, no_wrap=True, overflow="ellipsis")

        displayed_rows = rows[:row_limit]
        for sample in displayed_rows:
            table.add_row(*self._build_row_cells(sample, metric_keys))

        placeholder_message = empty_message if not displayed_rows else None
        for _ in range(row_limit - len(displayed_rows)):
            table.add_row(*self._blank_row_cells(metric_keys, placeholder_message))
            placeholder_message = None

        return table

    def _metric_keys(self, runtime_state: ProgressRuntimeState) -> List[str]:
        if self.metric_labels:
            return list(self.metric_labels.keys())
        return list(runtime_state.metric_totals.keys())

    def _metric_keys_from_rows(self, rows: List[SampleProgress]) -> List[str]:
        del rows
        if self.metric_labels:
            return list(self.metric_labels.keys())
        return []

    def _build_row_cells(self, sample: SampleProgress, metric_keys: List[str]) -> List[RenderableType]:
        score_cells = self._get_score_cells(sample, metric_keys)
        sample_num = str(sample.sample_id + 1)
        if sample.from_cache:
            sample_num = f"{sample_num} ♻"

        row_data: List[RenderableType] = [
            self._text_cell(sample_num),
            self._text_cell(sample.agent_id or "-"),
            self._text_cell(sample.model_name or "-"),
        ]
        if self.grader_kind == GraderKind.MODEL_JUDGE.value and self.rubric_model:
            row_data.append(self._text_cell(self.rubric_model))
        row_data.extend(
            [
                self._get_state_text(sample),
                *score_cells,
                self._text_cell(self._get_time_text(sample)),
                self._get_details_text(sample),
            ]
        )
        return row_data

    def _blank_row_cells(self, metric_keys: List[str], message: Optional[str] = None) -> List[RenderableType]:
        row_data: List[RenderableType] = [
            self._text_cell(""),
            self._text_cell(""),
            self._text_cell(""),
        ]
        if self.grader_kind == GraderKind.MODEL_JUDGE.value and self.rubric_model:
            row_data.append(self._text_cell(""))

        row_data.append(self._text_cell(""))
        if metric_keys:
            for _ in metric_keys:
                row_data.extend([self._text_cell(""), self._text_cell("")])
        else:
            row_data.extend([self._text_cell(""), self._text_cell("")])

        row_data.append(self._text_cell(""))
        row_data.append(self._text_cell(message or "", style="dim"))
        return row_data

    def _get_score_cells(self, sample: SampleProgress, metric_keys: List[str]) -> List[RenderableType]:
        cells: List[RenderableType] = []

        if sample.state == SampleState.GRADING_TURNS and sample.turn_scores:
            if metric_keys:
                for metric_key in metric_keys:
                    grader_scores = sample.turn_scores.get(metric_key)
                    if grader_scores:
                        score_cell = f"{calculate_turn_average(grader_scores):.2f}"
                        rationale = build_turn_symbols(grader_scores)
                    else:
                        score_cell = "-"
                        rationale = ""
                    cells.extend([self._text_cell(score_cell), self._text_cell(rationale)])
            else:
                first_grader = next(iter(sample.turn_scores.values()), None)
                if first_grader:
                    score_cell = f"{calculate_turn_average(first_grader):.2f}"
                    rationale = build_turn_symbols(first_grader)
                else:
                    score_cell = "-"
                    rationale = ""
                cells.extend([self._text_cell(score_cell), self._text_cell(rationale)])
            return cells

        if metric_keys:
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
                cells.extend([self._text_cell(score_cell), self._text_cell(rationale)])
            return cells

        score_cell = f"{sample.score:.2f}" if sample.score is not None else "-"
        rationale = sample.rationale or ""
        return [self._text_cell(score_cell), self._text_cell(rationale)]

    def _get_time_text(self, sample: SampleProgress) -> str:
        if sample.start_time and sample.end_time:
            duration = sample.end_time - sample.start_time
            return f"{duration:.1f}s"
        if sample.start_time:
            duration = time.time() - sample.start_time
            return f"{duration:.1f}s"
        return "-"

    def _get_details_text(self, sample: SampleProgress) -> Text:
        if sample.state == SampleState.SENDING_MESSAGES and sample.total_messages > 0:
            progress = sample.messages_sent / sample.total_messages
            bar_width = max(10, min(30, max(10, self.console.width // 6)))
            filled = int(progress * bar_width)
            bar = "▰" * filled + "▱" * (bar_width - filled)
            details = f"{bar}  msg {sample.messages_sent}/{sample.total_messages}"
            return self._text_cell(details)

        if sample.state == SampleState.GRADING_TURNS and sample.total_turns > 0:
            progress = sample.turns_graded / sample.total_turns
            bar_width = max(10, min(30, max(10, self.console.width // 6)))
            filled = int(progress * bar_width)
            bar = "▰" * filled + "▱" * (bar_width - filled)
            details = f"{bar}  turn {sample.turns_graded}/{sample.total_turns}"
            return self._text_cell(details)

        if sample.state == SampleState.LOADING_AGENT:
            return self._text_cell("Loading from cache…" if sample.from_cache else "Loading agent…")
        if sample.state == SampleState.GRADING:
            return self._text_cell("Grading response…")
        if sample.state == SampleState.COMPLETED:
            return self._text_cell("✓ Completed", style="green")
        if sample.state == SampleState.ERROR:
            detail = f"Error: {sample.error}" if sample.error else "Error"
            return self._text_cell(detail, style="red")
        if sample.state == SampleState.QUEUED:
            return self._text_cell("Waiting…", style="dim")
        return self._text_cell("")

    def _text_cell(self, value: str, style: str = "") -> Text:
        return Text(value, style=style, overflow="ellipsis", no_wrap=True, end="")

    def _row_snapshot(self, sample: SampleProgress, metric_keys: List[str]) -> tuple:
        score_snapshot = []
        for cell in self._get_score_cells(sample, metric_keys):
            if isinstance(cell, Text):
                score_snapshot.append(cell.plain)
            else:
                score_snapshot.append(str(cell))

        return (
            sample.sample_id,
            sample.from_cache,
            sample.agent_id or "-",
            sample.model_name or "-",
            self.rubric_model if self.grader_kind == GraderKind.MODEL_JUDGE.value and self.rubric_model else None,
            sample.state.value,
            sample.messages_sent,
            sample.total_messages,
            sample.turns_graded,
            sample.total_turns,
            tuple(score_snapshot),
            self._get_time_snapshot(sample),
            self._get_details_text(sample).plain,
        )

    def _get_time_snapshot(self, sample: SampleProgress) -> Optional[float]:
        if sample.start_time and sample.end_time:
            return round(sample.end_time - sample.start_time, 3)
        return round(sample.start_time, 3) if sample.start_time else None

    def _create_active_view(self, runtime_state: ProgressRuntimeState, limit: int) -> Table:
        rows = self.select_active_rows(runtime_state, limit=limit)
        return self._create_samples_table(rows, "Active Samples", limit, empty_message="No active samples")

    def _create_completed_view(self, runtime_state: ProgressRuntimeState, limit: int) -> Table:
        rows = self.select_completed_rows(runtime_state, limit=limit)
        return self._create_samples_table(rows, "Recent Completed", limit, empty_message="No completed samples yet")
