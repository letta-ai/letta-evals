import time
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from typing import Dict, List, Optional

from rich.align import Align
from rich.box import MINIMAL_DOUBLE_HEAD, ROUNDED
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text

from letta_evals.types import GraderKind
from letta_evals.visualization.base import ProgressCallback


class SampleState(Enum):
    """States a sample can be in during evaluation"""

    QUEUED = "queued"
    LOADING_AGENT = "loading"
    SENDING_MESSAGES = "sending"
    GRADING = "grading"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class SampleProgress:
    """Track progress of individual sample"""

    sample_id: int
    state: SampleState = SampleState.QUEUED
    agent_id: Optional[str] = None
    model_name: Optional[str] = None
    passed: Optional[bool] = None
    score: Optional[float] = None
    rationale: Optional[str] = None
    error: Optional[str] = None
    messages_sent: int = 0
    total_messages: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_update_ts: Optional[float] = None
    from_cache: bool = False
    # Per-metric live data (for multi-grader runs)
    metric_scores: Optional[Dict[str, float]] = None
    metric_rationales: Optional[Dict[str, str]] = None


class DisplayMode(Enum):
    """Display modes for progress visualization"""

    COMPACT = "compact"
    STANDARD = "standard"
    DETAILED = "detailed"


class EvalProgress(ProgressCallback):
    """Beautiful progress visualization for evaluation runs"""

    def __init__(
        self,
        suite_name: str,
        total_samples: int,
        target_kind: str = "agent",
        grader_kind: str = "tool",
        rubric_model: Optional[str] = None,
        max_concurrent: int = 15,
        display_mode: DisplayMode = DisplayMode.STANDARD,
        console: Optional[Console] = None,
        update_freq: float = 10.0,
        show_samples: bool = True,
        cached_mode: bool = False,
        metric_labels: Optional[Dict[str, str]] = None,
    ):
        self.suite_name = suite_name
        self.total_samples = total_samples
        self.target_kind = target_kind
        self.grader_kind = grader_kind
        self.rubric_model = rubric_model
        self.max_concurrent = max_concurrent
        self.display_mode = display_mode
        self.show_samples = show_samples
        self.console = console or Console()
        self.update_freq = update_freq
        self.cached_mode = cached_mode
        self.metric_labels: Dict[str, str] = metric_labels or {}
        # live aggregates per metric key
        self.metric_totals: Dict[str, float] = {}
        self.metric_counts: Dict[str, int] = {}
        self.metric_passed: Dict[str, int] = {}
        self.metric_failed: Dict[str, int] = {}

        self.samples: Dict[tuple, SampleProgress] = {}  # key: (sample_id, model_name)
        self.start_time = None
        self.live: Optional[Live] = None
        self.main_progress = Progress(
            SpinnerColumn(style="bold cyan"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=None, complete_style="cyan", finished_style="green", pulse_style="magenta"),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self.main_task_id = None

        self.passed_count = 0
        self.failed_count = 0
        self.error_count = 0
        self.total_score = 0.0
        self.score_count = 0

    def _get_state_icon(self, state: SampleState) -> Text:
        """Get icon for sample state"""
        icons = {
            SampleState.QUEUED: ("⋯", "dim"),
            SampleState.LOADING_AGENT: ("⊙", "yellow"),
            SampleState.SENDING_MESSAGES: ("⚡", "cyan"),
            SampleState.GRADING: ("🔍", "magenta"),
            SampleState.COMPLETED: ("✓", "green"),
            SampleState.FAILED: ("✗", "red"),
            SampleState.ERROR: ("⚠", "red"),
        }
        icon, style = icons.get(state, ("?", "white"))
        return Text(icon, style=style)

    def _get_state_text(self, sample: SampleProgress) -> Text:
        """Get text representation of sample state"""
        icon = self._get_state_icon(sample.state)

        if sample.state == SampleState.SENDING_MESSAGES and sample.total_messages > 0:
            text = Text()
            text.append(icon)
            text.append(f" sending [{sample.messages_sent}/{sample.total_messages}]")
            return text
        elif sample.state == SampleState.COMPLETED:
            icon = Text("✓", style="green")
            text = Text()
            text.append(icon)
            text.append(" completed")
            return text
        else:
            text = Text()
            text.append(icon)
            text.append(f" {sample.state.value}")
            return text

    def _create_header_panel(self) -> Panel:
        """Create header panel with suite info"""
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

        content = Group(*rows)

        return Panel(
            content,
            border_style="cyan",
            box=MINIMAL_DOUBLE_HEAD,
            padding=(0, 1),
        )

    def _create_samples_grid(self) -> Panel:
        """Create grid showing sample states"""
        if not self.show_samples or self.display_mode == DisplayMode.COMPACT:
            return Panel("")

        # collect all samples by their base sample_id
        sample_by_id = {}
        for key, sample in self.samples.items():
            sample_id, _ = key
            if sample_id not in sample_by_id or sample.last_update_ts > (sample_by_id[sample_id].last_update_ts or 0):
                sample_by_id[sample_id] = sample

        rows = []
        samples_per_row = 15

        for i in range(0, self.total_samples, samples_per_row):
            row_text = Text(f"[{i + 1:3d}-{min(i + samples_per_row, self.total_samples):3d}] ", style="dim")

            for j in range(i, min(i + samples_per_row, self.total_samples)):
                sample = sample_by_id.get(j, SampleProgress(j))
                icon = self._get_state_icon(sample.state)
                if j > i:
                    row_text.append(" ")  # space between icons
                row_text.append(icon)

            rows.append(row_text)

        content = Group(*rows) if rows else Text("No samples", style="dim")

        return Panel(
            content,
            title="Samples",
            border_style="blue",
            box=ROUNDED,
            padding=(0, 1),
        )

    def _create_progress_with_metrics(self) -> Panel:
        """Create progress bar with inline metrics"""
        completed = self.passed_count + self.failed_count + self.error_count

        if completed == 0:
            errors_text = "Errored: N/A"
        else:
            errors_pct = (self.error_count / completed * 100.0) if completed > 0 else 0.0
            errors_text = f"Errored: {errors_pct:.1f}%"

        chips = Text()
        chips.append(f"  {errors_text}", style="bold white")
        # add per-metric aggregates if available
        if self.metric_totals:
            chips.append("   ")
            first = True
            keys = list(self.metric_labels.keys()) if self.metric_labels else list(self.metric_totals.keys())
            for key in keys:
                if key not in self.metric_totals:
                    continue
                total = self.metric_totals[key]
                cnt = self.metric_counts.get(key, 0)
                if cnt == 0:
                    continue
                avg = total / cnt if cnt > 0 else 0.0
                acc_cnt = self.metric_passed.get(key, 0) + self.metric_failed.get(key, 0)
                acc = (self.metric_passed.get(key, 0) / acc_cnt * 100.0) if acc_cnt > 0 else 0.0
                label = self.metric_labels.get(key, key)
                if not first:
                    chips.append("   ")
                chips.append(f"{label}: {avg:.2f}, {acc:.0f}%", style="bold white")
                first = False
        chips.append("   ")
        chips.append(f"✓ {self.passed_count}", style="green")
        chips.append("   ")
        chips.append(f"✗ {self.failed_count}", style="red")
        if self.error_count:
            chips.append("   ")
            chips.append(f"⚠ {self.error_count}", style="yellow")

        content = Group(self.main_progress, Text(""), chips)

        return Panel(content, box=ROUNDED, border_style="blue", padding=(0, 1))

    def _create_metrics_panel(self) -> Panel:
        """Create panel showing live metrics"""
        completed = self.passed_count + self.failed_count + self.error_count

        if completed == 0:
            errors_text_row = "N/A"
        else:
            errors_text_row = f"{(self.error_count / completed * 100.0):.1f}%"

        metrics_table = Table.grid(padding=1)
        metrics_table.add_column(style="cyan", justify="right")
        metrics_table.add_column(style="white")

        metrics_table.add_row("🛡️ Errored:", f"{errors_text_row} ({self.error_count}/{completed})")

        if self.score_count > 0:
            avg_score = self.total_score / self.score_count
            metrics_table.add_row("📈 Avg Score:", f"{avg_score:.2f}")

        if self.failed_count > 0:
            failed_samples = [str(key[0] + 1) for key, s in self.samples.items() if s.passed is False][:5]
            failed_text = ", ".join(failed_samples)
            if len(failed_samples) < self.failed_count:
                failed_text += f" ... ({self.failed_count} total)"
            metrics_table.add_row("❌ Failed:", failed_text)

        if self.error_count > 0:
            metrics_table.add_row("⚠️ Errors:", str(self.error_count))

        # Per-metric grid
        if self.metric_totals:
            metrics_table.add_row("", "")
            metrics_table.add_row("[bold]By Metric[/bold]", "")
            for key in self.metric_labels.keys() or self.metric_totals.keys():
                if key not in self.metric_totals:
                    continue
                total = self.metric_totals.get(key, 0.0)
                cnt = self.metric_counts.get(key, 0)
                avg = total / cnt if cnt > 0 else 0.0
                att = self.metric_passed.get(key, 0) + self.metric_failed.get(key, 0)
                acc = (self.metric_passed.get(key, 0) / att * 100.0) if att > 0 else 0.0
                label = self.metric_labels.get(key, key)
                metrics_table.add_row(f"• {label}:", f"avg={avg:.2f}, acc={acc:.1f}%")

        if self.start_time and completed > 0 and completed < self.total_samples:
            elapsed = time.time() - self.start_time
            rate = completed / elapsed
            remaining = self.total_samples - completed
            eta = remaining / rate if rate > 0 else 0
            eta_text = str(timedelta(seconds=int(eta)))
            metrics_table.add_row("⏱️ ETA:", eta_text)

        return Panel(
            metrics_table,
            title="Metrics",
            border_style="green",
            box=ROUNDED,
            padding=(0, 1),
        )

    def _create_detailed_view(self) -> Table:
        """Create a modern, height-aware table that prioritizes active and recent samples.

        Strategy:
        - Compute how many rows fit in the terminal, accounting for header/progress chrome.
        - Always show currently active samples (loading/sending/grading).
        - Fill remaining space with most-recently updated completed/failed/error samples.
        - If still space, rotate through queued items to give visibility without overflowing.
        """
        terminal_height = self.console.height

        available_lines = max(5, terminal_height - 10)
        # Account for table chrome (title, headers, borders)
        max_rows = max(1, available_lines - 5)
        n_rows = min(self.total_samples, max_rows)

        def last_update_key(s: SampleProgress) -> float:
            return s.last_update_ts or s.end_time or s.start_time or 0.0

        active_states = {SampleState.LOADING_AGENT, SampleState.SENDING_MESSAGES, SampleState.GRADING}
        completed_states = {SampleState.COMPLETED, SampleState.FAILED, SampleState.ERROR}

        # gather all samples
        samples_list = list(self.samples.values())
        active = [s for s in samples_list if s.state in active_states]
        active.sort(key=last_update_key, reverse=True)

        recent_done = [s for s in samples_list if s.state in completed_states]
        recent_done.sort(key=last_update_key, reverse=True)

        queued = [s for s in samples_list if s.state == SampleState.QUEUED]

        rows: List[SampleProgress] = []

        rows.extend(active[:n_rows])
        remaining = n_rows - len(rows)

        # 2) Show a rotating window of recently updated completed items
        if remaining > 0 and recent_done:
            rotation_period = 5
            page_size = remaining
            pages = (len(recent_done) + page_size - 1) // page_size
            page_idx = int(time.time() // rotation_period) % max(1, pages)
            start = page_idx * page_size
            rows.extend(recent_done[start : start + page_size])
            remaining = n_rows - len(rows)

        # 3) Fill any remaining with queued, also rotated
        if remaining > 0 and queued:
            page_size = remaining
            pages = (len(queued) + page_size - 1) // page_size
            page_idx = int(time.time() // 7) % max(1, pages)
            start = page_idx * page_size
            rows.extend(queued[start : start + page_size])

        showing = len(rows)
        title = (
            f"Active + recent · showing {showing} of {self.total_samples}"
            if showing < self.total_samples
            else f"All {self.total_samples} samples"
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
        # Add per-metric columns (score + rationale) or single score/rationale
        metric_keys = list(self.metric_labels.keys())
        if metric_keys:
            for mk in metric_keys:
                lbl = self.metric_labels.get(mk, mk)
                table.add_column(f"{lbl} Score", width=10, justify="right")
                table.add_column(f"{lbl} Rationale", width=45, justify="left")
        else:
            table.add_column("Score", width=10, justify="right")
            table.add_column("Rationale", width=45, justify="left")
        table.add_column("Time", width=8, justify="right")
        table.add_column("Details", justify="left")

        for s in rows:
            if s.start_time and s.end_time:
                duration = s.end_time - s.start_time
                time_text = f"{duration:.1f}s"
            elif s.start_time:
                duration = time.time() - s.start_time
                time_text = f"{duration:.1f}s"
            else:
                time_text = "-"

            # Build score/rationale cells
            cells: List[str] = []
            if self.metric_labels:
                for mk in metric_keys:
                    val = None
                    rat = ""
                    if s.metric_scores and mk in s.metric_scores:
                        val = s.metric_scores.get(mk)
                    if s.metric_rationales and mk in s.metric_rationales:
                        rat = s.metric_rationales.get(mk) or ""
                    score_cell = f"{val:.2f}" if isinstance(val, (int, float)) and val is not None else "-"
                    if rat and len(rat) > 50:
                        rat = rat[:47] + "..."
                    cells.extend([score_cell, rat])
            else:
                score_cell = f"{s.score:.2f}" if s.score is not None else "-"
                rat = s.rationale or ""
                if rat and len(rat) > 50:
                    rat = rat[:47] + "..."
                cells.extend([score_cell, rat])

            if s.state == SampleState.SENDING_MESSAGES and s.total_messages > 0:
                p = s.messages_sent / s.total_messages
                bar_width = max(10, min(30, max(10, self.console.width // 6)))
                filled = int(p * bar_width)
                bar = "▰" * filled + "▱" * (bar_width - filled)
                details = f"{bar}  msg {s.messages_sent}/{s.total_messages}"
            elif s.state == SampleState.LOADING_AGENT:
                details = "Loading from cache…" if s.from_cache else "Loading agent…"
            elif s.state == SampleState.GRADING:
                details = "Grading response…"
            elif s.state == SampleState.COMPLETED:
                details = "[green]✓ Completed[/green]"
            elif s.state == SampleState.ERROR:
                details = f"[red]Error: {s.error[:25]}…[/red]" if s.error else "[red]Error[/red]"
            elif s.state == SampleState.QUEUED:
                details = "[dim]Waiting…[/dim]"
            else:
                details = ""

            sample_num = str(s.sample_id + 1)
            if s.from_cache:
                sample_num = f"{sample_num} ♻"

            row_data = [
                sample_num,
                s.agent_id or "-",
                s.model_name or "-",
            ]
            if self.grader_kind == GraderKind.MODEL_JUDGE.value and self.rubric_model:
                row_data.append(self.rubric_model)
            row_data.extend([self._get_state_text(s), *cells, time_text, details])
            table.add_row(*row_data)

        return table

    def _render(self) -> Layout:
        """Render the complete progress display"""
        layout = Layout()

        layout.split_column(
            Layout(self._create_header_panel(), size=4),
            Layout(self._create_progress_with_metrics(), size=5),  # increased size to show both progress and metrics
            Layout(self._create_detailed_view()),
        )

        return layout

    def reset(self):
        """Reset counters and state for a new run"""
        self.passed_count = 0
        self.failed_count = 0
        self.error_count = 0
        self.total_score = 0.0
        self.score_count = 0
        self.samples.clear()
        self.metric_totals.clear()
        self.metric_counts.clear()
        self.metric_passed.clear()
        self.metric_failed.clear()
        if self.main_task_id is not None:
            self.main_progress.update(self.main_task_id, completed=0)

    async def start(self):
        """Start the progress display"""
        self.start_time = time.time()
        task_description = "Re-grading cached trajectories" if self.cached_mode else "Evaluating samples"
        self.main_task_id = self.main_progress.add_task(
            task_description,
            total=self.total_samples,
            completed=0,
        )

        # initialize samples placeholder - actual entries will be created as evaluations start
        # no longer pre-populate since we need model_name for the key

        self.live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=self.update_freq,
            transient=False,
            vertical_overflow="visible",
        )
        self.live.start()

    def stop(self):
        """Stop the progress display"""
        if self.live:
            self.live.stop()
            self.console.print()

    async def update_sample_state(
        self,
        sample_id: int,
        state: SampleState,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """Update state of a sample"""
        key = (sample_id, model_name)

        # If we have a model_name and there's an existing entry with None, migrate it
        if model_name is not None:
            old_key = (sample_id, None)
            if old_key in self.samples and key not in self.samples:
                # Migrate the old entry to the new key
                self.samples[key] = self.samples[old_key]
                self.samples[key].model_name = model_name
                del self.samples[old_key]

        if key not in self.samples:
            self.samples[key] = SampleProgress(sample_id, agent_id=agent_id, model_name=model_name)

        sample = self.samples[key]
        previous_state = sample.state
        sample.state = state

        if agent_id is not None and sample.agent_id != agent_id:
            sample.agent_id = agent_id

        if model_name is not None and sample.model_name != model_name:
            sample.model_name = model_name

        if state == SampleState.LOADING_AGENT and sample.start_time is None:
            sample.start_time = time.time()
        elif state in [SampleState.COMPLETED, SampleState.FAILED, SampleState.ERROR]:
            sample.end_time = time.time()

        for key, value in kwargs.items():
            if hasattr(sample, key):
                setattr(sample, key, value)
        sample.last_update_ts = time.time()

        terminal_states = {SampleState.COMPLETED, SampleState.FAILED, SampleState.ERROR}
        is_new_completion = previous_state not in terminal_states and state in terminal_states

        if state == SampleState.COMPLETED and is_new_completion:
            if sample.passed is True:
                self.passed_count += 1
            elif sample.passed is False:
                self.failed_count += 1

            if sample.score is not None:
                self.total_score += sample.score
                self.score_count += 1

            completed = self.passed_count + self.failed_count + self.error_count
            self.main_progress.update(self.main_task_id, completed=completed)

        elif state == SampleState.ERROR and is_new_completion:
            self.error_count += 1
            completed = self.passed_count + self.failed_count + self.error_count
            self.main_progress.update(self.main_task_id, completed=completed)

        if self.live:
            self.live.update(self._render())

    async def sample_started(self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None):
        """Mark sample as started"""
        key = (sample_id, model_name)
        if key not in self.samples:
            self.samples[key] = SampleProgress(sample_id, agent_id=agent_id, model_name=model_name)
        # skip loading state if using cached trajectories
        if not self.cached_mode:
            await self.update_sample_state(
                sample_id, SampleState.LOADING_AGENT, agent_id=agent_id, model_name=model_name
            )

    async def agent_loading(
        self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None, from_cache: bool = False
    ):
        """Mark sample as loading agent"""
        await self.update_sample_state(
            sample_id, SampleState.LOADING_AGENT, agent_id=agent_id, model_name=model_name, from_cache=from_cache
        )

    async def message_sending(
        self,
        sample_id: int,
        message_num: int,
        total_messages: int,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Update message sending progress"""
        await self.update_sample_state(
            sample_id,
            SampleState.SENDING_MESSAGES,
            agent_id=agent_id,
            model_name=model_name,
            messages_sent=message_num,
            total_messages=total_messages,
        )

    async def grading_started(self, sample_id: int, agent_id: Optional[str] = None, model_name: Optional[str] = None):
        """Mark sample as being graded"""
        key = (sample_id, model_name)
        # Check both the current key and the None key for from_cache flag
        existing_from_cache = False
        if key in self.samples:
            existing_from_cache = self.samples[key].from_cache
        elif model_name is not None and (sample_id, None) in self.samples:
            existing_from_cache = self.samples[(sample_id, None)].from_cache

        await self.update_sample_state(
            sample_id, SampleState.GRADING, agent_id=agent_id, model_name=model_name, from_cache=existing_from_cache
        )

    async def sample_completed(
        self,
        sample_id: int,
        agent_id: Optional[str] = None,
        score: Optional[float] = None,
        model_name: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
        metric_pass: Optional[Dict[str, bool]] = None,
        rationale: Optional[str] = None,
        metric_rationales: Optional[Dict[str, str]] = None,
    ):
        """Mark sample as completed"""
        # preserve from_cache flag if it was set
        key = (sample_id, model_name)
        existing_from_cache = False
        if key in self.samples:
            existing_from_cache = self.samples[key].from_cache
        elif model_name is not None and (sample_id, None) in self.samples:
            existing_from_cache = self.samples[(sample_id, None)].from_cache

        await self.update_sample_state(
            sample_id,
            SampleState.COMPLETED,
            agent_id=agent_id,
            model_name=model_name,
            passed=None,
            score=score,
            rationale=rationale,
            from_cache=existing_from_cache,
            metric_scores=metric_scores,
            metric_rationales=metric_rationales,
        )
        # update per-metric aggregates
        if metric_scores:
            for mkey, mscore in metric_scores.items():
                self.metric_totals[mkey] = self.metric_totals.get(mkey, 0.0) + (mscore or 0.0)
                self.metric_counts[mkey] = self.metric_counts.get(mkey, 0) + 1
                if metric_pass and mkey in metric_pass:
                    if metric_pass[mkey]:
                        self.metric_passed[mkey] = self.metric_passed.get(mkey, 0) + 1
                    else:
                        self.metric_failed[mkey] = self.metric_failed.get(mkey, 0) + 1

    async def sample_error(
        self, sample_id: int, error: str, agent_id: Optional[str] = None, model_name: Optional[str] = None
    ):
        """Mark sample as having an error"""
        await self.update_sample_state(
            sample_id,
            SampleState.ERROR,
            agent_id=agent_id,
            model_name=model_name,
            error=error,
        )

    async def suite_completed(self, result):
        """Display summary and detailed results after evaluation completes"""
        from letta_evals.constants import MAX_SAMPLES_DISPLAY

        self.console.print()
        self.console.print(f"[bold]Evaluation Results: {result.suite}[/bold]")
        if self.cached_mode:
            self.console.print("[dim]Note: Results re-graded from cached trajectories[/dim]")
        self.console.print("=" * 50)

        # overall metrics
        metrics = result.metrics
        self.console.print("\n[bold]Overall Metrics:[/bold]")
        self.console.print(f"  Total samples: {metrics.total}")
        self.console.print(f"  Total attempted: {metrics.total_attempted}")
        errors = metrics.total - metrics.total_attempted
        errors_pct = (errors / metrics.total * 100.0) if metrics.total > 0 else 0.0
        self.console.print(f"  Errored: {errors_pct:.1f}% ({errors}/{metrics.total})")
        self.console.print(f"  Average score (attempted): {metrics.avg_score_attempted:.2f}")
        self.console.print(f"  Average score (total): {metrics.avg_score_total:.2f}")

        # per-metric aggregates
        if hasattr(metrics, "by_metric") and metrics.by_metric:
            self.console.print("\n[bold]Metrics by Metric:[/bold]")
            metrics_table = Table()
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Avg Score (Attempted)", style="white")
            metrics_table.add_column("Avg Score (Total)", style="white")
            # build key->label mapping from config
            label_map = {}
            if "graders" in result.config and isinstance(result.config["graders"], dict):
                for key, gspec in result.config["graders"].items():
                    label_map[key] = gspec.get("display_name") or key

            for key, agg in metrics.by_metric.items():
                label = label_map.get(key, key)
                metrics_table.add_row(label, f"{agg.avg_score_attempted:.2f}", f"{agg.avg_score_total:.2f}")
            self.console.print(metrics_table)

        # per-model metrics
        if metrics.per_model:
            self.console.print("\n[bold]Per-Model Metrics:[/bold]")
            model_table = Table()
            model_table.add_column("Model", style="cyan")
            model_table.add_column("Samples", style="white")
            model_table.add_column("Attempted", style="white")
            model_table.add_column("Avg Score (Attempted)", style="white")
            model_table.add_column("Avg Score (Total)", style="white")

            for model_metrics in metrics.per_model:
                model_table.add_row(
                    model_metrics.model_name,
                    str(model_metrics.total),
                    str(model_metrics.total_attempted),
                    f"{model_metrics.avg_score_attempted:.2f}",
                    f"{model_metrics.avg_score_total:.2f}",
                )

            self.console.print(model_table)

        # gate status
        gate = result.config["gate"]
        gate_kind = gate.get("kind", "simple")
        status = "[green]PASSED[/green]" if result.gates_passed else "[red]FAILED[/red]"

        op_symbols = {"gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "eq": "="}

        # format gate description based on kind
        if gate_kind == "simple":
            gate_op = gate["op"]
            gate_value = gate["value"]
            gate_aggregation = gate.get("aggregation", "avg_score")
            gate_metric_key = gate.get("metric_key")
            op_symbol = op_symbols.get(gate_op, gate_op)

            # get display name
            display_label = None
            if gate_metric_key and "graders" in result.config and isinstance(result.config["graders"], dict):
                gspec = result.config["graders"].get(gate_metric_key)
                if gspec:
                    display_label = gspec.get("display_name")
            metric_label = display_label or gate_metric_key or "metric"

            gate_desc = f"'{metric_label}' {gate_aggregation} {op_symbol} {gate_value}"

        elif gate_kind == "weighted_average":
            weights = gate.get("weights", {})
            gate_op = gate["op"]
            gate_value = gate["value"]
            gate_aggregation = gate.get("aggregation", "avg_score")
            op_symbol = op_symbols.get(gate_op, gate_op)

            weight_strs = [f"{k}({w})" for k, w in weights.items()]
            gate_desc = f"weighted_average[{', '.join(weight_strs)}] {gate_aggregation} {op_symbol} {gate_value}"

        elif gate_kind == "logical":
            operator = gate.get("operator", "and").upper()
            num_conditions = len(gate.get("conditions", []))
            gate_desc = f"logical {operator} with {num_conditions} conditions"

        else:
            gate_desc = f"unknown gate kind: {gate_kind}"

        self.console.print(f"\n[bold]Gate:[/bold] {gate_desc} → {status}")

        # sample results table
        self.console.print("\n[bold]Sample Results:[/bold]")

        total_samples = len(result.results)
        samples_to_display = result.results[:MAX_SAMPLES_DISPLAY]

        if total_samples > MAX_SAMPLES_DISPLAY:
            self.console.print(f"[dim]Showing first {MAX_SAMPLES_DISPLAY} of {total_samples} samples[/dim]")

        table = Table(show_header=True, header_style="bold cyan", border_style="blue", box=ROUNDED)
        table.add_column("Sample", style="cyan", no_wrap=True)
        table.add_column("Agent ID", style="dim cyan", no_wrap=False)
        table.add_column("Model", style="yellow", no_wrap=True)

        # determine available metrics and display labels
        metric_keys = []
        metric_labels = {}
        if "graders" in result.config and isinstance(result.config["graders"], dict):
            for k, gspec in result.config["graders"].items():
                metric_keys.append(k)
                metric_labels[k] = gspec.get("display_name") or k

        # add two sub-columns per metric: score + rationale
        for mk in metric_keys:
            lbl = metric_labels.get(mk, mk)
            table.add_column(f"{lbl} score", style="white", no_wrap=True)
            table.add_column(f"{lbl} rationale", style="dim", no_wrap=False)

        for sample_result in samples_to_display:
            # build per-metric cells in config order
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
                        try:
                            s_val = float(g.get("score"))
                            r_text = g.get("rationale", "")
                        except Exception:
                            s_val = None
                            r_text = ""
                    score_cell = f"{s_val:.2f}" if s_val is not None else "-"
                    cells.extend([score_cell, r_text])

            table.add_row(
                f"Sample {sample_result.sample.id + 1}",
                sample_result.agent_id or "-",
                sample_result.model_name or "-",
                *cells,
            )

        self.console.print(table)

        if total_samples > MAX_SAMPLES_DISPLAY:
            self.console.print(
                f"[dim]... and {total_samples - MAX_SAMPLES_DISPLAY} more samples (see output file for complete results)[/dim]"
            )
