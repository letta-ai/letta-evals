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
    model_name: Optional[str] = None
    passed: Optional[bool] = None
    score: Optional[float] = None
    error: Optional[str] = None
    messages_sent: int = 0
    total_messages: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_update_ts: Optional[float] = None
    from_cache: bool = False


class DisplayMode(Enum):
    """Display modes for progress visualization"""

    COMPACT = "compact"
    STANDARD = "standard"
    DETAILED = "detailed"


class EvalProgress:
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
            if sample.passed is not None:
                icon = Text("✓", style="green") if sample.passed else Text("✗", style="red")
            text = Text()
            text.append(icon)
            text.append(f" {'passed' if sample.passed else 'failed'}")
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

        content = Group(Align.center(header_title), Align.center(subtitle))

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
            row_text = Text(f"[{i+1:3d}-{min(i+samples_per_row, self.total_samples):3d}] ", style="dim")

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
            accuracy_text = "Accuracy: N/A"
            avg_score_text = "Avg: N/A"
        else:
            accuracy = (self.passed_count / completed) * 100 if completed > 0 else 0
            accuracy_text = f"Accuracy: {accuracy:.1f}%"
            avg_score = self.total_score / self.score_count if self.score_count > 0 else 0
            avg_score_text = f"Avg: {avg_score:.2f}"

        chips = Text()
        chips.append(f"  {accuracy_text}", style="bold white")
        chips.append("   ")
        chips.append(avg_score_text, style="bold white")
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
            accuracy_text = "N/A"
            avg_score_text = "N/A"
            correct_count = 0
        else:
            # Count failed agents and errors as incorrect (denominator = completed)
            correct_count = self.passed_count
            accuracy = (correct_count / completed) * 100 if completed > 0 else 0
            accuracy_text = f"{accuracy:.1f}%"
            avg_score = self.total_score / self.score_count if self.score_count > 0 else 0
            avg_score_text = f"{avg_score:.3f}"

        metrics_table = Table.grid(padding=1)
        metrics_table.add_column(style="cyan", justify="right")
        metrics_table.add_column(style="white")

        metrics_table.add_row("📊 Accuracy:", f"{accuracy_text} ({correct_count}/{completed})")

        if self.score_count > 0:
            metrics_table.add_row("📈 Avg Score:", avg_score_text)

        if self.failed_count > 0:
            failed_samples = [str(key[0] + 1) for key, s in self.samples.items() if s.passed is False][:5]
            failed_text = ", ".join(failed_samples)
            if len(failed_samples) < self.failed_count:
                failed_text += f" ... ({self.failed_count} total)"
            metrics_table.add_row("❌ Failed:", failed_text)

        if self.error_count > 0:
            metrics_table.add_row("⚠️ Errors:", str(self.error_count))

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

        table.add_column("#", style="cyan", width=12)
        table.add_column("Model", style="yellow", width=20)
        if self.grader_kind == GraderKind.RUBRIC.value and self.rubric_model:
            table.add_column("Rubric Model", style="magenta", width=15)
        table.add_column("Status", width=20)
        table.add_column("Score", width=6, justify="right")
        table.add_column("Time", width=6, justify="right")
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

            score_text = f"{s.score:.2f}" if s.score is not None else "-"

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
                details = "[green]✓ Passed[/green]" if s.passed else "[red]✗ Failed[/red]"
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
                s.model_name or "-",
            ]
            if self.grader_kind == GraderKind.RUBRIC.value and self.rubric_model:
                row_data.append(self.rubric_model)
            row_data.extend(
                [
                    self._get_state_text(s),
                    score_text,
                    time_text,
                    details,
                ]
            )
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

    async def update_sample_state(self, sample_id: int, state: SampleState, model_name: Optional[str] = None, **kwargs):
        """Update state of a sample"""
        key = (sample_id, model_name)
        if key not in self.samples:
            self.samples[key] = SampleProgress(sample_id, model_name=model_name)

        sample = self.samples[key]
        sample.state = state

        if state == SampleState.LOADING_AGENT and sample.start_time is None:
            sample.start_time = time.time()
        elif state in [SampleState.COMPLETED, SampleState.FAILED, SampleState.ERROR]:
            sample.end_time = time.time()

        for key, value in kwargs.items():
            if hasattr(sample, key):
                setattr(sample, key, value)
        # update last update timestamp for smart viewport
        sample.last_update_ts = time.time()

        if state == SampleState.COMPLETED:
            if sample.passed is True:
                self.passed_count += 1
            elif sample.passed is False:
                self.failed_count += 1

            if sample.score is not None:
                self.total_score += sample.score
                self.score_count += 1

            completed = self.passed_count + self.failed_count + self.error_count
            self.main_progress.update(self.main_task_id, completed=completed)

        elif state == SampleState.ERROR:
            self.error_count += 1
            completed = self.passed_count + self.failed_count + self.error_count
            self.main_progress.update(self.main_task_id, completed=completed)

        if self.live:
            self.live.update(self._render())

    async def sample_started(self, sample_id: int, model_name: Optional[str] = None):
        """Mark sample as started"""
        key = (sample_id, model_name)
        if key not in self.samples:
            self.samples[key] = SampleProgress(sample_id, model_name=model_name)
        # skip loading state if using cached trajectories
        if not self.cached_mode:
            await self.update_sample_state(sample_id, SampleState.LOADING_AGENT, model_name=model_name)

    async def agent_loading(self, sample_id: int, model_name: Optional[str] = None, from_cache: bool = False):
        """Mark sample as loading agent"""
        await self.update_sample_state(
            sample_id, SampleState.LOADING_AGENT, model_name=model_name, from_cache=from_cache
        )

    async def message_sending(
        self, sample_id: int, message_num: int, total_messages: int, model_name: Optional[str] = None
    ):
        """Update message sending progress"""
        await self.update_sample_state(
            sample_id,
            SampleState.SENDING_MESSAGES,
            model_name=model_name,
            messages_sent=message_num,
            total_messages=total_messages,
        )

    async def grading_started(self, sample_id: int, model_name: Optional[str] = None):
        """Mark sample as being graded"""
        key = (sample_id, model_name)
        existing_from_cache = self.samples[key].from_cache if key in self.samples else False
        await self.update_sample_state(
            sample_id, SampleState.GRADING, model_name=model_name, from_cache=existing_from_cache
        )

    async def sample_completed(
        self, sample_id: int, passed: bool, score: Optional[float] = None, model_name: Optional[str] = None
    ):
        """Mark sample as completed"""
        # preserve from_cache flag if it was set
        key = (sample_id, model_name)
        existing_from_cache = self.samples[key].from_cache if key in self.samples else False
        await self.update_sample_state(
            sample_id,
            SampleState.COMPLETED,
            model_name=model_name,
            passed=passed,
            score=score,
            from_cache=existing_from_cache,
        )

    async def sample_error(self, sample_id: int, error: str, model_name: Optional[str] = None):
        """Mark sample as having an error"""
        await self.update_sample_state(
            sample_id,
            SampleState.ERROR,
            model_name=model_name,
            error=error,
        )
