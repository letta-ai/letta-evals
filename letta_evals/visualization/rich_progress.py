import asyncio
import contextlib
import time
from enum import Enum
from typing import Any, Dict, List, Optional

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
from letta_evals.utils import build_turn_symbols, calculate_turn_average
from letta_evals.visualization.base import ProgressCallback
from letta_evals.visualization.state import (
    ProgressEvent,
    SampleProgress,
    SampleState,
    VisualizationStats,
    get_last_update_key,
    is_active_state,
    is_completed_state,
    is_terminal_state,
)
from letta_evals.visualization.summary import (
    build_rich_sample_results_table,
    format_gate_description,
    get_displayed_sample_results,
    get_metric_labels,
    print_basic_overall_metrics,
    print_remaining_samples_notice,
    print_truncated_samples_notice,
)


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
        update_freq: float = 2.0,
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
        self.frame_interval = (1.0 / update_freq) if update_freq > 0 else 0.25
        self.cached_mode = cached_mode
        self.metric_labels: Dict[str, str] = metric_labels or {}
        # live aggregates per metric key
        self.metric_totals: Dict[str, float] = {}
        self.metric_counts: Dict[str, int] = {}

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

        self.completed_count = 0
        self.error_count = 0
        self.total_score = 0.0
        self.score_count = 0
        self._dirty = False
        self._event_queue: Optional[asyncio.Queue[ProgressEvent]] = None
        self._dirty_event: Optional[asyncio.Event] = None
        self._event_task: Optional[asyncio.Task[None]] = None
        self._render_task: Optional[asyncio.Task[None]] = None
        self._background_error: Optional[BaseException] = None
        self._reset_visualization_stats()

    def _get_state_icon(self, state: SampleState) -> Text:
        """Get icon for sample state"""
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
        """Get text representation of sample state"""
        icon = self._get_state_icon(sample.state)

        if sample.state == SampleState.SENDING_MESSAGES and sample.total_messages > 0:
            text = Text()
            text.append(icon)
            text.append(f" sending [{sample.messages_sent}/{sample.total_messages}]")
            return text
        elif sample.state == SampleState.GRADING_TURNS and sample.total_turns > 0:
            text = Text()
            text.append(icon)
            text.append(f" grading [{sample.turns_graded}/{sample.total_turns}]")
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

    def _create_progress_with_metrics(self) -> Panel:
        """Create progress bar with inline metrics"""
        completed = self.completed_count + self.error_count

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
                label = self.metric_labels.get(key, key)
                if not first:
                    chips.append("   ")
                chips.append(f"{label}: {avg:.2f}", style="bold white")
                first = False
        chips.append("   ")
        chips.append(f"✓ {self.completed_count}", style="green")
        if self.error_count:
            chips.append("   ")
            chips.append(f"⚠ {self.error_count}", style="yellow")

        content = Group(self.main_progress, Text(""), chips)

        return Panel(content, box=ROUNDED, border_style="blue", padding=(0, 1))

    def _detail_layout_budget(self) -> tuple[int, int]:
        """Split the available height between active and completed panels."""
        available_lines = max(12, self.console.height - 10)
        has_completed = any(is_completed_state(sample.state) for sample in self.samples.values())

        if has_completed:
            completed_panel_size = min(11, max(6, available_lines // 3))
        else:
            completed_panel_size = 5

        active_panel_size = max(7, available_lines - completed_panel_size)
        return active_panel_size, completed_panel_size

    def _panel_row_limit(self, panel_size: int) -> int:
        """Approximate how many table rows fit in a panel of the given size."""
        return max(1, panel_size - 5)

    def _select_active_rows(self, limit: Optional[int] = None) -> List[SampleProgress]:
        """Select only currently active rows for the live top panel."""
        rows = [sample for sample in self.samples.values() if is_active_state(sample.state)]
        rows.sort(key=lambda sample: (sample.model_name or "", sample.sample_id))
        if limit is None:
            return rows
        return rows[:limit]

    def _select_completed_rows(self, limit: Optional[int] = None) -> List[SampleProgress]:
        """Select the most recent completed or errored rows for the bottom panel."""
        rows = [sample for sample in self.samples.values() if is_completed_state(sample.state)]
        rows.sort(key=lambda sample: (-get_last_update_key(sample), sample.model_name or "", sample.sample_id))
        if limit is None:
            return rows
        return rows[:limit]

    def _create_samples_table(self, rows: List[SampleProgress], title: str, empty_message: str):
        """Render a table for a specific sample slice."""
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

            # Check if we're in per-turn grading mode
            if s.state == SampleState.GRADING_TURNS and s.turn_scores:
                if self.metric_labels:
                    # Show per-grader progress in respective columns
                    for mk in metric_keys:
                        grader_scores = s.turn_scores.get(mk)
                        if grader_scores:
                            score_cell = f"{calculate_turn_average(grader_scores):.2f}"
                            rat = build_turn_symbols(grader_scores)
                        else:
                            score_cell = "-"
                            rat = ""
                        cells.extend([score_cell, rat])
                else:
                    # Single grader mode - use first/default grader
                    first_grader = next(iter(s.turn_scores.values()), None)
                    if first_grader:
                        score_cell = f"{calculate_turn_average(first_grader):.2f}"
                        rat = build_turn_symbols(first_grader)
                    else:
                        score_cell = "-"
                        rat = ""
                    cells.extend([score_cell, rat])
            elif self.metric_labels:
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
            elif s.state == SampleState.GRADING_TURNS and s.total_turns > 0:
                p = s.turns_graded / s.total_turns
                bar_width = max(10, min(30, max(10, self.console.width // 6)))
                filled = int(p * bar_width)
                bar = "▰" * filled + "▱" * (bar_width - filled)
                details = f"{bar}  turn {s.turns_graded}/{s.total_turns}"
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

    def _create_active_view(self, limit: int):
        """Create the top panel with only in-flight work."""
        rows = self._select_active_rows(limit=limit)
        title = f"Active Samples · showing {len(rows)}"
        return self._create_samples_table(rows, title, "No active samples")

    def _create_completed_view(self, limit: int):
        """Create the bottom panel with the most recent finished work."""
        rows = self._select_completed_rows(limit=limit)
        title = f"Recent Completed · showing {len(rows)}"
        return self._create_samples_table(rows, title, "No completed samples yet")

    def _render(self) -> Layout:
        """Render the complete progress display"""
        layout = Layout()
        active_panel_size, completed_panel_size = self._detail_layout_budget()
        active_row_limit = self._panel_row_limit(active_panel_size)
        completed_row_limit = self._panel_row_limit(completed_panel_size)

        details_layout = Layout()
        details_layout.split_column(
            Layout(self._create_active_view(active_row_limit), size=active_panel_size),
            Layout(self._create_completed_view(completed_row_limit), size=completed_panel_size),
        )

        layout.split_column(
            Layout(self._create_header_panel(), size=4),
            Layout(self._create_progress_with_metrics(), size=5),  # increased size to show both progress and metrics
            Layout(details_layout),
        )

        return layout

    def reset(self):
        """Reset counters and state for a new run"""
        self.start_time = time.time()
        self.completed_count = 0
        self.error_count = 0
        self.total_score = 0.0
        self.score_count = 0
        self._background_error = None
        self._reset_visualization_stats()
        self.samples.clear()
        self.metric_totals.clear()
        self.metric_counts.clear()
        if self.main_task_id is not None:
            self.main_progress.update(self.main_task_id, completed=0)
        self._mark_dirty()

    async def start(self):
        """Start the progress display"""
        self.start_time = time.time()
        self._background_error = None
        task_description = "Re-grading cached trajectories" if self.cached_mode else "Evaluating samples"
        self.main_task_id = self.main_progress.add_task(
            task_description,
            total=self.total_samples,
            completed=0,
        )

        # initialize samples placeholder - actual entries will be created as evaluations start
        # no longer pre-populate since we need model_name for the key

        self.live = Live(
            console=self.console,
            auto_refresh=False,
            transient=False,
            vertical_overflow="ellipsis",
            get_renderable=self._render,
        )
        self.live.start()
        self._start_background_tasks()
        self._mark_dirty()
        self._refresh_live(reason="start")

    def stop(self):
        """Stop the progress display"""
        self._stop_background_tasks()
        if self.live:
            if self._dirty:
                with contextlib.suppress(Exception):
                    self._refresh_live(reason="stop")
            self.live.stop()
            self.live = None
            self.console.print()

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._dirty_mark_count += 1
        if self._dirty_event is not None:
            self._dirty_event.set()

    def _refresh_live(self, reason: str = "tick") -> None:
        if not self.live or not self._dirty:
            return
        self.live.refresh()
        self._refresh_count += 1
        self._refresh_reasons[reason] = self._refresh_reasons.get(reason, 0) + 1
        self._dirty = False

    def _record_background_error(self, task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError):
            exc = task.exception()
            if exc is not None and self._background_error is None:
                self._background_error = exc

    def _raise_background_error(self) -> None:
        if self._background_error is not None:
            raise RuntimeError("EvalProgress background task failed") from self._background_error

    def _start_background_tasks(self) -> None:
        self._stop_background_tasks()
        self._event_queue = asyncio.Queue()
        self._dirty_event = asyncio.Event()
        self._event_task = asyncio.create_task(self._event_loop())
        self._render_task = asyncio.create_task(self._render_loop())
        self._event_task.add_done_callback(self._record_background_error)
        self._render_task.add_done_callback(self._record_background_error)

    def _stop_background_tasks(self) -> None:
        for task in (self._event_task, self._render_task):
            if task is not None and not task.done():
                task.cancel()
        self._event_task = None
        self._render_task = None
        self._event_queue = None
        self._dirty_event = None

    async def _event_loop(self) -> None:
        queue = self._event_queue
        if queue is None:
            return
        while True:
            event = await queue.get()
            try:
                self._apply_event(event)
                self._events_processed += 1
            except Exception as exc:
                if event.ack is not None and not event.ack.done():
                    event.ack.set_exception(exc)
                raise
            else:
                self._mark_dirty()
                if event.ack is not None and not event.ack.done():
                    event.ack.set_result(None)
            finally:
                queue.task_done()

    async def _render_loop(self) -> None:
        dirty_event = self._dirty_event
        if dirty_event is None:
            return
        while True:
            await dirty_event.wait()
            dirty_event.clear()
            self._render_wakeup_count += 1
            await asyncio.sleep(self.frame_interval)
            self._refresh_live()

    async def _emit_event(self, kind: str, **payload: Any) -> None:
        self._raise_background_error()
        self._events_emitted += 1

        if self._event_queue is None:
            self._apply_event(ProgressEvent(kind=kind, payload=payload))
            self._events_processed += 1
            self._mark_dirty()
            return

        loop = asyncio.get_running_loop()
        ack: asyncio.Future[None] = loop.create_future()
        await self._event_queue.put(ProgressEvent(kind=kind, payload=payload, ack=ack))
        self._max_queue_depth = max(self._max_queue_depth, self._event_queue.qsize())
        await ack
        self._raise_background_error()

    def _apply_event(self, event: ProgressEvent) -> None:
        if event.kind == "update_sample_state":
            self._apply_sample_state_update(**event.payload)
            return
        raise ValueError(f"Unknown progress event kind: {event.kind}")

    def _apply_sample_state_update(
        self,
        sample_id: int,
        state: SampleState,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Apply a state transition inside the single-threaded reducer."""
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

        is_new_completion = not is_terminal_state(previous_state) and is_terminal_state(state)

        if state == SampleState.COMPLETED and is_new_completion:
            self.completed_count += 1

            if sample.score is not None:
                self.total_score += sample.score
                self.score_count += 1

            # Update per-metric aggregates before render so the header shows correct averages
            if sample.metric_scores:
                for mkey, mscore in sample.metric_scores.items():
                    self.metric_totals[mkey] = self.metric_totals.get(mkey, 0.0) + (mscore or 0.0)
                    self.metric_counts[mkey] = self.metric_counts.get(mkey, 0) + 1

            completed = self.completed_count + self.error_count
            self.main_progress.update(self.main_task_id, completed=completed)

        elif state == SampleState.ERROR and is_new_completion:
            self.error_count += 1
            completed = self.completed_count + self.error_count
            self.main_progress.update(self.main_task_id, completed=completed)

    async def update_sample_state(
        self,
        sample_id: int,
        state: SampleState,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ):
        """Queue a state transition for the reducer task."""
        await self._emit_event(
            "update_sample_state",
            sample_id=sample_id,
            state=state,
            agent_id=agent_id,
            model_name=model_name,
            **kwargs,
        )

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

    async def agent_created(
        self, sample_id: int, agent_id: str, model_name: Optional[str] = None, from_cache: bool = False
    ):
        """Update sample with agent_id once agent is provisioned."""
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

    async def turn_graded(
        self,
        sample_id: int,
        turn_num: int,
        total_turns: int,
        turn_score: float,
        grader_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """Update progress for per-turn grading"""
        key = (sample_id, model_name)

        # Get existing from_cache flag
        existing_from_cache = False
        if key in self.samples:
            existing_from_cache = self.samples[key].from_cache
        elif model_name is not None and (sample_id, None) in self.samples:
            existing_from_cache = self.samples[(sample_id, None)].from_cache

        # Initialize sample and turn_scores dict BEFORE updating state
        if key not in self.samples:
            self.samples[key] = SampleProgress(sample_id, agent_id=agent_id, model_name=model_name)
        if self.samples[key].turn_scores is None:
            self.samples[key].turn_scores = {}

        # Initialize this grader's turn scores if needed
        gk = grader_key or "_default"
        if gk not in self.samples[key].turn_scores:
            self.samples[key].turn_scores[gk] = [None] * total_turns

        # Update score at the turn index for this grader
        if turn_num < len(self.samples[key].turn_scores[gk]):
            self.samples[key].turn_scores[gk][turn_num] = turn_score

        await self.update_sample_state(
            sample_id,
            SampleState.GRADING_TURNS,
            agent_id=agent_id,
            model_name=model_name,
            from_cache=existing_from_cache,
            turns_graded=turn_num + 1,  # turn_num is 0-indexed
            total_turns=total_turns,
        )

    async def sample_completed(
        self,
        sample_id: int,
        agent_id: Optional[str] = None,
        score: Optional[float] = None,
        model_name: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
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
            score=score,
            rationale=rationale,
            from_cache=existing_from_cache,
            metric_scores=metric_scores,
            metric_rationales=metric_rationales,
        )

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

    def get_stats_snapshot(self) -> VisualizationStats:
        """Return instrumentation counters for tuning the live renderer."""
        runtime_seconds = (time.time() - self.start_time) if self.start_time is not None else 0.0
        pending_events = self._event_queue.qsize() if self._event_queue is not None else 0
        return VisualizationStats(
            events_emitted=self._events_emitted,
            events_processed=self._events_processed,
            refreshes=self._refresh_count,
            refreshes_by_reason=dict(self._refresh_reasons),
            render_wakeups=self._render_wakeup_count,
            dirty_marks=self._dirty_mark_count,
            max_queue_depth=self._max_queue_depth,
            pending_events=pending_events,
            frame_interval=self.frame_interval,
            runtime_seconds=runtime_seconds,
        )

    def _reset_visualization_stats(self) -> None:
        self._events_emitted = 0
        self._events_processed = 0
        self._refresh_count = 0
        self._refresh_reasons: Dict[str, int] = {}
        self._render_wakeup_count = 0
        self._dirty_mark_count = 0
        self._max_queue_depth = 0

    async def suite_completed(self, result):
        """Display summary and detailed results after evaluation completes"""
        self.console.print()
        self.console.print(f"[bold]Evaluation Results: {result.suite}[/bold]")
        if self.cached_mode:
            self.console.print("[dim]Note: Results re-graded from cached trajectories[/dim]")
        self.console.print("=" * 50)

        metrics = result.metrics
        print_basic_overall_metrics(self.console, metrics)

        # usage metrics
        if metrics.usage_metrics:
            self.console.print("\n[bold]Usage:[/bold]")
            self.console.print(f"  Total prompt tokens: {metrics.usage_metrics.total_prompt_tokens:,}")
            self.console.print(f"  Total completion tokens: {metrics.usage_metrics.total_completion_tokens:,}")
            if metrics.usage_metrics.total_cost is not None:
                self.console.print(f"  Total cost: ${metrics.usage_metrics.total_cost:.4f}")
            if metrics.usage_metrics.total_cached_input_tokens > 0:
                self.console.print(f"  Total cached input tokens: {metrics.usage_metrics.total_cached_input_tokens:,}")
            if metrics.usage_metrics.total_cache_write_tokens > 0:
                self.console.print(f"  Total cache write tokens: {metrics.usage_metrics.total_cache_write_tokens:,}")
            if metrics.usage_metrics.total_reasoning_tokens > 0:
                self.console.print(f"  Total reasoning tokens: {metrics.usage_metrics.total_reasoning_tokens:,}")

        # per-metric aggregates
        if hasattr(metrics, "by_metric") and metrics.by_metric:
            self.console.print("\n[bold]Metrics by Metric:[/bold]")
            metrics_table = Table()
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Avg Score (Attempted)", style="white")
            metrics_table.add_column("Avg Score (Total)", style="white")
            label_map = get_metric_labels(result.config)

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

            # per-model usage metrics
            has_usage_data = any(m.usage_metrics for m in metrics.per_model)
            if has_usage_data:
                self.console.print("\n[bold]Per-Model Usage:[/bold]")
                usage_table = Table()
                usage_table.add_column("Model", style="cyan")
                usage_table.add_column("Prompt Tokens", style="white")
                usage_table.add_column("Completion Tokens", style="white")
                usage_table.add_column("Cost", style="white")
                usage_table.add_column("Cached Input", style="dim white")
                usage_table.add_column("Cache Write", style="dim white")
                usage_table.add_column("Reasoning", style="dim white")

                for model_metrics in metrics.per_model:
                    if model_metrics.usage_metrics:
                        u = model_metrics.usage_metrics
                        usage_table.add_row(
                            model_metrics.model_name,
                            f"{u.total_prompt_tokens:,}",
                            f"{u.total_completion_tokens:,}",
                            f"${u.total_cost:.4f}" if u.total_cost is not None else "-",
                            f"{u.total_cached_input_tokens:,}" if u.total_cached_input_tokens > 0 else "-",
                            f"{u.total_cache_write_tokens:,}" if u.total_cache_write_tokens > 0 else "-",
                            f"{u.total_reasoning_tokens:,}" if u.total_reasoning_tokens > 0 else "-",
                        )

                self.console.print(usage_table)

        # gate status
        status = "[green]PASSED[/green]" if result.gates_passed else "[red]FAILED[/red]"
        gate_desc = format_gate_description(
            result.config,
            prefer_display_label=True,
            quote_metric_label=True,
            default_metric_label="metric",
        )

        self.console.print(f"\n[bold]Gate:[/bold] {gate_desc} → {status}")

        # sample results table
        self.console.print("\n[bold]Sample Results:[/bold]")
        total_samples, displayed_results = get_displayed_sample_results(result)
        print_truncated_samples_notice(self.console, total_samples, len(displayed_results))
        self.console.print(build_rich_sample_results_table(result))
        print_remaining_samples_notice(self.console, total_samples, len(displayed_results))
