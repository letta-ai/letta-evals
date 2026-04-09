import asyncio
import contextlib
import time
from typing import Any, Dict, Optional

from rich.console import Console
from rich.live import Live
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

from letta_evals.visualization.base import ProgressCallback
from letta_evals.visualization.reducer import ProgressRuntimeState, ProgressStateReducer
from letta_evals.visualization.rich_renderer import RichProgressRenderer
from letta_evals.visualization.state import (
    ProgressEvent,
    SampleState,
    VisualizationStats,
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
        console: Optional[Console] = None,
        update_freq: float = 2.0,
        cached_mode: bool = False,
        metric_labels: Optional[Dict[str, str]] = None,
    ):
        self.total_samples = total_samples
        self.console = console or Console()
        self.frame_interval = (1.0 / update_freq) if update_freq > 0 else 0.25
        self.cached_mode = cached_mode
        renderer_metric_labels: Dict[str, str] = metric_labels or {}
        self._runtime_state = ProgressRuntimeState()
        self._reducer = ProgressStateReducer(self._runtime_state)
        self._renderer = RichProgressRenderer(
            console=self.console,
            suite_name=suite_name,
            target_kind=target_kind,
            grader_kind=grader_kind,
            rubric_model=rubric_model,
            max_concurrent=max_concurrent,
            metric_labels=renderer_metric_labels,
        )
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
        self._dirty = False
        self._event_queue: Optional[asyncio.Queue[ProgressEvent]] = None
        self._dirty_event: Optional[asyncio.Event] = None
        self._event_task: Optional[asyncio.Task[None]] = None
        self._render_task: Optional[asyncio.Task[None]] = None
        self._background_error: Optional[BaseException] = None
        self._reset_visualization_stats()

    def _render(self):
        """Render the complete progress display."""
        return self._renderer.render(self._runtime_state, self.main_progress)

    def reset(self):
        """Reset counters and state for a new run"""
        self.start_time = time.time()
        self._background_error = None
        self._reset_visualization_stats()
        self._reducer.reset()
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

    def _refresh_live(self, reason: str = "tick", *, force: bool = False) -> None:
        if not self.live or (not force and not self._dirty):
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
            if not self._dirty:
                try:
                    await asyncio.wait_for(dirty_event.wait(), timeout=self.frame_interval)
                except asyncio.TimeoutError:
                    self._render_wakeup_count += 1
                    self._refresh_live(reason="timer", force=True)
                    continue
                dirty_event.clear()

            self._render_wakeup_count += 1
            await asyncio.sleep(self.frame_interval)
            self._refresh_live(reason="dirty")

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
        result = self._reducer.apply_event(event)
        if result.progress_completed is not None:
            self.main_progress.update(self.main_task_id, completed=result.progress_completed)

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
        self._reducer.ensure_sample(sample_id, agent_id=agent_id, model_name=model_name)
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
        existing_from_cache = self._reducer.get_from_cache(sample_id, model_name)

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
        sample = self._reducer.record_turn_grade(
            sample_id,
            turn_num=turn_num,
            total_turns=total_turns,
            turn_score=turn_score,
            grader_key=grader_key,
            agent_id=agent_id,
            model_name=model_name,
        )

        await self.update_sample_state(
            sample_id,
            SampleState.GRADING_TURNS,
            agent_id=agent_id,
            model_name=model_name,
            from_cache=sample.from_cache,
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
        existing_from_cache = self._reducer.get_from_cache(sample_id, model_name)

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
        self.console.print(build_rich_sample_results_table(result.config, displayed_results))
        print_remaining_samples_notice(self.console, total_samples, len(displayed_results))
