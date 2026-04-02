from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from letta_evals.visualization.state import (
    ProgressEvent,
    SampleProgress,
    SampleState,
    is_active_state,
    is_terminal_state,
)

SampleKey = tuple[int, Optional[str]]


@dataclass
class ProgressRuntimeState:
    """Mutable state consumed by the live progress renderer."""

    samples: Dict[SampleKey, SampleProgress] = field(default_factory=dict)
    metric_totals: Dict[str, float] = field(default_factory=dict)
    metric_counts: Dict[str, int] = field(default_factory=dict)
    completed_count: int = 0
    error_count: int = 0
    next_active_sort_sequence: int = 1
    next_completion_sequence: int = 1


@dataclass(frozen=True)
class ReducerResult:
    """Result metadata emitted after applying a reducer action."""

    progress_completed: Optional[int] = None


class ProgressStateReducer:
    """Apply serialized progress events to mutable visualization state."""

    def __init__(self, state: Optional[ProgressRuntimeState] = None) -> None:
        self.state = state or ProgressRuntimeState()

    def reset(self) -> None:
        self.state.samples.clear()
        self.state.metric_totals.clear()
        self.state.metric_counts.clear()
        self.state.completed_count = 0
        self.state.error_count = 0
        self.state.next_active_sort_sequence = 1
        self.state.next_completion_sequence = 1

    def ensure_sample(
        self,
        sample_id: int,
        *,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> SampleProgress:
        key = (sample_id, model_name)

        if model_name is not None:
            old_key = (sample_id, None)
            if old_key in self.state.samples and key not in self.state.samples:
                self.state.samples[key] = self.state.samples.pop(old_key)
                self.state.samples[key].model_name = model_name

        if key not in self.state.samples:
            self.state.samples[key] = SampleProgress(sample_id, agent_id=agent_id, model_name=model_name)

        sample = self.state.samples[key]
        if agent_id is not None and sample.agent_id != agent_id:
            sample.agent_id = agent_id
        if model_name is not None and sample.model_name != model_name:
            sample.model_name = model_name
        return sample

    def get_from_cache(self, sample_id: int, model_name: Optional[str] = None) -> bool:
        sample = self.get_sample(sample_id, model_name)
        return sample.from_cache if sample is not None else False

    def get_sample(self, sample_id: int, model_name: Optional[str] = None) -> Optional[SampleProgress]:
        key = (sample_id, model_name)
        if key in self.state.samples:
            return self.state.samples[key]
        if model_name is not None:
            return self.state.samples.get((sample_id, None))
        return None

    def record_turn_grade(
        self,
        sample_id: int,
        *,
        turn_num: int,
        total_turns: int,
        turn_score: float,
        grader_key: Optional[str] = None,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> SampleProgress:
        sample = self.ensure_sample(sample_id, agent_id=agent_id, model_name=model_name)
        if sample.turn_scores is None:
            sample.turn_scores = {}

        key = grader_key or "_default"
        if key not in sample.turn_scores:
            sample.turn_scores[key] = [None] * total_turns

        if turn_num < len(sample.turn_scores[key]):
            sample.turn_scores[key][turn_num] = turn_score

        return sample

    def apply_event(self, event: ProgressEvent) -> ReducerResult:
        if event.kind == "update_sample_state":
            return self.apply_sample_state_update(**event.payload)
        raise ValueError(f"Unknown progress event kind: {event.kind}")

    def apply_sample_state_update(
        self,
        sample_id: int,
        state: SampleState,
        agent_id: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> ReducerResult:
        """Apply a state transition inside the single-threaded reducer."""
        sample = self.ensure_sample(sample_id, agent_id=agent_id, model_name=model_name)
        previous_state = sample.state
        sample.state = state

        now = time.time()
        if state == SampleState.LOADING_AGENT and sample.start_time is None:
            sample.start_time = now
        elif state in [SampleState.COMPLETED, SampleState.ERROR]:
            sample.end_time = now

        for key, value in kwargs.items():
            if hasattr(sample, key):
                setattr(sample, key, value)
        sample.last_update_ts = now

        if previous_state != state and is_active_state(state):
            sample.active_sort_sequence = self.state.next_active_sort_sequence
            self.state.next_active_sort_sequence += 1

        is_new_completion = not is_terminal_state(previous_state) and is_terminal_state(state)

        if is_new_completion:
            sample.completion_sequence = self.state.next_completion_sequence
            self.state.next_completion_sequence += 1

        if state == SampleState.COMPLETED and is_new_completion:
            self.state.completed_count += 1

            if sample.metric_scores:
                for metric_key, metric_score in sample.metric_scores.items():
                    self.state.metric_totals[metric_key] = self.state.metric_totals.get(metric_key, 0.0) + (
                        metric_score or 0.0
                    )
                    self.state.metric_counts[metric_key] = self.state.metric_counts.get(metric_key, 0) + 1

            return ReducerResult(progress_completed=self.state.completed_count + self.state.error_count)

        if state == SampleState.ERROR and is_new_completion:
            self.state.error_count += 1
            return ReducerResult(progress_completed=self.state.completed_count + self.state.error_count)

        return ReducerResult()
