from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SampleState(Enum):
    """States a sample can be in during evaluation."""

    QUEUED = "queued"
    LOADING_AGENT = "loading"
    SENDING_MESSAGES = "sending"
    GRADING = "grading"
    GRADING_TURNS = "grading_turns"
    COMPLETED = "completed"
    ERROR = "error"


ACTIVE_SAMPLE_STATES = frozenset(
    {
        SampleState.LOADING_AGENT,
        SampleState.SENDING_MESSAGES,
        SampleState.GRADING,
        SampleState.GRADING_TURNS,
    }
)
COMPLETED_SAMPLE_STATES = frozenset(
    {
        SampleState.COMPLETED,
        SampleState.ERROR,
    }
)


@dataclass
class SampleProgress:
    """Track progress of an individual sample."""

    sample_id: int
    state: SampleState = SampleState.QUEUED
    agent_id: Optional[str] = None
    model_name: Optional[str] = None
    score: Optional[float] = None
    target_cost: Optional[float] = None
    rationale: Optional[str] = None
    error: Optional[str] = None
    messages_sent: int = 0
    total_messages: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_update_ts: Optional[float] = None
    from_cache: bool = False
    metric_scores: Optional[Dict[str, float]] = None
    metric_rationales: Optional[Dict[str, str]] = None
    turns_graded: int = 0
    total_turns: int = 0
    turn_scores: Optional[Dict[str, List[Optional[float]]]] = None


@dataclass
class ProgressEvent:
    """Serializable progress update processed by the reducer task."""

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    ack: Optional[asyncio.Future[None]] = None


@dataclass(frozen=True)
class VisualizationStats:
    """Snapshot of visualization runtime behavior for tuning."""

    events_emitted: int
    events_processed: int
    refreshes: int
    refreshes_by_reason: Dict[str, int]
    render_wakeups: int
    dirty_marks: int
    max_queue_depth: int
    pending_events: int
    frame_interval: float
    runtime_seconds: float

    @property
    def avg_events_per_refresh(self) -> float:
        return self.events_processed / self.refreshes if self.refreshes > 0 else 0.0


def is_active_state(state: SampleState) -> bool:
    return state in ACTIVE_SAMPLE_STATES


def is_completed_state(state: SampleState) -> bool:
    return state in COMPLETED_SAMPLE_STATES


def is_terminal_state(state: SampleState) -> bool:
    return state in COMPLETED_SAMPLE_STATES


def get_last_update_key(sample: SampleProgress) -> float:
    return sample.last_update_ts or sample.end_time or sample.start_time or 0.0
