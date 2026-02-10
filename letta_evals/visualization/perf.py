from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List


@dataclass
class RenderPerfTracker:
    """Track rendering and event throughput statistics for progress UIs."""

    render_history_size: int = 2048
    start_ts: float = field(default_factory=time.time)
    event_count: int = 0
    event_counts: Counter[str] = field(default_factory=Counter)
    render_count: int = 0
    render_durations_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=2048))
    coalesced_render_requests: int = 0
    noop_state_updates: int = 0

    def __post_init__(self) -> None:
        if self.render_history_size != 2048:
            self.render_durations_ms = deque(maxlen=self.render_history_size)

    def reset(self) -> None:
        self.start_ts = time.time()
        self.event_count = 0
        self.event_counts.clear()
        self.render_count = 0
        self.render_durations_ms.clear()
        self.coalesced_render_requests = 0
        self.noop_state_updates = 0

    def record_event(self, event_name: str) -> None:
        self.event_count += 1
        self.event_counts[event_name] += 1

    def record_render_request(self, *, already_dirty: bool) -> None:
        if already_dirty:
            self.coalesced_render_requests += 1

    def record_noop_state_update(self) -> None:
        self.noop_state_updates += 1

    def record_render_duration(self, duration_ms: float) -> None:
        self.render_count += 1
        self.render_durations_ms.append(duration_ms)

    @staticmethod
    def _percentile(values: List[float], pct: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        if len(sorted_values) == 1:
            return sorted_values[0]

        pos = (pct / 100.0) * (len(sorted_values) - 1)
        lower = int(pos)
        upper = min(lower + 1, len(sorted_values) - 1)
        frac = pos - lower
        return (sorted_values[lower] * (1.0 - frac)) + (sorted_values[upper] * frac)

    def snapshot(self) -> Dict[str, object]:
        elapsed = max(0.001, time.time() - self.start_ts)
        render_ms = list(self.render_durations_ms)
        return {
            "uptime_sec": round(elapsed, 3),
            "events_total": self.event_count,
            "events_per_sec": round(self.event_count / elapsed, 3),
            "renders_total": self.render_count,
            "renders_per_sec": round(self.render_count / elapsed, 3),
            "render_p50_ms": round(self._percentile(render_ms, 50), 3),
            "render_p95_ms": round(self._percentile(render_ms, 95), 3),
            "coalesced_render_requests": self.coalesced_render_requests,
            "noop_state_updates": self.noop_state_updates,
            "event_counts": dict(self.event_counts),
        }
