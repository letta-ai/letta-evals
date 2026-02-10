from __future__ import annotations

from letta_evals.visualization.perf import RenderPerfTracker


def test_perf_tracker_records_events_and_renders() -> None:
    tracker = RenderPerfTracker()

    tracker.record_event("sample_started")
    tracker.record_event("sample_started")
    tracker.record_event("sample_completed")
    tracker.record_render_request(already_dirty=False)
    tracker.record_render_request(already_dirty=True)
    tracker.record_render_duration(1.2)
    tracker.record_render_duration(3.4)
    tracker.record_noop_state_update()

    snapshot = tracker.snapshot()

    assert snapshot["events_total"] == 3
    assert snapshot["event_counts"]["sample_started"] == 2
    assert snapshot["event_counts"]["sample_completed"] == 1
    assert snapshot["renders_total"] == 2
    assert snapshot["coalesced_render_requests"] == 1
    assert snapshot["noop_state_updates"] == 1


def test_perf_tracker_reset_clears_counters() -> None:
    tracker = RenderPerfTracker()
    tracker.record_event("sample_started")
    tracker.record_render_request(already_dirty=True)
    tracker.record_render_duration(2.0)
    tracker.record_noop_state_update()

    tracker.reset()
    snapshot = tracker.snapshot()

    assert snapshot["events_total"] == 0
    assert snapshot["renders_total"] == 0
    assert snapshot["coalesced_render_requests"] == 0
    assert snapshot["noop_state_updates"] == 0
    assert snapshot["event_counts"] == {}
