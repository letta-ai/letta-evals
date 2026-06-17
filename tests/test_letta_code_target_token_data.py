"""Unit tests for LettaCodeTarget token-data fetching."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from letta_evals.targets.letta_code_target import LettaCodeTarget


class _FakeRuns:
    def __init__(self, *, items, runs_by_id, reject_order: bool = False):
        self.items = items
        self.runs_by_id = runs_by_id
        self.reject_order = reject_order
        self.list_calls = []
        self.retrieve_ids = []

    async def list(self, **kwargs):
        self.list_calls.append(kwargs)
        if self.reject_order and "order" in kwargs:
            raise TypeError("list() got an unexpected keyword argument 'order'")
        return SimpleNamespace(items=self.items)

    async def retrieve(self, *, run_id):
        self.retrieve_ids.append(run_id)
        return self.runs_by_id[run_id]


def _run_summary(run_id: str, created_at: datetime):
    return SimpleNamespace(id=run_id, created_at=created_at)


def _run_with_turns(*turns):
    return SimpleNamespace(metadata={"result": {"turns": list(turns)}})


def _assistant_turn(input_ids, output_ids):
    return {
        "role": "assistant",
        "input_ids": input_ids,
        "output_ids": output_ids,
        "output_token_logprobs": [[-0.1, token_id, None] for token_id in output_ids],
    }


def _tool_turn(content="tool output"):
    return {"role": "tool", "content": content}


def _make_target(fake_runs: _FakeRuns) -> LettaCodeTarget:
    return LettaCodeTarget(
        client=SimpleNamespace(runs=fake_runs),
        model_handle="tinker/Qwen/Qwen3.6-35B-A3B",
    )


@pytest.mark.asyncio
async def test_fetch_token_data_requests_and_processes_runs_chronologically():
    older = _run_summary("run-older", datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    newer = _run_summary("run-newer", datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
    fake_runs = _FakeRuns(
        # Simulate a server/client returning newest-first despite the requested
        # order. _fetch_token_data should still sort locally before retrieval.
        items=[newer, older],
        runs_by_id={
            "run-older": _run_with_turns(_assistant_turn([10, 11], [12]), _tool_turn("first tool")),
            "run-newer": _run_with_turns(_assistant_turn([10, 11, 12, 13], [14])),
        },
    )

    token_data = await _make_target(fake_runs)._fetch_token_data("agent-123")

    assert fake_runs.list_calls == [{"agent_id": "agent-123", "limit": 100, "order": "asc"}]
    assert fake_runs.retrieve_ids == ["run-older", "run-newer"]
    assert [turn.role for turn in token_data] == ["assistant", "tool", "assistant"]
    assert token_data[0].input_ids == [10, 11]
    assert token_data[0].output_ids == [12]
    assert token_data[1].content == "first tool"
    assert token_data[2].input_ids == [10, 11, 12, 13]
    assert token_data[2].output_ids == [14]


@pytest.mark.asyncio
async def test_fetch_token_data_sorts_locally_when_client_lacks_order_kwarg():
    older = _run_summary("run-older", datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    newer = _run_summary("run-newer", datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
    fake_runs = _FakeRuns(
        items=[newer, older],
        runs_by_id={
            "run-older": _run_with_turns(_assistant_turn([1], [2])),
            "run-newer": _run_with_turns(_assistant_turn([1, 2, 3], [4])),
        },
        reject_order=True,
    )

    token_data = await _make_target(fake_runs)._fetch_token_data("agent-123")

    assert fake_runs.list_calls == [
        {"agent_id": "agent-123", "limit": 100, "order": "asc"},
        {"agent_id": "agent-123", "limit": 100},
    ]
    assert fake_runs.retrieve_ids == ["run-older", "run-newer"]
    assert [turn.output_ids for turn in token_data] == [[2], [4]]
