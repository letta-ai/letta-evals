"""Unit tests for execution trace token-data fetching."""

from datetime import datetime, timezone
from types import SimpleNamespace

import httpx
import pytest
from letta_client import APIConnectionError

from letta_evals.execution.trace import TokenDataFetchError, fetch_token_data


class _FakeRuns:
    def __init__(self, *, items, runs_by_id, reject_order: bool = False, retrieve_failures=None):
        self.items = items
        self.runs_by_id = runs_by_id
        self.reject_order = reject_order
        self.retrieve_failures = dict(retrieve_failures or {})
        self.list_calls = []
        self.retrieve_ids = []

    async def list(self, **kwargs):
        self.list_calls.append(kwargs)
        if self.reject_order and "order" in kwargs:
            raise TypeError("list() got an unexpected keyword argument 'order'")
        return SimpleNamespace(items=self.items)

    async def retrieve(self, *, run_id):
        self.retrieve_ids.append(run_id)
        remaining_failures = self.retrieve_failures.get(run_id, 0)
        if remaining_failures:
            self.retrieve_failures[run_id] = remaining_failures - 1
            raise APIConnectionError(request=httpx.Request("GET", f"https://example.test/v1/runs/{run_id}"))
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


def _client(fake_runs: _FakeRuns) -> SimpleNamespace:
    return SimpleNamespace(runs=fake_runs)


@pytest.mark.asyncio
async def test_fetch_token_data_requests_and_processes_runs_chronologically():
    older = _run_summary("run-older", datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    newer = _run_summary("run-newer", datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
    fake_runs = _FakeRuns(
        # Simulate a server/client returning newest-first despite the requested
        # order. fetch_token_data should still sort locally before retrieval.
        items=[newer, older],
        runs_by_id={
            "run-older": _run_with_turns(_assistant_turn([10, 11], [12]), _tool_turn("first tool")),
            "run-newer": _run_with_turns(_assistant_turn([10, 11, 12, 13], [14])),
        },
    )

    token_data = await fetch_token_data(_client(fake_runs), "agent-123")

    assert fake_runs.list_calls == [{"agent_id": "agent-123", "limit": 100, "order": "asc"}]
    assert fake_runs.retrieve_ids == ["run-older", "run-newer"]
    assert [turn.role for turn in token_data] == ["assistant", "tool", "assistant"]
    assert token_data[0].input_ids == [10, 11]
    assert token_data[0].output_ids == [12]
    assert token_data[1].content == "first tool"
    assert token_data[2].input_ids == [10, 11, 12, 13]
    assert token_data[2].output_ids == [14]


@pytest.mark.asyncio
async def test_fetch_token_data_reads_all_run_pages():
    first = _run_summary("run-first", datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    second = _run_summary("run-second", datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))

    class _Page:
        def __init__(self, items, next_page=None):
            self.items = items
            self.next_page = next_page

        def has_next_page(self):
            return self.next_page is not None

        async def get_next_page(self):
            return self.next_page

    fake_runs = _FakeRuns(
        items=[],
        runs_by_id={
            "run-first": _run_with_turns(_assistant_turn([1], [2])),
            "run-second": _run_with_turns(_assistant_turn([1, 2, 3], [4])),
        },
    )
    second_page = _Page([second])
    first_page = _Page([first], next_page=second_page)

    async def list_pages(**kwargs):
        fake_runs.list_calls.append(kwargs)
        return first_page

    fake_runs.list = list_pages

    token_data = await fetch_token_data(_client(fake_runs), "agent-123")

    assert fake_runs.retrieve_ids == ["run-first", "run-second"]
    assert [turn.output_ids for turn in token_data] == [[2], [4]]


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

    token_data = await fetch_token_data(_client(fake_runs), "agent-123")

    assert fake_runs.list_calls == [
        {"agent_id": "agent-123", "limit": 100, "order": "asc"},
        {"agent_id": "agent-123", "limit": 100},
    ]
    assert fake_runs.retrieve_ids == ["run-older", "run-newer"]
    assert [turn.output_ids for turn in token_data] == [[2], [4]]


@pytest.mark.asyncio
async def test_fetch_token_data_retries_the_complete_snapshot_without_duplicate_prefixes():
    older = _run_summary("run-older", datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    newer = _run_summary("run-newer", datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
    fake_runs = _FakeRuns(
        items=[older, newer],
        runs_by_id={
            "run-older": _run_with_turns(_assistant_turn([1], [2])),
            "run-newer": _run_with_turns(_assistant_turn([1, 2, 3], [4])),
        },
        retrieve_failures={"run-newer": 1},
    )

    token_data = await fetch_token_data(
        _client(fake_runs),
        "agent-123",
        max_attempts=2,
        retry_base_seconds=0,
    )

    assert fake_runs.retrieve_ids == ["run-older", "run-newer", "run-older", "run-newer"]
    assert [turn.output_ids for turn in token_data] == [[2], [4]]


@pytest.mark.asyncio
async def test_fetch_token_data_raises_instead_of_returning_a_partial_prefix():
    older = _run_summary("run-older", datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    newer = _run_summary("run-newer", datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
    fake_runs = _FakeRuns(
        items=[older, newer],
        runs_by_id={
            "run-older": _run_with_turns(_assistant_turn([1], [2])),
            "run-newer": _run_with_turns(_assistant_turn([1, 2, 3], [4])),
        },
        retrieve_failures={"run-newer": 2},
    )

    with pytest.raises(TokenDataFetchError) as exc_info:
        await fetch_token_data(
            _client(fake_runs),
            "agent-123",
            max_attempts=2,
            retry_base_seconds=0,
        )

    assert exc_info.value.completed_runs == 1
    assert exc_info.value.total_runs == 2
    assert exc_info.value.failed_run_id == "run-newer"
    assert isinstance(exc_info.value.__cause__, APIConnectionError)
    assert fake_runs.retrieve_ids == ["run-older", "run-newer", "run-older", "run-newer"]


@pytest.mark.asyncio
async def test_fetch_token_data_rejects_half_written_turns_instead_of_returning_a_prefix():
    older = _run_summary("run-older", datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc))
    newer = _run_summary("run-newer", datetime(2026, 1, 1, 12, 1, tzinfo=timezone.utc))
    half_written_turn = _assistant_turn([1, 2, 3], [4, 5])
    half_written_turn["output_token_logprobs"] = [[-0.1, 4, None]]
    fake_runs = _FakeRuns(
        items=[older, newer],
        runs_by_id={
            "run-older": _run_with_turns(_assistant_turn([1], [2])),
            "run-newer": _run_with_turns(half_written_turn),
        },
    )

    with pytest.raises(TokenDataFetchError, match="half-written turn"):
        await fetch_token_data(
            _client(fake_runs),
            "agent-123",
            max_attempts=1,
            retry_base_seconds=0,
        )
