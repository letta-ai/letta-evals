"""Unit tests for run-id listing and token-data fetching utils."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from letta_evals.utils import fetch_token_data, list_run_ids


class _FakeRunsAPI:
    def __init__(self):
        self.list_calls = []

    async def list(self, **kwargs):
        self.list_calls.append(kwargs)
        return SimpleNamespace(
            items=[
                SimpleNamespace(id="run-2", created_at=datetime(2026, 1, 2, tzinfo=timezone.utc)),
                SimpleNamespace(id="run-1", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc)),
                SimpleNamespace(id="run-2", created_at=datetime(2026, 1, 2, tzinfo=timezone.utc)),
            ]
        )

    async def retrieve(self, run_id: str):
        turns_by_run = {
            "run-1": [
                {"role": "assistant", "output_ids": [11], "output_token_logprobs": [-0.1]},
                {"role": "tool_return", "content": "tool-r1"},
                {"role": "assistant", "output_ids": [12], "output_token_logprobs": [-0.2]},
            ],
            "run-2": [
                {"role": "assistant", "output_ids": [21], "output_token_logprobs": [-0.3]},
            ],
        }
        return SimpleNamespace(metadata={"result": {"turns": turns_by_run[run_id]}})


class _FakeClient:
    def __init__(self):
        self.runs = _FakeRunsAPI()


@pytest.mark.asyncio
async def test_list_run_ids_and_fetch_token_data_preserves_tool_turns():
    client = _FakeClient()
    run_ids = await list_run_ids(client, "agent-1")
    token_data = await fetch_token_data(client, run_ids)

    assert run_ids == ["run-1", "run-2"]
    assert [t.role for t in token_data] == ["assistant", "tool_return", "assistant", "assistant"]
    assert [t.output_ids for t in token_data] == [[11], None, [12], [21]]
    assert token_data[1].content == "tool-r1"
