"""Unit tests for LettaCodeTarget run/token extraction helpers."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from letta_evals.targets.letta_code_target import LettaCodeTarget


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
async def test_letta_code_lists_run_ids_and_preserves_tool_turns(tmp_path):
    target = LettaCodeTarget(
        client=_FakeClient(),
        model_handle="openai/gpt-4.1",
        working_dir=tmp_path,
        sandbox=False,
    )

    run_ids = await target._list_run_ids("agent-1")
    token_data = await target._fetch_token_data(run_ids)

    assert run_ids == ["run-1", "run-2"]
    assert [t.role for t in token_data] == ["assistant", "tool_return", "assistant", "assistant"]
    assert [t.output_ids for t in token_data] == [[11], None, [12], [21]]
    assert token_data[1].content == "tool-r1"
