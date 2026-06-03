"""Unit tests for LettaCodeTarget token-data reconstruction."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from letta_evals.targets.letta_code_target import LettaCodeTarget


def _run_summary(run_id: str) -> SimpleNamespace:
    return SimpleNamespace(id=run_id)


def _run_with_turn(token_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        metadata={
            "result": {
                "turns": [
                    {
                        "role": "assistant",
                        "input_ids": [token_id - 1],
                        "output_ids": [token_id],
                        "output_token_logprobs": [-0.1],
                    }
                ]
            }
        }
    )


@pytest.mark.asyncio
async def test_fetch_token_data_requests_runs_chronologically(tmp_path):
    """Ask the runs API for oldest-first order before reconstructing token_data."""
    older = _run_summary("run-older")
    newer = _run_summary("run-newer")

    client = MagicMock()
    client.runs.list = AsyncMock(return_value=SimpleNamespace(items=[older, newer]))
    client.runs.retrieve = AsyncMock(
        side_effect=lambda *, run_id: {
            "run-older": _run_with_turn(100),
            "run-newer": _run_with_turn(200),
        }[run_id]
    )

    target = LettaCodeTarget(client=client, model_handle="fake/model", base_dir=tmp_path)

    token_data = await target._fetch_token_data("agent-123")

    client.runs.list.assert_awaited_once_with(
        agent_id="agent-123",
        limit=100,
        order_by="created_at",
        order="asc",
    )
    assert [turn.output_ids for turn in token_data] == [[100], [200]]
    assert [call.kwargs["run_id"] for call in client.runs.retrieve.await_args_list] == ["run-older", "run-newer"]
