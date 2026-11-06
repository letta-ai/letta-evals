"""Runner on backend: local — the trace comes from the target, nothing is read
from or deleted on a server, and cleanup removes the sample's local store and
working directory."""

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from letta_client.types.agents import AssistantMessage

from letta_evals.models import RewardOutput, Sample, TargetTrace
from letta_evals.targets.errors import TargetError
from tests.test_runner_token_data import _FakeGrader, _make_runner

MESSAGE = AssistantMessage(
    id="assistant-1", date=datetime.now(timezone.utc), content="done", message_type="assistant_message"
)
USAGE = [{"message_type": "usage_statistics", "prompt_tokens": 5, "completion_tokens": 2}]


def _local_runner(monkeypatch, target: MagicMock, *, cleanup: bool = False):
    for name in ("fetch_trajectory", "fetch_token_data"):
        monkeypatch.setattr(f"letta_evals.runner.{name}", AsyncMock(side_effect=AssertionError(f"{name} called")))
    runner = _make_runner(_FakeGrader(), target)
    runner.suite.target.backend = "local"
    runner.suite.cleanup = cleanup
    runner.client = MagicMock()
    runner.client.agents.retrieve = AsyncMock()
    runner.client.agents.delete = AsyncMock()
    runner.stores = []
    runner.cwds = []

    def create_target(model_handle, default_cwd, local_backend_dir=None):
        assert default_cwd.is_dir() and not any(default_cwd.iterdir())
        assert local_backend_dir is not None and local_backend_dir.is_dir()
        assert default_cwd != local_backend_dir
        runner.cwds.append(default_cwd)
        runner.stores.append(local_backend_dir)
        return target

    runner._create_letta_code_target = create_target

    async def reward_composer(ctx):
        return RewardOutput(score=1.0)

    runner.reward_composer = reward_composer
    return runner


@pytest.mark.asyncio
async def test_trace_comes_from_the_target(monkeypatch):
    target = MagicMock()
    target.run = AsyncMock(
        return_value=TargetTrace(
            agent_id="agent-local-1",
            model_handle="llama.cpp/local-model",
            agent_usage=USAGE,
            trajectory=[[MESSAGE]],
        )
    )
    runner = _local_runner(monkeypatch, target)

    result = await runner.run_sample(Sample(id=0, input="hi"), model_handle=None, return_token_data=True)

    [store] = runner.stores
    [cwd] = runner.cwds
    assert result.error is None
    assert result.trajectory == [[MESSAGE]]
    assert result.token_data is None
    assert result.local_backend_dir == store
    runner.client.agents.retrieve.assert_not_awaited()
    assert store.is_dir()
    assert cwd.is_dir()


@pytest.mark.asyncio
async def test_failed_target_keeps_its_partial_trajectory(monkeypatch):
    target = MagicMock()
    target.run = AsyncMock(side_effect=TargetError("boom", agent_id="agent-local-1", trajectory=[[MESSAGE]]))
    runner = _local_runner(monkeypatch, target)

    result = await runner.run_sample(Sample(id=0, input="hi"), model_handle=None, return_token_data=True)

    [store] = runner.stores
    assert result.error is not None and result.error.message == "boom"
    assert result.trajectory == [[MESSAGE]]
    assert result.local_backend_dir == store


@pytest.mark.asyncio
async def test_cleanup_removes_the_local_store_and_working_directory(monkeypatch):
    target = MagicMock()
    target.run = AsyncMock(return_value=TargetTrace(agent_id="agent-local-1", model_handle="m", trajectory=[[MESSAGE]]))
    runner = _local_runner(monkeypatch, target, cleanup=True)

    await runner.run_sample(Sample(id=0, input="hi"), model_handle=None)

    [store] = runner.stores
    [cwd] = runner.cwds
    assert not store.exists()
    assert not cwd.exists()
    runner.client.agents.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_sample_ids_with_path_separators_get_flat_temp_directories(monkeypatch):
    target = MagicMock()
    target.run = AsyncMock(return_value=TargetTrace(agent_id="agent-local-1", model_handle="m", trajectory=[[MESSAGE]]))
    runner = _local_runner(monkeypatch, target)

    sample_id = "collection/train/" + "x" * 300
    result = await runner.run_sample(Sample(id=sample_id, input="hi"), model_handle=None)

    [store] = runner.stores
    [cwd] = runner.cwds
    assert result.error is None
    assert store.parent == cwd.parent == Path(tempfile.gettempdir())
    assert cwd.name.startswith("letta-evals-cwd-collection_train_xxx")
