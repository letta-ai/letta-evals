"""Unit tests for LettaAgentTarget token-data retrieval across multi-turn runs."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from letta_client.types.agents import AssistantMessage

from letta_evals.models import Sample
from letta_evals.targets.letta_agent import LettaAgentTarget

_FAKE_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


class _FakeAgentsMessagesAPI:
    async def create(self, **kwargs):
        return object()


class _FakeAgentsAPI:
    def __init__(self, model_name: str):
        self.messages = _FakeAgentsMessagesAPI()
        self.model_name = model_name

    async def retrieve(self, agent_id: str, include: list):
        return SimpleNamespace(llm_config=SimpleNamespace(model=self.model_name))


class _FakeRunsMessagesAPI:
    async def stream(self, *args, **kwargs):
        return object()


class _FakeRunsAPI:
    def __init__(self):
        self.messages = _FakeRunsMessagesAPI()

    async def retrieve(self, run_id: str):
        """Return empty metadata so the unified fetcher falls through to messages."""
        return SimpleNamespace(metadata={})


class _FakeClient:
    def __init__(self, model_name: str = "test-model"):
        self.agents = _FakeAgentsAPI(model_name=model_name)
        self.runs = _FakeRunsAPI()


@pytest.mark.asyncio
async def test_letta_agent_returns_all_run_ids_and_token_data(monkeypatch):
    import letta_evals.targets.letta_agent as agent_module
    import letta_evals.utils as utils_module

    run_id_iter = iter(["run-1", "run-2"])

    async def _fake_consume_stream_with_resumes(*args, **kwargs):
        return next(run_id_iter), 1

    list_calls: list[tuple[str, dict | None]] = []

    async def _fake_list_all_run_messages(client, run_id, params=None):
        list_calls.append((run_id, params))

        # Trajectory fetch path (no params)
        if not params:
            return [
                AssistantMessage(
                    id=f"traj-{run_id}",
                    message_type="assistant_message",
                    date=_FAKE_DATE,
                    content=f"resp-{run_id}",
                )
            ]

        # Token fetch path (return_token_ids=true)
        if run_id == "run-1":
            return [
                SimpleNamespace(id="a1", role="assistant", output_ids=[101], output_token_logprobs=[-0.1]),
                SimpleNamespace(id="t1", role="tool_return", content="tool-output-1"),
            ]
        return [
            SimpleNamespace(id="a2", role="assistant", output_ids=[201], output_token_logprobs=[-0.2]),
        ]

    monkeypatch.setattr(agent_module, "consume_stream_with_resumes", _fake_consume_stream_with_resumes)
    # Patch list_all_run_messages in both modules (target uses it for trajectory,
    # utils uses it for the messages fallback in fetch_token_data)
    monkeypatch.setattr(agent_module, "list_all_run_messages", _fake_list_all_run_messages)
    monkeypatch.setattr(utils_module, "list_all_run_messages", _fake_list_all_run_messages)

    target = LettaAgentTarget(client=_FakeClient(), agent_id="agent-1", timeout=30)
    sample = Sample(id=0, input=["turn-1", "turn-2"])

    result = await target.run(sample, return_token_data=True)

    assert result.run_ids == ["run-1", "run-2"]
    assert len(result.trajectory) == 2
    assert [t.role for t in result.token_data] == ["assistant", "tool_return", "assistant"]
    assert result.token_data[0].output_ids == [101]
    assert result.token_data[1].content == "tool-output-1"
    assert result.token_data[2].output_ids == [201]

    trajectory_calls = [run_id for run_id, params in list_calls if params is None]
    token_calls = [run_id for run_id, params in list_calls if params is not None]
    assert trajectory_calls == ["run-1", "run-2"]
    assert token_calls == ["run-1", "run-2"]
