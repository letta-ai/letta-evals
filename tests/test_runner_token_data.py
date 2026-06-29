"""Tests for ``Runner.run_sample(return_token_data=True)`` and the
``SampleResult.{token_data, agent_state}`` plumbing.

Three contracts:

  T0.1: ``Runner.run_sample(return_token_data=True)`` fetches token data from
        server-side trace metadata after the target returns an ``agent_id`` and
        surfaces it on ``SampleResult.token_data``.
  T0.2: When at least one grader's extractor needs ``agent_state``,
        ``Runner.run_sample`` retrieves it from the target and surfaces it on
        ``SampleResult.agent_state`` so callers do not have to re-fetch it.
  T0.3: ``SampleResult`` validates with the new optional ``token_data`` and
        ``agent_state`` fields — both as ``None`` (default) and as fully
        populated objects.

These tests bypass ``Runner.__init__`` (which builds an AsyncLetta client and
loads model handles) and inject only the attributes
``run_sample`` actually reads. The goal is to lock down the wire-up between
``Runner`` and the new fields, not to re-test target/grader internals.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from letta_evals.graders.base import Grader
from letta_evals.models import (
    AgentState,
    GradeResult,
    LettaMessageUnion,
    Sample,
    SampleResult,
    TargetResult,
    Timing,
    TurnTokenData,
)
from letta_evals.runner import Runner

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeGrader(Grader):
    """Minimal Grader that records calls and returns a fixed score.

    ``requires_agent_state`` is a property on ``Grader``; we override it via
    instance dict so the parent's property short-circuits in our favour.
    """

    def __init__(self, *, requires_agent_state: bool = False, score: float = 1.0):
        self._requires_agent_state = requires_agent_state
        self._score = score
        self.last_agent_state: Optional[AgentState] = None
        self.call_count = 0

    @property
    def requires_agent_state(self) -> bool:  # type: ignore[override]
        return self._requires_agent_state

    async def grade(
        self,
        sample: Sample,
        trajectory: List[List[LettaMessageUnion]],
        agent_state: Optional[AgentState] = None,
    ) -> Tuple[GradeResult, str]:
        self.call_count += 1
        self.last_agent_state = agent_state
        return GradeResult(score=self._score, rationale="fake"), "submission"


def _make_agent_state() -> AgentState:
    """Build a minimal AgentState — only the id is exercised by these tests."""
    # AgentState is re-exported from letta_evals.models. Use the constructor
    # path that requires the fewest mandatory fields by mocking it: the test
    # only checks identity, never field values.
    return MagicMock(spec=AgentState, id="agent-fake-1")


def _make_target_result() -> TargetResult:
    """Build the minimal TargetResult returned by LettaCodeTarget."""
    return TargetResult(
        agent_id="agent-fake-1",
        model_handle="fake/model",
        agent_usage=None,
    )


def _make_runner(grader: _FakeGrader, target: MagicMock) -> Runner:
    """Construct a Runner with only the attributes ``run_sample`` reads.

    Skips __init__ (which would build an AsyncLetta client + load model
    configs from disk) by going through ``Runner.__new__``.
    """
    runner = Runner.__new__(Runner)
    runner.suite = MagicMock()
    runner.suite.cleanup = False  # _should_cleanup_agent → False, no client.delete
    runner.suite.sandbox = None  # _run_sample sandbox dispatch → in-process path
    runner.suite.target = MagicMock()
    runner.suite.target.kind = MagicMock()
    runner.client = MagicMock()
    runner.graders = {"acc": grader}
    runner.results = []
    runner.max_concurrent = 1
    runner.semaphore = anyio.Semaphore(1)
    runner.progress_callback = None
    runner.model_handles = [None]
    runner.cached_results = None
    runner._cached_trajectories = {}
    runner.stream_writer = None
    runner.output_path = None
    runner.project_id = None

    # Patch _create_letta_code_target so _get_or_run_target_trace returns our canned target.
    runner._create_letta_code_target = lambda model_handle=None: target  # type: ignore[method-assign]
    return runner


# ---------------------------------------------------------------------------
# T0.3 — SampleResult schema accepts both new fields
# ---------------------------------------------------------------------------


def test_t0_3_sample_result_validates_with_new_fields_none():
    """T0.3a: callers that don't pass these fields → both are None."""
    result = SampleResult(
        sample_id=0,
        trajectory=[[]],
        submissions={"acc": "x"},
        grades={"acc": GradeResult(score=1.0)},
        timing=Timing(total=0.0, target=0.0),
    )
    assert result.token_data is None
    assert result.agent_state is None


def test_t0_3_sample_result_validates_with_token_data_populated():
    """T0.3b: callers passing populated token_data → schema accepts it."""
    token_data = [
        TurnTokenData(
            role="assistant_message",
            content="hello",
            input_ids=[100, 101],
            output_ids=[1, 2, 3],
            output_token_logprobs=[(-0.1,), (-0.2,), (-0.3,)],
        )
    ]
    result = SampleResult(
        sample_id=0,
        trajectory=[[]],
        submissions={"acc": "x"},
        grades={"acc": GradeResult(score=1.0)},
        timing=Timing(total=0.0, target=0.0),
        token_data=token_data,
    )
    assert result.token_data == token_data
    assert result.token_data[0].input_ids == [100, 101]
    assert result.token_data[0].output_ids == [1, 2, 3]


def test_t0_3_sample_result_round_trips_through_pydantic():
    """T0.3c: the new fields survive model_dump → model_validate.

    Anyone caching SampleResults (e.g. for re-grading) needs both fields
    to round-trip cleanly. agent_state is intentionally omitted here because
    it's a complex Letta SDK type whose serialization is its own concern;
    token_data is the new addition this PR introduces.
    """
    original = SampleResult(
        sample_id=0,
        trajectory=[[]],
        submissions={"acc": "x"},
        grades={"acc": GradeResult(score=1.0)},
        timing=Timing(total=0.0, target=0.0),
        token_data=[
            TurnTokenData(
                role="tool_call_message",
                content="t",
                input_ids=[7, 8],
                output_ids=[9, 10],
            )
        ],
    )
    dumped = original.model_dump()
    revived = SampleResult.model_validate(dumped)
    assert revived.token_data is not None
    assert revived.token_data[0].input_ids == [7, 8]
    assert revived.token_data[0].output_ids == [9, 10]


# ---------------------------------------------------------------------------
# T0.1 — return_token_data plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t0_1_return_token_data_fetches_after_target_run(monkeypatch):
    """run_sample(return_token_data=True) fetches token data from agent_id in Runner."""
    grader = _FakeGrader(requires_agent_state=False)
    token_data = [TurnTokenData(role="assistant_message", output_ids=[100])]
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result())
    fetch_trajectory = AsyncMock(return_value=[[]])
    fetch_token_data = AsyncMock(return_value=token_data)
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", fetch_trajectory)
    monkeypatch.setattr("letta_evals.runner.fetch_token_data", fetch_token_data)

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None, return_token_data=True)

    target.run.assert_awaited_once()
    kwargs = target.run.await_args.kwargs
    assert "return_token_data" not in kwargs
    fetch_trajectory.assert_awaited_once_with(runner.client, "agent-fake-1")
    fetch_token_data.assert_awaited_once_with(runner.client, "agent-fake-1")
    assert result.token_data == token_data


@pytest.mark.asyncio
async def test_t0_1_token_data_surfaces_on_sample_result(monkeypatch):
    """Fetched token data appears on SampleResult.token_data."""
    grader = _FakeGrader(requires_agent_state=False)
    token_data = [
        TurnTokenData(
            role="assistant_message",
            content="hello",
            output_ids=[100, 101, 102],
            output_token_logprobs=[(-0.1,)] * 3,
        ),
    ]
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result())
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))
    monkeypatch.setattr("letta_evals.runner.fetch_token_data", AsyncMock(return_value=token_data))

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None, return_token_data=True)

    assert result.token_data == token_data, (
        "T0.1: SampleResult.token_data must equal the list fetched by Runner. "
        "Pydantic re-validates the list during model construction so identity is "
        "not preserved, but value-equality must hold exactly."
    )
    assert result.token_data[0].output_ids == [100, 101, 102]


@pytest.mark.asyncio
async def test_t0_1_default_keeps_token_data_none(monkeypatch):
    """Default eval-mode call (return_token_data not passed) → field stays None.

    This is the back-compat guarantee: existing eval callers don't get a
    surprise field populated, and they don't pay the token-fetch cost.
    """
    grader = _FakeGrader(requires_agent_state=False)
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result())
    fetch_token_data = AsyncMock()
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))
    monkeypatch.setattr("letta_evals.runner.fetch_token_data", fetch_token_data)

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None)

    assert "return_token_data" not in target.run.await_args.kwargs
    fetch_token_data.assert_not_awaited()
    assert result.token_data is None


# ---------------------------------------------------------------------------
# T0.2 — agent_state plumbing (gated by graders)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t0_2_agent_state_requested_when_grader_needs_it(monkeypatch):
    """Grader with requires_agent_state=True → Runner fetches agent_state."""
    grader = _FakeGrader(requires_agent_state=True)
    agent_state = _make_agent_state()
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result())
    fetch_agent_state = AsyncMock(return_value=agent_state)
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))
    monkeypatch.setattr("letta_evals.runner.fetch_agent_state", fetch_agent_state)

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None)

    target.run.assert_awaited_once()
    kwargs = target.run.await_args.kwargs
    assert "retrieve_agent_state" not in kwargs
    fetch_agent_state.assert_awaited_once_with(runner.client, "agent-fake-1")
    # And it surfaces on SampleResult so callers don't re-fetch it.
    assert result.agent_state is agent_state
    # Also: the grader actually received the agent_state in its grade() call.
    assert grader.last_agent_state is agent_state


@pytest.mark.asyncio
async def test_t0_2_agent_state_skipped_when_no_grader_needs_it(monkeypatch):
    """Grader with requires_agent_state=False → Runner skips agent_state fetch.

    This is the perf-sensitive default: no grader needs agent_state →
    skip the extra Letta server round-trip the eval would otherwise pay.
    """
    grader = _FakeGrader(requires_agent_state=False)
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result())
    fetch_agent_state = AsyncMock()
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))
    monkeypatch.setattr("letta_evals.runner.fetch_agent_state", fetch_agent_state)

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None)

    assert "retrieve_agent_state" not in target.run.await_args.kwargs
    fetch_agent_state.assert_not_awaited()
    assert result.agent_state is None


@pytest.mark.asyncio
async def test_t0_2_agent_state_and_token_data_are_fetched_after_one_target_run(monkeypatch):
    """Both flags True → one target.run plus Runner-owned trace fetches.

    The whole point of the plumbing: a single ``run_sample`` call returns
    trajectory, agent_state, and token_data together, instead of forcing
    callers to re-fetch state via separate client.agents.* calls.
    """
    grader = _FakeGrader(requires_agent_state=True)
    agent_state = _make_agent_state()
    token_data = [TurnTokenData(role="assistant_message", output_ids=[1, 2])]
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result())
    fetch_agent_state = AsyncMock(return_value=agent_state)
    fetch_token_data = AsyncMock(return_value=token_data)
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))
    monkeypatch.setattr("letta_evals.runner.fetch_agent_state", fetch_agent_state)
    monkeypatch.setattr("letta_evals.runner.fetch_token_data", fetch_token_data)

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None, return_token_data=True)

    assert target.run.await_count == 1, "T0.2 + T0.1: combined RL fetch must be a single target.run call."
    kwargs = target.run.await_args.kwargs
    assert "retrieve_agent_state" not in kwargs
    assert "return_token_data" not in kwargs
    fetch_agent_state.assert_awaited_once_with(runner.client, "agent-fake-1")
    fetch_token_data.assert_awaited_once_with(runner.client, "agent-fake-1")
    assert result.agent_state is agent_state  # AgentState is not re-validated
    assert result.token_data == token_data
