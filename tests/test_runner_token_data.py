"""Tests for ``Runner.run_sample(return_token_data=True)`` and the
``SampleResult.{token_data, agent_state}`` plumbing.

Three contracts:

  T0.1: ``Runner.run_sample(return_token_data=True)`` propagates the flag to
        the target and surfaces the resulting token data on
        ``SampleResult.token_data``.
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


def _make_target_result(
    *,
    agent_state: Optional[AgentState] = None,
    token_data: Optional[List[TurnTokenData]] = None,
) -> TargetResult:
    """Build a TargetResult with an empty trajectory.

    SampleResult/TargetResult only accept ``LettaMessageUnion`` instances
    in trajectory — building a real one drags in the Letta SDK. The fake
    grader in this file does not inspect trajectory contents, and
    ``_detect_errors`` only flags empty trajectories when score is 0.0,
    so we use an empty turn list and ensure the fake grader returns a
    non-zero score.
    """
    return TargetResult(
        trajectory=[[]],
        agent_id="agent-fake-1",
        model_handle="fake/model",
        agent_usage=None,
        agent_state=agent_state,
        run_ids=None,
        token_data=token_data,
    )


def _make_runner(grader: _FakeGrader, target: MagicMock) -> Runner:
    """Construct a Runner with only the attributes ``run_sample`` reads.

    Skips __init__ (which would build an AsyncLetta client + load model
    configs from disk) by going through ``Runner.__new__``.
    """
    runner = Runner.__new__(Runner)
    runner.suite = MagicMock()
    runner.suite.cleanup = False  # _should_cleanup_agent → False, no client.delete
    runner.suite.sandbox = None  # _run_sample sandbox gate → in-process path
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

    # Patch _create_letta_code_target so _get_or_run_trajectory returns our canned target.
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
async def test_t0_1_return_token_data_flag_propagates_to_target():
    """run_sample(return_token_data=True) calls target.run with the flag."""
    grader = _FakeGrader(requires_agent_state=False)
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result(token_data=None))

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    await runner.run_sample(sample, model_handle=None, return_token_data=True)

    target.run.assert_awaited_once()
    kwargs = target.run.await_args.kwargs
    assert kwargs.get("return_token_data") is True, (
        "T0.1: Runner.run_sample must propagate return_token_data to the target."
    )


@pytest.mark.asyncio
async def test_t0_1_token_data_surfaces_on_sample_result():
    """When the target returns token_data, it appears on SampleResult.token_data."""
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
    target.run = AsyncMock(return_value=_make_target_result(token_data=token_data))

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None, return_token_data=True)

    assert result.token_data == token_data, (
        "T0.1: SampleResult.token_data must equal the list returned by the target. "
        "Pydantic re-validates the list during model construction so identity is "
        "not preserved, but value-equality must hold exactly."
    )
    assert result.token_data[0].output_ids == [100, 101, 102]


@pytest.mark.asyncio
async def test_t0_1_default_keeps_token_data_none():
    """Default eval-mode call (return_token_data not passed) → field stays None.

    This is the back-compat guarantee: existing eval callers don't get a
    surprise field populated, and they don't pay the token-fetch cost.
    """
    grader = _FakeGrader(requires_agent_state=False)
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result(token_data=None))

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None)

    assert target.run.await_args.kwargs.get("return_token_data") is False
    assert result.token_data is None


# ---------------------------------------------------------------------------
# T0.2 — agent_state plumbing (gated by graders)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t0_2_agent_state_requested_when_grader_needs_it():
    """Grader with requires_agent_state=True → target.run gets retrieve_agent_state=True."""
    grader = _FakeGrader(requires_agent_state=True)
    agent_state = _make_agent_state()
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result(agent_state=agent_state))

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None)

    target.run.assert_awaited_once()
    kwargs = target.run.await_args.kwargs
    assert kwargs.get("retrieve_agent_state") is True, "T0.2: Runner must request agent_state when any grader needs it."
    # And it surfaces on SampleResult so callers don't re-fetch it.
    assert result.agent_state is agent_state
    # Also: the grader actually received the agent_state in its grade() call.
    assert grader.last_agent_state is agent_state


@pytest.mark.asyncio
async def test_t0_2_agent_state_skipped_when_no_grader_needs_it():
    """Grader with requires_agent_state=False → target.run gets retrieve_agent_state=False.

    This is the perf-sensitive default: no grader needs agent_state →
    skip the extra Letta server round-trip the eval would otherwise pay.
    """
    grader = _FakeGrader(requires_agent_state=False)
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result(agent_state=None))

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None)

    kwargs = target.run.await_args.kwargs
    assert kwargs.get("retrieve_agent_state") is False
    assert result.agent_state is None


@pytest.mark.asyncio
async def test_t0_2_agent_state_and_token_data_combine_in_one_call():
    """Both flags True → one target.run call carrying both.

    The whole point of the plumbing: a single ``run_sample`` call returns
    trajectory, agent_state, and token_data together, instead of forcing
    callers to re-fetch state via separate client.agents.* calls.
    """
    grader = _FakeGrader(requires_agent_state=True)
    agent_state = _make_agent_state()
    token_data = [TurnTokenData(role="assistant_message", output_ids=[1, 2])]
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result(agent_state=agent_state, token_data=token_data))

    runner = _make_runner(grader, target)
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None, return_token_data=True)

    assert target.run.await_count == 1, "T0.2 + T0.1: combined RL fetch must be a single target.run call."
    kwargs = target.run.await_args.kwargs
    assert kwargs.get("retrieve_agent_state") is True
    assert kwargs.get("return_token_data") is True
    assert result.agent_state is agent_state  # AgentState is not re-validated
    assert result.token_data == token_data
