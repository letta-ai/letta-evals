"""Tests for ``Runner.run_sample(return_token_data=True)`` and the
``SampleResult.{token_data, context}`` plumbing.

Three contracts:

  T0.1: ``Runner.run_sample(return_token_data=True)`` fetches token data from
        server-side trace metadata after the target returns an ``agent_id`` and
        surfaces it on ``SampleResult.token_data``.
  T0.2: The target's per-rollout ``context`` reaches graders and reward
        composers and is persisted on ``SampleResult.context``, including when
        the trace comes from cached results.
  T0.3: ``SampleResult`` validates with the optional ``token_data`` and
        ``context`` fields — both as ``None`` (default) and as fully
        populated objects.

These tests bypass ``Runner.__init__`` (which builds an AsyncLetta client and
loads model handles) and inject only the attributes
``run_sample`` actually reads. The goal is to lock down the wire-up between
``Runner`` and the new fields, not to re-test target/grader internals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest

from letta_evals.graders.base import Grader
from letta_evals.models import (
    GradeResult,
    LettaMessageUnion,
    RewardOutput,
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
    """Minimal Grader that records calls and returns a fixed score."""

    def __init__(self, *, score: float = 1.0):
        self._score = score
        self.last_context: Optional[Dict[str, Any]] = None
        self.call_count = 0

    async def grade(
        self,
        sample: Sample,
        trajectory: List[List[LettaMessageUnion]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GradeResult, str]:
        self.call_count += 1
        self.last_context = context
        return GradeResult(score=self._score, rationale="fake"), "submission"


def _make_target_result(context: Optional[Dict[str, Any]] = None) -> TargetResult:
    """Build the minimal TargetResult returned by LettaCodeTarget."""
    return TargetResult(
        agent_id="agent-fake-1",
        model_handle="fake/model",
        agent_usage=None,
        context=context,
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
    runner.suite.target.backend = "api"
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
    runner._pricing_overrides = None
    runner._model_identities = {}

    # Patch _create_letta_code_target so _run_target_trace runs our canned target.
    runner._create_letta_code_target = lambda model_handle=None, local_backend_dir=None: target  # type: ignore[method-assign]
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
    assert result.context is None


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
    to round-trip cleanly.
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
        context={"memory_dir": "/ws", "seed": 3},
    )
    dumped = original.model_dump()
    revived = SampleResult.model_validate(dumped)
    assert revived.context == {"memory_dir": "/ws", "seed": 3}
    assert revived.token_data is not None
    assert revived.token_data[0].input_ids == [7, 8]
    assert revived.token_data[0].output_ids == [9, 10]


# ---------------------------------------------------------------------------
# T0.1 — return_token_data plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t0_1_return_token_data_fetches_after_target_run(monkeypatch):
    """run_sample(return_token_data=True) fetches token data from agent_id in Runner."""
    grader = _FakeGrader()
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
    grader = _FakeGrader()
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
    grader = _FakeGrader()
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
# T0.2 — rollout context plumbing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_t0_2_target_context_reaches_graders_reward_and_result(monkeypatch):
    """Context from the target flows to graders, the reward composer, and SampleResult."""
    grader = _FakeGrader()
    context = {"memory_dir": "/ws/agent-fake-1"}
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result(context=context))
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))

    runner = _make_runner(grader, target)
    reward_contexts = []

    async def reward_composer(ctx):
        reward_contexts.append(ctx.context)
        return RewardOutput(score=1.0)

    runner.reward_composer = reward_composer
    sample = Sample(id=0, input="hi", ground_truth="ok")

    result = await runner.run_sample(sample, model_handle=None)

    assert grader.last_context == context
    assert reward_contexts == [context]
    assert result.context == context
    assert result.error is None


@pytest.mark.asyncio
async def test_t0_2_missing_context_gives_reward_composer_empty_dict(monkeypatch):
    """No factory/workspace → result.context stays None; composers still get a dict."""
    grader = _FakeGrader()
    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result())
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))

    runner = _make_runner(grader, target)
    reward_contexts = []

    async def reward_composer(ctx):
        reward_contexts.append(ctx.context)
        return RewardOutput(score=1.0)

    runner.reward_composer = reward_composer
    result = await runner.run_sample(Sample(id=0, input="hi", ground_truth="ok"), model_handle=None)

    assert grader.last_context is None
    assert reward_contexts == [{}]
    assert result.context is None


@pytest.mark.asyncio
async def test_t0_2_cached_result_context_is_regraded_without_target(monkeypatch):
    """Re-grading from cache reuses the persisted context; the factory/target does not rerun."""
    grader = _FakeGrader()
    target = MagicMock()
    target.run = AsyncMock()
    fetch_trajectory = AsyncMock()
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", fetch_trajectory)

    runner = _make_runner(grader, target)
    cached = SampleResult(
        sample_id=0,
        agent_id="agent-cached",
        model_handle="fake/model",
        trajectory=[[]],
        submissions={"acc": "x"},
        grades={"acc": GradeResult(score=0.0)},
        timing=Timing(total=0.0, target=0.0),
        context={"memory_dir": "/ws/agent-cached"},
    )
    runner.cached_results = MagicMock()
    runner._cached_trajectories = {0: {"fake/model": cached}}

    result = await runner.run_sample(Sample(id=0, input="hi", ground_truth="ok"), model_handle=None)

    target.run.assert_not_awaited()
    fetch_trajectory.assert_not_awaited()
    assert grader.last_context == {"memory_dir": "/ws/agent-cached"}
    assert result.context == {"memory_dir": "/ws/agent-cached"}


@pytest.mark.asyncio
async def test_t0_2_grading_error_still_persists_context(monkeypatch):
    """A grading failure keeps context on the result so a cached re-grade after the fix has it."""

    class _RaisingGrader(_FakeGrader):
        async def grade(self, sample, trajectory, context=None):
            raise RuntimeError("extractor bug")

    target = MagicMock()
    target.run = AsyncMock(return_value=_make_target_result(context={"memory_dir": "/ws"}))
    monkeypatch.setattr("letta_evals.runner.fetch_trajectory", AsyncMock(return_value=[[]]))

    runner = _make_runner(_RaisingGrader(), target)
    result = await runner.run_sample(Sample(id=0, input="hi", ground_truth="ok"), model_handle=None)

    assert result.error is not None
    assert result.error.message == "extractor bug"
    assert result.context == {"memory_dir": "/ws"}
