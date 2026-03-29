"""Unit tests for Runner token-data/run-id plumbing."""

from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import pytest
from letta_client.types.agents import AssistantMessage

from letta_evals.models import GradeResult, Metrics, RunnerResult, Sample, SampleResult, SuiteSpec, TurnTokenData
from letta_evals.runner import Runner

_FAKE_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_suite() -> SuiteSpec:
    return SuiteSpec(
        name="token-plumbing-suite",
        dataset=Path("/tmp/nonexistent.jsonl"),
        target={"kind": "letta_agent", "agent_id": "agent-1"},
        graders={"check": {"kind": "tool", "function": "exact_match"}},
        gate={"kind": "simple", "metric_key": "check", "aggregation": "avg_score", "op": "gte", "value": 0.0},
    )


def _make_trajectory(content: str = "hello") -> list[list]:
    return [
        [
            AssistantMessage(
                id="msg-1",
                message_type="assistant_message",
                date=_FAKE_DATE,
                content=content,
            )
        ]
    ]


@pytest.mark.asyncio
async def test_run_sample_includes_run_ids_and_token_data():
    runner = Runner(_make_suite(), max_concurrent=1)
    sample = Sample(id=0, input="hello", ground_truth="hello")
    token_data = [TurnTokenData(role="assistant", output_ids=[101], output_token_logprobs=[-0.1])]

    async def _fake_get_or_run_trajectory(self, *_args, **_kwargs):
        return _make_trajectory("hello"), "agent-1", "model-1", None, None, ["run-1", "run-2"], token_data

    async def _fake_grade_sample(self, *_args, **_kwargs):
        grade = GradeResult(score=1.0, rationale="ok")
        return {"check": grade}, {"check": "hello"}, {}

    runner._get_or_run_trajectory = MethodType(_fake_get_or_run_trajectory, runner)
    runner._grade_sample = MethodType(_fake_grade_sample, runner)
    runner._requires_agent_state = MethodType(lambda self: False, runner)

    result = await runner.run_sample(sample)

    assert result.run_ids == ["run-1", "run-2"]
    assert result.token_data == token_data


@pytest.mark.asyncio
async def test_get_or_run_trajectory_uses_cached_run_ids_and_token_data():
    sample = Sample(id=0, input="hello", ground_truth="hello")
    token_data = [TurnTokenData(role="assistant", output_ids=[42], output_token_logprobs=[-0.2])]
    cached_sample_result = SampleResult(
        sample=sample,
        submission="hello",
        trajectory=_make_trajectory("hello"),
        agent_id="agent-1",
        grade=GradeResult(score=1.0, rationale="ok"),
        model_name="model-1",
        run_ids=["run-cached"],
        token_data=token_data,
    )
    cached_results = RunnerResult(
        suite="cached-suite",
        config={},
        results=[cached_sample_result],
        metrics=Metrics(
            total=1,
            total_attempted=1,
            avg_score_attempted=1.0,
            avg_score_total=1.0,
            metrics={},
        ),
        gates_passed=True,
    )

    runner = Runner(_make_suite(), max_concurrent=1, cached_results=cached_results)
    runner._create_target = MethodType(
        lambda self, *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("target should not be created")),
        runner,
    )

    _, _, _, _, _, run_ids, returned_token_data = await runner._get_or_run_trajectory(sample, "model-1")
    assert run_ids == ["run-cached"]
    assert returned_token_data == token_data

