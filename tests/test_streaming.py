"""Unit tests for letta_evals.streaming module (new partitioned layout).

The layout is documented in ``letta_evals/streaming.py``. We exercise four
configurations:

  - single-run, no model partition  → ``results.jsonl`` + ``summary.json``
  - single-run, partitioned          → ``<model>.jsonl`` + ``summary.json``
  - multi-run, partitioned           → ``<model>/run_*.jsonl`` + per-model summary
  - multi-run, no partition          → ``run_*.jsonl`` at top level
"""

import json
import tempfile
from pathlib import Path

import pytest

from letta_evals.metrics import summarize_model, summarize_runs
from letta_evals.models import (
    GradeResult,
    LettaAgentTargetSpec,
    Sample,
    SampleResult,
    SimpleGateSpec,
    Summary,
    SuiteSpec,
    Timing,
    ToolGraderSpec,
)
from letta_evals.streaming import StreamingReader, StreamingWriter
from letta_evals.types import Aggregation, GateKind, GraderKind, MetricOp, TargetKind


def _make_suite() -> SuiteSpec:
    return SuiteSpec(
        name="test-suite",
        dataset="ignored",
        target=LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_id="agent-fake-1"),
        graders={"check": ToolGraderSpec(function="exact_match", display_name="Check")},
        gate=SimpleGateSpec(
            kind=GateKind.SIMPLE, metric_key="check", aggregation=Aggregation.AVG_SCORE,
            op=MetricOp.GTE, value=0.5,
        ),
    )


def _make_samples(n: int = 2):
    return [Sample(id=i, input=f"in-{i}", ground_truth=f"gt-{i}") for i in range(n)]


def _make_result(sample_id: int, score: float) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        agent_id="agent-x",
        trajectory=[[]],
        submissions={"check": "x"},
        grades={"check": GradeResult(score=score, rationale="r")},
        timing=Timing(total=1.0, target=0.8),
    )


def _summary_for(model: str, results) -> "ModelSummary":  # noqa: F821 - returned for use by caller
    return summarize_model(
        model=model, results=results, grader_keys=["check"],
        gate=SimpleGateSpec(
            kind=GateKind.SIMPLE, metric_key="check", aggregation=Aggregation.AVG_SCORE,
            op=MetricOp.GTE, value=0.5,
        ),
    )


# ── single-run, no model partition ──


class TestSingleRunUnpartitioned:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            suite = _make_suite()
            samples = _make_samples(2)

            writer = StreamingWriter(output_path, suite, samples, models=None, num_runs=1)
            await writer.initialize()

            r1 = _make_result(0, 1.0)
            r2 = _make_result(1, 0.0)
            await writer.append_result(r1)
            await writer.append_result(r2)

            model_summary = _summary_for("default", [r1, r2])
            summary = Summary(suite="test-suite", models=[model_summary], gates_passed=True)
            await writer.write_summary(summary)

            assert (output_path / "suite.json").exists()
            assert (output_path / "results.jsonl").exists()
            assert (output_path / "summary.json").exists()

            result = await StreamingReader.to_runner_result(output_path)
            assert result.suite_spec.name == "test-suite"
            assert len(result.samples) == 2
            assert result.gates_passed is True
            assert "default" in result.runs
            assert len(result.runs["default"].results) == 2
            assert result.runs["default"].results[0].grades["check"].score == 1.0


# ── single-run, partitioned by model ──


class TestSingleRunPartitioned:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            suite = _make_suite()
            samples = _make_samples(2)
            models = ["gpt-4o", "claude-3-opus"]

            writer = StreamingWriter(output_path, suite, samples, models=models, num_runs=1)
            await writer.initialize()

            r_a = _make_result(0, 1.0)
            r_b = _make_result(1, 0.5)
            await writer.append_result(r_a, model="gpt-4o")
            await writer.append_result(r_b, model="claude-3-opus")

            summary = Summary(
                suite="test-suite",
                models=[_summary_for("gpt-4o", [r_a]), _summary_for("claude-3-opus", [r_b])],
                gates_passed=True,
            )
            await writer.write_summary(summary)

            assert (output_path / "gpt-4o.jsonl").exists()
            assert (output_path / "claude-3-opus.jsonl").exists()

            result = await StreamingReader.to_runner_result(output_path)
            assert set(result.runs.keys()) == {"gpt-4o", "claude-3-opus"}
            assert result.runs["gpt-4o"].results[0].grades["check"].score == 1.0
            assert result.runs["claude-3-opus"].results[0].grades["check"].score == 0.5


# ── multi-run, partitioned ──


class TestMultiRunPartitioned:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            suite = _make_suite()
            samples = _make_samples(1)

            writer = StreamingWriter(output_path, suite, samples, models=["m1"], num_runs=3)
            await writer.initialize()

            run_results = []
            for run in (1, 2, 3):
                r = _make_result(0, score=run / 3.0)
                await writer.append_result(r, model="m1", run=run)
                run_results.append([r])

            gate = SimpleGateSpec(
                kind=GateKind.SIMPLE, metric_key="check", aggregation=Aggregation.AVG_SCORE,
                op=MetricOp.GTE, value=0.5,
            )
            model_summary = summarize_runs(
                model="m1", per_run_results=run_results, grader_keys=["check"], gate=gate
            )
            await writer.write_model_summary(model_summary)

            top = Summary(
                suite="test-suite", models=[model_summary], gates_passed=True, runs_passed=2,
            )
            await writer.write_summary(top)

            assert (output_path / "m1" / "run_1.jsonl").exists()
            assert (output_path / "m1" / "run_2.jsonl").exists()
            assert (output_path / "m1" / "run_3.jsonl").exists()
            assert (output_path / "m1" / "summary.json").exists()
            assert (output_path / "summary.json").exists()

            # top-level summary should NOT include per-run breakdown
            with open(output_path / "summary.json") as f:
                top_obj = json.load(f)
            assert "runs" not in top_obj["models"][0]

            # per-model summary SHOULD include runs
            with open(output_path / "m1" / "summary.json") as f:
                detail_obj = json.load(f)
            assert "runs" in detail_obj and len(detail_obj["runs"]) == 3

            result = await StreamingReader.to_runner_result(output_path)
            assert result.runs["m1"].runs is not None
            assert len(result.runs["m1"].runs) == 3


# ── multi-run, no partition ──


class TestMultiRunUnpartitioned:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            suite = _make_suite()
            samples = _make_samples(1)

            writer = StreamingWriter(output_path, suite, samples, models=None, num_runs=2)
            await writer.initialize()

            r1 = _make_result(0, 1.0)
            r2 = _make_result(0, 0.0)
            await writer.append_result(r1, run=1)
            await writer.append_result(r2, run=2)

            gate = SimpleGateSpec(
                kind=GateKind.SIMPLE, metric_key="check", aggregation=Aggregation.AVG_SCORE,
                op=MetricOp.GTE, value=0.5,
            )
            model_summary = summarize_runs(
                model="default", per_run_results=[[r1], [r2]], grader_keys=["check"], gate=gate
            )
            top = Summary(
                suite="test-suite", models=[model_summary], gates_passed=False, runs_passed=1
            )
            await writer.write_summary(top)

            assert (output_path / "run_1.jsonl").exists()
            assert (output_path / "run_2.jsonl").exists()

            result = await StreamingReader.to_runner_result(output_path)
            assert result.runs["default"].runs is not None
            assert len(result.runs["default"].runs) == 2
            assert result.gates_passed is False


# ── error paths ──


class TestReaderErrors:
    @pytest.mark.asyncio
    async def test_missing_suite_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            (output_path / "summary.json").write_text(
                '{"suite":"x","models":[],"gates_passed":false}'
            )
            with pytest.raises(FileNotFoundError):
                await StreamingReader.to_runner_result(output_path)

    @pytest.mark.asyncio
    async def test_missing_summary_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            (output_path / "suite.json").write_text(
                json.dumps({"suite": "x", "config": {}, "samples": []})
            )
            with pytest.raises(FileNotFoundError):
                await StreamingReader.to_runner_result(output_path)
