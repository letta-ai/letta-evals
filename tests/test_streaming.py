"""Unit tests for letta_evals.streaming.StreamingWriter (always-partitioned layout)."""

import json
import tempfile
from pathlib import Path

import pytest

from letta_evals.metrics import summarize_model, summarize_runs
from letta_evals.models import (
    GradeResult,
    LettaCodeTargetSpec,
    MetricRewardSpec,
    RewardOutput,
    Sample,
    SampleResult,
    SuiteSpec,
    Summary,
    Timing,
    ToolGraderSpec,
)
from letta_evals.streaming import StreamingWriter
from letta_evals.types import RewardKind


def _make_suite() -> SuiteSpec:
    return SuiteSpec(
        name="test-suite",
        dataset="ignored",
        target=LettaCodeTargetSpec(model_handles=["openai/gpt-4.1-mini"]),
        graders={"check": ToolGraderSpec(function="exact_match", display_name="Check")},
        reward=MetricRewardSpec(kind=RewardKind.METRIC, metric_key="check"),
    )


def _make_samples(n: int = 2):
    return [Sample(id=i, input=f"in-{i}", ground_truth=f"gt-{i}") for i in range(n)]


def _make_result(sample_id, score: float) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        agent_id="agent-x",
        trajectory=[[]],
        submissions={"check": "x"},
        grades={"check": GradeResult(score=score, rationale="r")},
        reward=RewardOutput(score=score),
        timing=Timing(total=1.0, target=0.8),
    )


def _read_results(path: Path):
    return [SampleResult.model_validate_json(line) for line in path.read_text().splitlines() if line]


def _summary_for(model: str, results):
    return summarize_model(
        model=model,
        results=results,
        grader_keys=["check"],
    )


# ── construction guards ──


class TestWriterConstruction:
    def test_empty_models_rejected(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError):
                StreamingWriter(Path(tmpdir), _make_suite(), _make_samples(), models=[])


# ── single-run ──


class TestSingleRun:
    @pytest.mark.asyncio
    async def test_single_model_round_trip(self):
        """Even a 'single-model' suite uses the partitioned layout."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = StreamingWriter(
                output_path,
                _make_suite(),
                _make_samples(2),
                models=["default"],
                num_runs=1,
            )
            await writer.initialize()

            r1 = _make_result(0, 1.0)
            r2 = _make_result(1, 0.0)
            await writer.append_result(r1, model="default")
            await writer.append_result(r2, model="default")

            summary = Summary(
                suite="test-suite",
                models=[_summary_for("default", [r1, r2])],
            )
            await writer.write_summary(summary)

            assert (output_path / "suite.json").exists()
            assert (output_path / "default.jsonl").exists()
            assert (output_path / "summary.json").exists()
            # No legacy results.jsonl
            assert not (output_path / "results.jsonl").exists()

            suite_obj = json.loads((output_path / "suite.json").read_text())
            assert suite_obj["config"]["name"] == "test-suite"
            assert len(suite_obj["samples"]) == 2
            assert _read_results(output_path / "default.jsonl")[0].grades["check"].score == 1.0

    @pytest.mark.asyncio
    async def test_string_sample_id_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            samples = [Sample(id="sample-a", input="in-a", ground_truth="gt-a")]
            writer = StreamingWriter(
                output_path,
                _make_suite(),
                samples,
                models=["default"],
                num_runs=1,
            )
            await writer.initialize()

            r = _make_result("sample-a", 1.0)
            await writer.append_result(r, model="default")
            await writer.write_summary(
                Summary(
                    suite="test-suite",
                    models=[_summary_for("default", [r])],
                )
            )

            suite_obj = json.loads((output_path / "suite.json").read_text())
            assert suite_obj["samples"][0]["id"] == "sample-a"
            assert _read_results(output_path / "default.jsonl")[0].sample_id == "sample-a"

    @pytest.mark.asyncio
    async def test_multiple_models_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = StreamingWriter(
                output_path,
                _make_suite(),
                _make_samples(2),
                models=["gpt-4o", "claude-3-opus"],
                num_runs=1,
            )
            await writer.initialize()

            r_a = _make_result(0, 1.0)
            r_b = _make_result(1, 0.5)
            await writer.append_result(r_a, model="gpt-4o")
            await writer.append_result(r_b, model="claude-3-opus")

            summary = Summary(
                suite="test-suite",
                models=[
                    _summary_for("gpt-4o", [r_a]),
                    _summary_for("claude-3-opus", [r_b]),
                ],
            )
            await writer.write_summary(summary)

            assert (output_path / "gpt-4o.jsonl").exists()
            assert (output_path / "claude-3-opus.jsonl").exists()

            assert _read_results(output_path / "gpt-4o.jsonl")[0].grades["check"].score == 1.0
            assert _read_results(output_path / "claude-3-opus.jsonl")[0].grades["check"].score == 0.5

    @pytest.mark.asyncio
    async def test_safe_segment_for_slash_in_model(self):
        """Model handles with slashes are sanitised to filenames."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = StreamingWriter(
                output_path,
                _make_suite(),
                _make_samples(1),
                models=["openai/gpt-4"],
                num_runs=1,
            )
            await writer.initialize()

            r = _make_result(0, 1.0)
            await writer.append_result(r, model="openai/gpt-4")

            summary = Summary(
                suite="test-suite",
                models=[_summary_for("openai/gpt-4", [r])],
            )
            await writer.write_summary(summary)

            assert (output_path / "openai-gpt-4.jsonl").exists()


# ── multi-run ──


class TestMultiRun:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = StreamingWriter(
                output_path,
                _make_suite(),
                _make_samples(1),
                models=["m1"],
                num_runs=3,
            )
            await writer.initialize()

            run_results = []
            for run in (1, 2, 3):
                r = _make_result(0, score=run / 3.0)
                await writer.append_result(r, model="m1", run=run)
                run_results.append([r])

            model_summary = summarize_runs(
                model="m1",
                per_run_results=run_results,
                grader_keys=["check"],
            )
            await writer.write_model_summary(model_summary)

            top = Summary(
                suite="test-suite",
                models=[model_summary],
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

            for run in (1, 2, 3):
                [r] = _read_results(output_path / "m1" / f"run_{run}.jsonl")
                assert r.grades["check"].score == run / 3.0
