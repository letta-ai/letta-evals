"""Unit tests for letta_evals.streaming module."""

import tempfile
from pathlib import Path

import pytest

from letta_evals.models import GradeResult, Metrics, Sample, SampleResult
from letta_evals.streaming import StreamingReader, StreamingWriter


def _make_sample_result(sample_id: int = 0, score: float = 1.0) -> SampleResult:
    sample = Sample(id=sample_id, input="test input", ground_truth="test answer")
    grade = GradeResult(score=score, rationale="test rationale")
    return SampleResult(
        sample=sample,
        submission="test submission",
        trajectory=[],
        grade=grade,
        model_name="gpt-4o",
    )


def _make_metrics() -> Metrics:
    return Metrics(
        total=2,
        total_attempted=2,
        avg_score_attempted=0.75,
        avg_score_total=0.75,
        metrics={"default": 75.0},
    )


class TestStreamingWriterReader:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        """Write results with StreamingWriter, read back with StreamingReader."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            config = {"target": {"kind": "letta_agent"}, "graders": {"check": {"kind": "tool"}}}

            # Write
            writer = StreamingWriter(output_path, "test-suite", config)
            await writer.initialize()

            r1 = _make_sample_result(0, 1.0)
            r2 = _make_sample_result(1, 0.5)
            await writer.append_result(r1)
            await writer.append_result(r2)

            metrics = _make_metrics()
            await writer.write_metrics(metrics, gates_passed=True)

            # Read
            result = await StreamingReader.to_runner_result(output_path)

            assert result.suite == "test-suite"
            assert result.config == config
            assert len(result.results) == 2
            assert result.results[0].sample.id == 0
            assert result.results[0].grade.score == 1.0
            assert result.results[1].grade.score == 0.5
            assert result.metrics.total == 2
            assert result.metrics.avg_score_attempted == pytest.approx(0.75)
            assert result.gates_passed is True

    @pytest.mark.asyncio
    async def test_creates_output_directory(self):
        """StreamingWriter should create the output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir"
            writer = StreamingWriter(output_path, "test-suite", {})
            await writer.initialize()
            assert (output_path / "header.json").exists()

    @pytest.mark.asyncio
    async def test_header_written_on_initialize(self):
        """Header file should be created on initialize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = StreamingWriter(output_path, "my-suite", {"key": "val"})
            await writer.initialize()

            import json

            with open(output_path / "header.json") as f:
                header = json.load(f)
            assert header["type"] == "header"
            assert header["suite"] == "my-suite"
            assert header["config"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_reader_missing_header(self):
        """StreamingReader should raise FileNotFoundError if header.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            (output_path / "summary.json").write_text('{"type":"summary","metrics":{},"gates_passed":false}')
            (output_path / "results.jsonl").write_text("")
            with pytest.raises(FileNotFoundError):
                await StreamingReader.to_runner_result(output_path)

    @pytest.mark.asyncio
    async def test_reader_missing_summary(self):
        """StreamingReader should raise FileNotFoundError if summary.json is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            import json

            with open(output_path / "header.json", "w") as f:
                json.dump({"type": "header", "suite": "s", "config": {}}, f)
            (output_path / "results.jsonl").write_text("")
            with pytest.raises(FileNotFoundError):
                await StreamingReader.to_runner_result(output_path)

    @pytest.mark.asyncio
    async def test_reader_malformed_result_record(self):
        """StreamingReader should raise if a result record is malformed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)

            writer = StreamingWriter(output_path, "test-suite", {})
            await writer.initialize()
            await writer.write_metrics(_make_metrics(), gates_passed=True)

            # Write a malformed record (missing "result" key)
            with open(output_path / "results.jsonl", "w") as f:
                f.write('{"type": "result", "bad_key": {}}\n')

            with pytest.raises(KeyError):
                await StreamingReader.to_runner_result(output_path)

    @pytest.mark.asyncio
    async def test_gates_passed_false(self):
        """Verify gates_passed=False is preserved through round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir)
            writer = StreamingWriter(output_path, "test-suite", {})
            await writer.initialize()
            await writer.append_result(_make_sample_result(0, 0.0))
            await writer.write_metrics(_make_metrics(), gates_passed=False)

            result = await StreamingReader.to_runner_result(output_path)
            assert result.gates_passed is False
