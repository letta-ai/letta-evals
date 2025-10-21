from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Iterable

import pytest
import yaml

from letta_evals.models import SuiteSpec
from letta_evals.runner import run_suite
from letta_evals.types import GraderKind
from letta_evals.visualization.factory import ProgressStyle

logger = logging.getLogger(__name__)


def _find_example_suites(base: Path) -> list[Path]:
    examples_dir = base / "examples"
    patterns: Iterable[str] = ("suite.yaml", "suite.*.yaml")
    suites: list[Path] = []
    for sub in examples_dir.rglob("*"):
        if not sub.is_dir():
            continue
        for pat in patterns:
            for p in sub.glob(pat):
                suites.append(p)
    # de-duplicate, preserve order
    seen: set[Path] = set()
    result: list[Path] = []
    for s in suites:
        if s not in seen:
            result.append(s)
            seen.add(s)
    return result


def _requires_openai(suite: SuiteSpec) -> bool:
    if not suite.graders:
        return False
    return any(g.kind == GraderKind.RUBRIC for g in suite.graders.values())


@pytest.mark.asyncio
@pytest.mark.parametrize("suite_path", _find_example_suites(Path(__file__).resolve().parents[1]))
async def test_examples_run_live_and_pass_gate(suite_path: Path, tmp_path: Path, caplog) -> None:
    # require letta api key for all live runs
    if not os.getenv("LETTA_API_KEY"):
        pytest.skip("LETTA_API_KEY not set; skipping live e2e example run")

    # load to know whether we need openai
    with open(suite_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    suite = SuiteSpec.from_yaml(raw, base_dir=suite_path.parent)

    if _requires_openai(suite) and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping rubric-based example run")

    output_path = tmp_path / "stream.jsonl"

    logger.info(f"\n{'=' * 60}\nRunning suite: {suite_path.name}\n{'=' * 60}")

    # run suite live against letta cloud
    result = await run_suite(
        suite_path=suite_path,
        max_concurrent=2,
        progress_style=ProgressStyle.SIMPLE,
        cached_results_path=None,
        output_path=output_path,
        letta_api_key=os.getenv("LETTA_API_KEY"),
        letta_base_url="https://api.letta.com/",
    )

    logger.info(f"\n{'=' * 60}\nResults: {result.metrics.passed_attempts}/{result.metrics.total} passed\n{'=' * 60}")

    assert result.gates_passed, f"Gate failed for example suite: {suite_path}"
    assert result.metrics.total > 0
    assert output_path.exists()


@pytest.mark.asyncio
async def test_single_suite(request, tmp_path: Path, caplog) -> None:
    """test a single suite specified via --suite-path command line option"""
    suite_path_str = request.config.getoption("--suite-path")
    if not suite_path_str:
        pytest.skip("--suite-path not provided")

    suite_path = Path(suite_path_str).resolve()
    if not suite_path.exists():
        pytest.fail(f"Suite path does not exist: {suite_path}")

    # require letta api key for all live runs
    if not os.getenv("LETTA_API_KEY"):
        pytest.skip("LETTA_API_KEY not set; skipping live e2e example run")

    # load to know whether we need openai
    with open(suite_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    suite = SuiteSpec.from_yaml(raw, base_dir=suite_path.parent)

    if _requires_openai(suite) and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping rubric-based example run")

    output_path = tmp_path / "stream.jsonl"

    logger.info(f"\n{'=' * 60}\nRunning suite: {suite_path.name}\n{'=' * 60}")

    # run suite live against letta cloud
    result = await run_suite(
        suite_path=suite_path,
        max_concurrent=10,
        progress_style=ProgressStyle.SIMPLE,
        cached_results_path=None,
        output_path=output_path,
        letta_api_key=os.getenv("LETTA_API_KEY"),
        letta_base_url="https://api.letta.com/",
    )

    logger.info(f"\n{'=' * 60}\nResults: {result.metrics.passed_attempts}/{result.metrics.total} passed\n{'=' * 60}")

    assert result.gates_passed, f"Gate failed for suite: {suite_path}"
    assert result.metrics.total > 0
    assert output_path.exists()


@pytest.mark.asyncio
async def test_multi_run_statistics(tmp_path: Path) -> None:
    """Test multiple runs with aggregate statistics computation."""
    # require letta api key for all live runs
    if not os.getenv("LETTA_API_KEY"):
        pytest.skip("LETTA_API_KEY not set; skipping multi-run test")

    # require openai for rubric grader
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping multi-run test with rubric grader")

    # use the existing multi-run example suite
    base_dir = Path(__file__).resolve().parents[1]
    suite_path = base_dir / "examples" / "simple-rubric-grader" / "suite.multi-run.yaml"

    if not suite_path.exists():
        pytest.fail(f"Multi-run suite not found: {suite_path}")

    # load suite to verify it has num_runs configured
    with open(suite_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    suite = SuiteSpec.from_yaml(raw, base_dir=suite_path.parent)

    assert suite.num_runs is not None and suite.num_runs >= 2, "suite.multi-run.yaml should have num_runs >= 2"

    output_path = tmp_path / "multi_run_output"

    logger.info(f"\n{'=' * 60}\nRunning multi-run suite: {suite_path.name}\n{'=' * 60}")

    # run suite with multiple runs
    result = await run_suite(
        suite_path=suite_path,
        max_concurrent=2,
        progress_style=ProgressStyle.SIMPLE,
        cached_results_path=None,
        output_path=output_path,
        letta_api_key=os.getenv("LETTA_API_KEY"),
        letta_base_url="https://api.letta.com/",
    )

    logger.info(
        f"\n{'=' * 60}\nMulti-run completed: {result.run_statistics.runs_passed}/{result.run_statistics.num_runs} runs passed\n{'=' * 60}"
    )

    # verify run_statistics exists and is populated
    assert result.run_statistics is not None, "run_statistics should be present for multi-run"
    stats = result.run_statistics

    # verify number of runs matches expectation
    assert stats.num_runs == suite.num_runs, f"Expected {suite.num_runs} runs, got {stats.num_runs}"
    assert stats.num_runs >= 2, "Should have at least 2 runs"

    # verify individual run metrics are collected
    assert len(stats.individual_run_metrics) == stats.num_runs, "Should have metrics for each run"
    for run_metrics in stats.individual_run_metrics:
        assert run_metrics.total > 0, "Each run should have evaluated samples"

    # verify aggregate statistics are computed
    assert stats.mean_avg_score_attempted >= 0.0, "Mean avg_score_attempted should be non-negative"
    assert stats.mean_avg_score_total >= 0.0, "Mean avg_score_total should be non-negative"

    # verify std dev is computed (should be 0.0 if all runs identical, which is unlikely with rubric grader)
    assert stats.std_avg_score_attempted >= 0.0, "Std dev should be non-negative"
    assert stats.std_avg_score_total >= 0.0, "Std dev should be non-negative"

    # verify per-metric statistics
    assert len(stats.mean_scores) > 0, "Should have per-metric mean scores"
    assert len(stats.std_scores) > 0, "Should have per-metric std scores"

    # verify output directory structure when output_path is provided
    if output_path:
        assert output_path.exists(), "Output directory should exist"

        # check individual run directories
        for i in range(1, stats.num_runs + 1):
            run_dir = output_path / f"run_{i}"
            assert run_dir.exists(), f"run_{i} directory should exist"
            assert (run_dir / "header.json").exists(), f"run_{i}/header.json should exist"
            assert (run_dir / "results.jsonl").exists(), f"run_{i}/results.jsonl should exist"
            assert (run_dir / "summary.json").exists(), f"run_{i}/summary.json should exist"

        # check aggregate stats file
        aggregate_file = output_path / "aggregate_stats.json"
        assert aggregate_file.exists(), "aggregate_stats.json should exist"

        # verify aggregate_stats.json can be loaded and has correct structure
        with open(aggregate_file, "r", encoding="utf-8") as f:
            aggregate_data = json.load(f)
            assert aggregate_data["num_runs"] == stats.num_runs
            assert "mean_avg_score_attempted" in aggregate_data
            assert "std_avg_score_attempted" in aggregate_data
            assert "mean_scores" in aggregate_data
            assert "std_scores" in aggregate_data
            assert "individual_run_metrics" in aggregate_data
            assert len(aggregate_data["individual_run_metrics"]) == stats.num_runs

    # verify gate pass behavior
    assert stats.runs_passed <= stats.num_runs, "Runs passed should not exceed total runs"
    assert stats.runs_passed >= 0, "Runs passed should be non-negative"
