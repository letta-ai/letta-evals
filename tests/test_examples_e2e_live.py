from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from letta_evals.runner import run_suite
from letta_evals.visualization.factory import ProgressStyle

logger = logging.getLogger(__name__)


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

    # Aggregate across all model partitions for the summary log line.
    total = sum(ms.n_total for ms in result.summary.models)
    attempted = sum(ms.n_attempted for ms in result.summary.models)
    # Simple unweighted mean of model scores (parity with the prior log line).
    mean_score = (
        sum(ms.score for ms in result.summary.models) / len(result.summary.models)
        if result.summary.models
        else 0.0
    )
    logger.info(
        f"\n{'=' * 60}\n"
        f"Results: {attempted}/{total} attempted, "
        f"avg score: {mean_score:.2f}\n"
        f"{'=' * 60}"
    )

    assert result.gates_passed, f"Gate failed for suite: {suite_path}"
    assert total > 0
    assert output_path.exists()
