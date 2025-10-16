from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, Optional

import pytest
import yaml

from letta_evals.models import SuiteSpec
from letta_evals.runner import run_suite
from letta_evals.types import GraderKind

logger = logging.getLogger(__name__)


class SimpleTestProgressCallback:
    """simple logging-based progress callback for tests"""

    def __init__(self):
        self.current_sample = None

    async def sample_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        self.current_sample = sample_id
        model_info = f" [{model_name}]" if model_name else ""
        logger.info(f"▶ sample {sample_id}{model_info} started")

    async def agent_loading(self, sample_id: int, model_name: Optional[str] = None, from_cache: bool = False) -> None:
        cache_info = " (cached)" if from_cache else ""
        logger.info(f"  ↳ loading agent{cache_info}")

    async def message_sending(
        self, sample_id: int, message_num: int, total_messages: int, model_name: Optional[str] = None
    ) -> None:
        logger.info(f"  ↳ sending message {message_num}/{total_messages}")

    async def grading_started(self, sample_id: int, model_name: Optional[str] = None) -> None:
        logger.info("  ↳ grading")

    async def sample_completed(
        self,
        sample_id: int,
        passed: bool,
        score: Optional[float] = None,
        model_name: Optional[str] = None,
        metric_scores: Optional[Dict[str, float]] = None,
        metric_pass: Optional[Dict[str, bool]] = None,
        rationale: Optional[str] = None,
        metric_rationales: Optional[Dict[str, str]] = None,
    ) -> None:
        status = "✓ passed" if passed else "✗ failed"
        score_info = f" (score: {score:.2f})" if score is not None else ""
        logger.info(f"  {status}{score_info}")

    async def sample_error(self, sample_id: int, error: str, model_name: Optional[str] = None) -> None:
        logger.error(f"  ✗ error: {error}")


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

    # enable logging for progress callback
    caplog.set_level(logging.INFO)
    progress_callback = SimpleTestProgressCallback()

    logger.info(f"\n{'=' * 60}\nRunning suite: {suite_path.name}\n{'=' * 60}")

    # run suite live against letta cloud
    result = await run_suite(
        suite_path=suite_path,
        max_concurrent=2,
        progress_callback=progress_callback,
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

    # enable logging for progress callback
    caplog.set_level(logging.INFO)
    progress_callback = SimpleTestProgressCallback()

    logger.info(f"\n{'=' * 60}\nRunning suite: {suite_path.name}\n{'=' * 60}")

    # run suite live against letta cloud
    result = await run_suite(
        suite_path=suite_path,
        max_concurrent=2,
        progress_callback=progress_callback,
        cached_results_path=None,
        output_path=output_path,
        letta_api_key=os.getenv("LETTA_API_KEY"),
        letta_base_url="https://api.letta.com/",
    )

    logger.info(f"\n{'=' * 60}\nResults: {result.metrics.passed_attempts}/{result.metrics.total} passed\n{'=' * 60}")

    assert result.gates_passed, f"Gate failed for suite: {suite_path}"
    assert result.metrics.total > 0
    assert output_path.exists()
