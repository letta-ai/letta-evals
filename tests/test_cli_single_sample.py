"""Tests for the ``letta-evals run --sample ... --output-json ...`` short-circuit.

These tests exercise the in-sandbox entrypoint without standing up a Modal
sandbox. They patch ``Runner.run_sample`` to return a canned ``SampleResult``
so we can assert on the round-trip: suite YAML + Sample JSON → SampleResult
JSON file on disk.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import anyio
import pytest

from letta_evals.models import GradeResult, RewardOutput, SampleResult, Timing


def _write_minimal_suite(tmp_path: Path, *, with_sandbox: bool = False) -> Path:
    """Write a tiny suite.yaml + a single-sample dataset.jsonl. Returns the suite path."""
    suite_yaml = tmp_path / "suite.yaml"
    dataset_jsonl = tmp_path / "samples.jsonl"

    dataset_jsonl.write_text(json.dumps({"id": "s1", "input": "hello", "ground_truth": "hi"}) + "\n")

    yaml_text = """\
name: cli-single-sample-test
dataset: samples.jsonl
target:
  kind: letta_code
  model_handles: ["openai/gpt-4.1-mini"]
graders:
  acc:
    kind: tool
    function: exact_match
reward:
  kind: metric
  metric_key: acc
"""
    if with_sandbox:
        yaml_text += """\
sandbox:
  kind: modal
  image: ghcr.io/letta/letta-evals-runtime:test
"""
    suite_yaml.write_text(yaml_text)
    return suite_yaml


def _write_sample(tmp_path: Path) -> Path:
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps({"id": "s1", "input": "hello", "ground_truth": "hi"}))
    return sample_path


def _canned_result() -> SampleResult:
    return SampleResult(
        sample_id="s1",
        trajectory=[[]],
        submissions={"acc": "hi"},
        grades={"acc": GradeResult(score=1.0, rationale="canned")},
        reward=RewardOutput(score=1.0),
        timing=Timing(total=0.01, target=0.005),
    )


class TestRunSingleSampleHelper:
    def test_round_trip_writes_sample_result_json(self, tmp_path):
        from letta_evals.cli import _run_single_sample

        suite_path = _write_minimal_suite(tmp_path)
        sample_path = _write_sample(tmp_path)
        out_path = tmp_path / "out.json"

        with patch("letta_evals.runner.Runner.run_sample", new_callable=AsyncMock) as run_sample:
            run_sample.return_value = _canned_result()
            anyio.run(
                _run_single_sample,
                suite_path,
                sample_path,
                out_path,
                None,  # api_key
                None,  # base_url
                None,  # project_id
                None,  # model_handle
            )

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["sample_id"] == "s1"
        assert data["grades"]["acc"]["score"] == 1.0

        revived = SampleResult.model_validate(data)
        assert revived.grades["acc"].rationale == "canned"

    def test_single_sample_output_excludes_agent_state(self, tmp_path):
        """The sandbox entrypoint consumes agent_state in-process for grading,
        but must not serialize it across the sandbox→host boundary."""
        from letta_evals.cli import _run_single_sample

        suite_path = _write_minimal_suite(tmp_path)
        sample_path = _write_sample(tmp_path)
        out_path = tmp_path / "out.json"

        result = _canned_result().model_copy(update={"agent_state": {"provider_type": "unknown-preview-provider"}})

        with patch("letta_evals.runner.Runner.run_sample", new_callable=AsyncMock) as run_sample:
            run_sample.return_value = result
            anyio.run(
                _run_single_sample,
                suite_path,
                sample_path,
                out_path,
                None,
                None,
                None,
                None,
            )

        data = json.loads(out_path.read_text())
        assert data["sample_id"] == "s1"
        assert "agent_state" not in data

    def test_drops_sandbox_field_to_prevent_reentry(self, tmp_path):
        """If the suite YAML declares sandbox:, the in-sandbox path must
        strip it before constructing SuiteSpec — otherwise the Runner would
        try to re-enter a sandbox from inside a sandbox."""
        from letta_evals.cli import _run_single_sample

        suite_path = _write_minimal_suite(tmp_path, with_sandbox=True)
        sample_path = _write_sample(tmp_path)
        out_path = tmp_path / "out.json"

        seen_sandbox = {}

        async def fake_run_sample(self, sample, model_handle=None, return_token_data=False):
            seen_sandbox["value"] = self.suite.sandbox
            return _canned_result()

        with patch("letta_evals.runner.Runner.run_sample", new=fake_run_sample):
            anyio.run(
                _run_single_sample,
                suite_path,
                sample_path,
                out_path,
                None,
                None,
                None,
                None,
            )

        assert seen_sandbox["value"] is None, "sandbox: field must be stripped on the in-sandbox side"

    def test_model_handle_override_narrows_to_single_model(self, tmp_path):
        from letta_evals.cli import _run_single_sample

        # Write a suite with two model handles; pass --model-handle to scope.
        suite_yaml = tmp_path / "suite.yaml"
        (tmp_path / "samples.jsonl").write_text(json.dumps({"id": "s1", "input": "hello", "ground_truth": "hi"}) + "\n")
        suite_yaml.write_text(
            """\
name: multi-model
dataset: samples.jsonl
target:
  kind: letta_code
  model_handles: ["openai/gpt-4.1-mini"]
  model_handles: ["openai/gpt-a", "openai/gpt-b"]
graders:
  acc:
    kind: tool
    function: exact_match
reward:
  kind: metric
  metric_key: acc
"""
        )
        sample_path = _write_sample(tmp_path)
        out_path = tmp_path / "out.json"

        seen_models = {}

        async def fake_run_sample(self, sample, model_handle=None, return_token_data=False):
            seen_models["handles"] = list(self.model_handles)
            seen_models["model_handle"] = model_handle
            return _canned_result()

        with patch("letta_evals.runner.Runner.run_sample", new=fake_run_sample):
            anyio.run(
                _run_single_sample,
                suite_yaml,
                sample_path,
                out_path,
                None,
                None,
                None,
                "openai/gpt-a",
            )

        assert seen_models["handles"] == ["openai/gpt-a"]
        assert seen_models["model_handle"] == "openai/gpt-a"


@pytest.mark.skipif(sys.platform == "win32", reason="subprocess invocation differences")
class TestCLIInvocation:
    def test_missing_output_json_fails(self, tmp_path):
        suite_path = _write_minimal_suite(tmp_path)
        sample_path = _write_sample(tmp_path)

        env = os.environ.copy()
        env["NO_COLOR"] = "1"
        result = subprocess.run(
            [sys.executable, "-m", "letta_evals", "run", str(suite_path), "--sample", str(sample_path)],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, result.stdout + result.stderr
