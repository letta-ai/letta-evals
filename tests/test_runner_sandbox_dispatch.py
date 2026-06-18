"""Tests for the sandbox dispatch path.

These tests mock out ``ModalSandbox`` so we exercise the orchestration code
(version check, suite-dir upload, command construction, result round-trip,
teardown) without touching Modal at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock

import anyio
import pytest

from letta_evals.models import (
    GradeResult,
    ModalSandboxSpec,
    RewardOutput,
    Sample,
    SampleResult,
    Timing,
)
from letta_evals.runner import Runner, run_suite
from letta_evals.sandbox.base import ExecResult
from letta_evals.sandbox.dispatch import run_sample_in_sandbox


def _canned_result(sample_id) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        trajectory=[[]],
        submissions={"acc": "hi"},
        grades={"acc": GradeResult(score=1.0, rationale="canned")},
        reward=RewardOutput(score=1.0),
        timing=Timing(total=0.01, target=0.005),
    )


class _StubSandbox:
    """Stand-in for ModalSandbox that records calls."""

    def __init__(self, version_output: str = "letta-evals 0.17.0", exec_return_code: int = 0):
        self.started = False
        self.stopped = False
        self.uploaded_files: list[tuple[Path, str]] = []
        self.uploaded_dirs: list[tuple[Path, str]] = []
        self.execs: list[tuple[str, Optional[dict], Optional[int]]] = []
        self.downloaded: list[tuple[str, Path]] = []
        self._version_output = version_output
        self._exec_return_code = exec_return_code
        self.sandbox_id = "sb-test-1"
        self._result_to_write: Optional[SampleResult] = None

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def exec(self, command, env=None, timeout_sec=None) -> ExecResult:
        self.execs.append((command, env, timeout_sec))
        if "--version" in command:
            return ExecResult(stdout=self._version_output, stderr="", return_code=0)
        return ExecResult(stdout="", stderr="", return_code=self._exec_return_code)

    async def upload_file(self, local: Path, remote: str) -> None:
        self.uploaded_files.append((local, remote))

    async def upload_dir(self, local: Path, remote: str) -> None:
        self.uploaded_dirs.append((local, remote))

    async def download_file(self, remote: str, local: Path) -> None:
        self.downloaded.append((remote, local))
        # Pretend the in-sandbox CLI wrote a SampleResult.
        result = self._result_to_write or _canned_result("s1")
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "w") as f:
            f.write(result.model_dump_json())


def _make_runner_with_sandbox(
    tmp_path: Path,
    sandbox_spec: Optional[ModalSandboxSpec] = None,
    suite_filename: str = "suite.yaml",
) -> Runner:
    """Construct a Runner via __new__ with only what run_sample dispatch reads."""
    runner = Runner.__new__(Runner)
    runner.suite = MagicMock()
    runner.suite.name = "test-suite"
    runner.suite.base_dir = tmp_path
    runner.suite.suite_path = tmp_path / suite_filename
    runner.suite.sandbox = sandbox_spec or ModalSandboxSpec(image="img:test", timeout_sec=60)
    runner.suite.cleanup = False
    runner.suite.target = MagicMock()
    runner.client = MagicMock()
    runner.graders = {}
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
    return runner


def _write_suite_yaml(base: Path, filename: str = "suite.yaml") -> None:
    """Drop a suite YAML in base so the sandbox command can reference it."""
    (base / filename).write_text("name: test-suite\n")


def _sandbox_cli_command(stub: _StubSandbox) -> str:
    cli_commands = [c[0] for c in stub.execs if "letta-evals" in c[0] and "--sample" in c[0]]
    assert cli_commands, stub.execs
    return cli_commands[0]


class TestRunSampleInSandbox:
    def test_round_trip_uploads_suite_runs_cli_pulls_result(self, tmp_path, monkeypatch):
        _write_suite_yaml(tmp_path)
        runner = _make_runner_with_sandbox(tmp_path)
        stub = _StubSandbox()
        monkeypatch.setattr("letta_evals.sandbox.dispatch.ModalSandbox", lambda spec, session_id: stub)

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        result = anyio.run(
            run_sample_in_sandbox,
            runner.suite,
            sample,
            "openai/gpt-a",
            False,
            0.0,
        )

        assert stub.started and stub.stopped
        # Suite dir uploaded to /mnt/suite.
        assert any(remote == "/mnt/suite" for _, remote in stub.uploaded_dirs)
        # Sample JSON uploaded to /mnt/sample.json.
        assert any(remote == "/mnt/sample.json" for _, remote in stub.uploaded_files)
        # CLI command references the suite + sample + output paths and the model handle.
        cmd = _sandbox_cli_command(stub)
        assert "/mnt/suite/suite.yaml" in cmd
        assert "/mnt/sample.json" in cmd
        assert "/mnt/result.json" in cmd
        assert "--model-handle 'openai/gpt-a'" in cmd or "--model-handle openai/gpt-a" in cmd
        # Result round-tripped.
        assert result.sample_id == "s1"
        assert result.grades["acc"].score == 1.0

    @pytest.mark.parametrize("suite_filename", ["suite.yaml", "suite-mini.yaml"])
    def test_preserves_exact_suite_file_when_multiple_suite_yamls_exist(
        self, tmp_path, monkeypatch, suite_filename
    ):
        _write_suite_yaml(tmp_path, "suite.yaml")
        _write_suite_yaml(tmp_path, "suite-mini.yaml")
        runner = _make_runner_with_sandbox(tmp_path, suite_filename=suite_filename)
        stub = _StubSandbox()
        monkeypatch.setattr("letta_evals.sandbox.dispatch.ModalSandbox", lambda spec, session_id: stub)

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        anyio.run(run_sample_in_sandbox, runner.suite, sample, "openai/gpt-a", False, 0.0)

        cmd = _sandbox_cli_command(stub)
        assert f"/mnt/suite/{suite_filename}" in cmd

        other_suite_filename = "suite-mini.yaml" if suite_filename == "suite.yaml" else "suite.yaml"
        assert f"/mnt/suite/{other_suite_filename}" not in cmd

    def test_run_suite_records_loaded_suite_path(self, tmp_path, monkeypatch):
        dataset = tmp_path / "data.jsonl"
        dataset.write_text('{"input": "hi", "ground_truth": "hi"}\n')
        suite_path = tmp_path / "suite-mini.yaml"
        suite_path.write_text(
            """
name: test-suite
dataset: data.jsonl
target:
  kind: letta_code
  model_handles:
    - openai/gpt-4.1-mini
graders:
  acc:
    kind: tool
    function: exact_match
reward:
  kind: metric
  metric_key: acc
sandbox:
  kind: modal
  image: img:test
"""
        )
        captured = {}

        async def fake_execute_runs(**kwargs):
            captured["suite"] = kwargs["suite"]
            return object()

        monkeypatch.setattr("letta_evals.runner._execute_runs", fake_execute_runs)

        anyio.run(run_suite, suite_path, 1)

        assert captured["suite"].suite_path == suite_path

    def test_forwards_allowlisted_env_vars(self, tmp_path, monkeypatch):
        """Allowlisted host env vars (+ forward_env extras) reach the in-sandbox
        CLI exec; unrelated vars and absent ones do not."""
        _write_suite_yaml(tmp_path)
        runner = _make_runner_with_sandbox(
            tmp_path,
            sandbox_spec=ModalSandboxSpec(image="img:test", timeout_sec=60, forward_env=["MY_EXTRA"]),
        )
        stub = _StubSandbox()
        monkeypatch.setattr("letta_evals.sandbox.dispatch.ModalSandbox", lambda spec, session_id: stub)

        monkeypatch.setenv("LETTA_API_KEY", "sk-letta")  # default allowlist
        monkeypatch.setenv("MY_EXTRA", "extra-val")  # via forward_env
        monkeypatch.setenv("SECRET_UNRELATED", "should-not-leak")  # not listed
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # absent -> skipped

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        anyio.run(run_sample_in_sandbox, runner.suite, sample, "openai/gpt-a", False, 0.0)

        cli_execs = [c for c in stub.execs if "--sample" in c[0]]
        assert cli_execs, stub.execs
        env = cli_execs[0][1] or {}
        assert env.get("LETTA_API_KEY") == "sk-letta"
        assert env.get("MY_EXTRA") == "extra-val"
        assert "SECRET_UNRELATED" not in env
        assert "OPENAI_API_KEY" not in env

    def test_sandbox_torn_down_on_exec_failure(self, tmp_path, monkeypatch):
        _write_suite_yaml(tmp_path)
        runner = _make_runner_with_sandbox(tmp_path)
        stub = _StubSandbox(exec_return_code=1)
        monkeypatch.setattr("letta_evals.sandbox.dispatch.ModalSandbox", lambda spec, session_id: stub)

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        result = anyio.run(
            run_sample_in_sandbox,
            runner.suite,
            sample,
            "openai/gpt-a",
            False,
            0.0,
        )

        assert stub.stopped, "sandbox must be torn down even when exec fails"
        assert result.error is not None
        assert result.error.exception_type == "SandboxExecError"

    def test_exec_timeout_returns_sandbox_timeout(self, tmp_path, monkeypatch):
        """An exec returning -1 (Modal's exec-deadline sentinel) surfaces as a
        SandboxTimeout (category=target), not a generic SandboxExecError."""
        _write_suite_yaml(tmp_path)
        runner = _make_runner_with_sandbox(tmp_path)
        stub = _StubSandbox(exec_return_code=-1)
        monkeypatch.setattr("letta_evals.sandbox.dispatch.ModalSandbox", lambda spec, session_id: stub)

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        result = anyio.run(
            run_sample_in_sandbox,
            runner.suite,
            sample,
            "openai/gpt-a",
            False,
            0.0,
        )

        assert stub.stopped, "sandbox must be torn down on timeout"
        assert result.error is not None
        assert result.error.exception_type == "SandboxTimeout"
        assert result.error.category.value == "target"

    def test_version_mismatch_short_circuits(self, tmp_path, monkeypatch):
        _write_suite_yaml(tmp_path)
        runner = _make_runner_with_sandbox(
            tmp_path,
            sandbox_spec=ModalSandboxSpec(
                image="img:test",
                letta_evals_version="0.99.0",
                timeout_sec=60,
            ),
        )
        stub = _StubSandbox(version_output="letta-evals 0.17.0")
        monkeypatch.setattr("letta_evals.sandbox.dispatch.ModalSandbox", lambda spec, session_id: stub)

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        result = anyio.run(
            run_sample_in_sandbox,
            runner.suite,
            sample,
            "openai/gpt-a",
            False,
            0.0,
        )

        assert result.error is not None
        assert result.error.exception_type == "VersionMismatch"
        # No CLI exec should have happened after the failed version check.
        cli_commands = [c[0] for c in stub.execs if "--sample" in c[0]]
        assert not cli_commands

    def test_start_failure_returns_target_error(self, tmp_path, monkeypatch):
        _write_suite_yaml(tmp_path)
        runner = _make_runner_with_sandbox(tmp_path)

        class _BadStartSandbox(_StubSandbox):
            async def start(self):
                raise RuntimeError("modal create boom")

        bad = _BadStartSandbox()
        monkeypatch.setattr("letta_evals.sandbox.dispatch.ModalSandbox", lambda spec, session_id: bad)

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        result = anyio.run(
            run_sample_in_sandbox,
            runner.suite,
            sample,
            "openai/gpt-a",
            False,
            0.0,
        )

        assert result.error is not None
        assert result.error.category.value == "target"
        assert "modal create boom" in result.error.message

    def test_run_sample_dispatches_to_sandbox_when_configured(self, tmp_path, monkeypatch):
        """High-level dispatch: run_sample with sandbox-configured suite must
        delegate to the sandbox path and return its result without touching
        the in-process target/grader pipeline."""
        _write_suite_yaml(tmp_path)
        runner = _make_runner_with_sandbox(tmp_path)
        # If the dispatch slipped, the in-process path would blow up because
        # graders/target are not set up. We assert the dispatched method is
        # called instead.
        called = {}

        async def fake_in_sandbox(suite, sample, model_handle, return_token_data, t_sample_start):
            called["suite"] = suite
            called["sample"] = sample
            called["model_handle"] = model_handle
            return _canned_result(sample.id)

        monkeypatch.setattr("letta_evals.runner.run_sample_in_sandbox", fake_in_sandbox)

        sample = Sample(id="s1", input="hi", ground_truth="hi")
        result = anyio.run(runner.run_sample, sample, "openai/gpt-a")

        assert called["suite"] is runner.suite
        assert called["sample"].id == "s1"
        assert called["model_handle"] == "openai/gpt-a"
        assert result.grades["acc"].score == 1.0
