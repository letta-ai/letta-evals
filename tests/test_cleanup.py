"""Unit tests for the SuiteSpec cleanup attribute and Runner._should_cleanup_agent."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta_evals.models import (
    LettaAgentTargetSpec,
    LettaCodeTargetSpec,
    SimpleGateSpec,
    SuiteSpec,
    ToolGraderSpec,
)
from letta_evals.types import GateKind, GraderKind, MetricOp, TargetKind


def _make_suite(target_spec, cleanup: bool = False) -> SuiteSpec:
    """Create a minimal SuiteSpec for testing."""
    return SuiteSpec(
        name="test-cleanup",
        dataset=Path("fake.jsonl"),
        target=target_spec,
        graders={"accuracy": ToolGraderSpec(kind=GraderKind.TOOL, function="exact_match")},
        gate=SimpleGateSpec(kind=GateKind.SIMPLE, metric_key="accuracy", op=MetricOp.GTE, value=0.5),
        cleanup=cleanup,
    )


# ── SuiteSpec.cleanup field ──


class TestSuiteSpecCleanupField:
    def test_defaults_to_false(self):
        suite = _make_suite(
            LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_file=Path("agent.af")),
        )
        assert suite.cleanup is False

    def test_explicit_true(self):
        suite = _make_suite(
            LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_file=Path("agent.af")),
            cleanup=True,
        )
        assert suite.cleanup is True

    def test_explicit_false(self):
        suite = _make_suite(
            LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_file=Path("agent.af")),
            cleanup=False,
        )
        assert suite.cleanup is False


# ── Runner._should_cleanup_agent ──


class TestShouldCleanupAgent:
    """Test _should_cleanup_agent with various target/cleanup configurations."""

    def _make_runner(self, suite: SuiteSpec):
        """Create a Runner with mocked dependencies."""
        with patch("letta_evals.runner.AsyncLetta"):
            from letta_evals.runner import Runner

            runner = MagicMock(spec=Runner)
            runner.suite = suite
            runner._should_cleanup_agent = Runner._should_cleanup_agent.__get__(runner)
            return runner

    def test_cleanup_false_letta_agent_file(self):
        suite = _make_suite(
            LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_file=Path("agent.af")),
            cleanup=False,
        )
        runner = self._make_runner(suite)
        assert runner._should_cleanup_agent() is False

    def test_cleanup_true_letta_agent_file(self):
        suite = _make_suite(
            LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_file=Path("agent.af")),
            cleanup=True,
        )
        runner = self._make_runner(suite)
        assert runner._should_cleanup_agent() is True

    def test_cleanup_true_letta_agent_script(self):
        suite = _make_suite(
            LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_script="setup.py:factory"),
            cleanup=True,
        )
        runner = self._make_runner(suite)
        assert runner._should_cleanup_agent() is True

    def test_cleanup_true_preexisting_agent_id(self):
        """Pre-existing agent_id should never be cleaned up."""
        suite = _make_suite(
            LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_id="agent-123"),
            cleanup=True,
        )
        runner = self._make_runner(suite)
        assert runner._should_cleanup_agent() is False

    def test_cleanup_true_letta_code(self):
        suite = _make_suite(
            LettaCodeTargetSpec(kind=TargetKind.LETTA_CODE),
            cleanup=True,
        )
        runner = self._make_runner(suite)
        assert runner._should_cleanup_agent() is True

    def test_cleanup_false_letta_code(self):
        suite = _make_suite(
            LettaCodeTargetSpec(kind=TargetKind.LETTA_CODE),
            cleanup=False,
        )
        runner = self._make_runner(suite)
        assert runner._should_cleanup_agent() is False
