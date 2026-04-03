"""Unit tests for cleanup behavior across runner and judge agents."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from letta_evals.graders.agent_judge import AgentJudgeGrader
from letta_evals.models import (
    LettaAgentTargetSpec,
    LettaCodeTargetSpec,
    LettaJudgeGraderSpec,
    Sample,
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


# ── Runner grader cleanup propagation ──


class TestRunnerCleanupPropagation:
    def test_runner_passes_suite_cleanup_to_letta_judge_grader(self):
        suite = SuiteSpec(
            name="test-letta-judge-cleanup",
            dataset=Path("fake.jsonl"),
            target=LettaAgentTargetSpec(kind=TargetKind.LETTA_AGENT, agent_id="existing-agent"),
            graders={"quality": LettaJudgeGraderSpec(kind=GraderKind.LETTA_JUDGE, prompt="Judge this", agent_id="judge")},
            gate=SimpleGateSpec(kind=GateKind.SIMPLE, metric_key="quality", op=MetricOp.GTE, value=0.5),
            cleanup=True,
        )

        with patch("letta_evals.runner.AsyncLetta"), patch("letta_evals.runner.AgentJudgeGrader") as mock_judge_grader:
            from letta_evals.runner import Runner

            Runner(suite, max_concurrent=1)

        assert mock_judge_grader.call_args.kwargs["cleanup"] is True


# ── AgentJudgeGrader cleanup ──


@pytest.mark.asyncio
class TestAgentJudgeCleanup:
    JUDGE_AGENT_FILE = Path(__file__).resolve().parents[1] / "letta_evals/graders/letta-evals-judge-agent.af"

    def _make_judge_client(self, imported_agent_id: str = "judge-created") -> MagicMock:
        client = MagicMock()
        client.agents = MagicMock()
        client.agents.import_file = AsyncMock(return_value=SimpleNamespace(agent_ids=[imported_agent_id]))
        client.agents.delete = AsyncMock()
        client.agents.messages = MagicMock()
        client.agents.messages.create = AsyncMock(return_value=object())
        client.runs = MagicMock()
        client.runs.messages = MagicMock()
        client.runs.messages.stream = AsyncMock()
        return client

    def _make_grader(
        self,
        client: MagicMock,
        *,
        cleanup: bool,
        agent_file: Path | None = None,
        agent_id: str | None = None,
    ) -> AgentJudgeGrader:
        grader = AgentJudgeGrader(
            prompt="Judge this",
            client=client,
            agent_file=agent_file,
            agent_id=agent_id,
            cleanup=cleanup,
        )
        grader.extract = MagicMock(return_value=("submission", 0.01, None))
        grader._parse_tool_calls = MagicMock(return_value=(0.8, "Looks good"))
        return grader

    async def test_cleanup_true_deletes_imported_judge_agent(self):
        client = self._make_judge_client()
        grader = self._make_grader(client, cleanup=True, agent_file=self.JUDGE_AGENT_FILE)

        with (
            patch("letta_evals.graders.agent_judge.consume_stream_with_resumes", new=AsyncMock(return_value=("run-1", None))),
            patch("letta_evals.graders.agent_judge.list_all_run_messages", new=AsyncMock(return_value=[])),
        ):
            grade, submission = await grader.grade(Sample(id=0, input="hello"), trajectory=[["message"]])

        assert submission == "submission"
        assert grade.score == 0.8
        client.agents.delete.assert_awaited_once_with(agent_id="judge-created")

    async def test_cleanup_false_keeps_imported_judge_agent(self):
        client = self._make_judge_client()
        grader = self._make_grader(client, cleanup=False, agent_file=self.JUDGE_AGENT_FILE)

        with (
            patch("letta_evals.graders.agent_judge.consume_stream_with_resumes", new=AsyncMock(return_value=("run-1", None))),
            patch("letta_evals.graders.agent_judge.list_all_run_messages", new=AsyncMock(return_value=[])),
        ):
            await grader.grade(Sample(id=0, input="hello"), trajectory=[["message"]])

        client.agents.delete.assert_not_called()

    async def test_cleanup_true_does_not_delete_preexisting_judge_agent_id(self):
        client = self._make_judge_client()
        grader = self._make_grader(client, cleanup=True, agent_id="judge-existing")

        with (
            patch("letta_evals.graders.agent_judge.consume_stream_with_resumes", new=AsyncMock(return_value=("run-1", None))),
            patch("letta_evals.graders.agent_judge.list_all_run_messages", new=AsyncMock(return_value=[])),
        ):
            await grader.grade(Sample(id=0, input="hello"), trajectory=[["message"]])

        client.agents.import_file.assert_not_called()
        client.agents.delete.assert_not_called()


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
