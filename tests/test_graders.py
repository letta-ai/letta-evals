"""Unit tests for built-in graders."""

from datetime import datetime, timezone

import pytest
from letta_client.types.agents import AssistantMessage

from letta_evals.graders.tool import ToolGrader
from letta_evals.models import Sample

_FAKE_DATE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _make_sample(ground_truth: str = "hello world") -> Sample:
    return Sample(id=0, input="test input", ground_truth=ground_truth)


def _make_trajectory(messages: list[str]) -> list[list]:
    """Create a trajectory with AssistantMessage objects."""
    return [
        [
            AssistantMessage(id=f"msg-{i}", message_type="assistant_message", date=_FAKE_DATE, content=msg)
            for i, msg in enumerate(messages)
        ]
    ]


# ── ToolGrader with built-in functions ──


class TestToolGraderExactMatch:
    @pytest.fixture
    def grader(self):
        return ToolGrader(function="exact_match")

    @pytest.mark.asyncio
    async def test_exact_match_pass(self, grader):
        sample = _make_sample("hello world")
        trajectory = _make_trajectory(["hello world"])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 1.0
        assert submission == "hello world"

    @pytest.mark.asyncio
    async def test_exact_match_fail(self, grader):
        sample = _make_sample("hello world")
        trajectory = _make_trajectory(["goodbye world"])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_exact_match_whitespace_stripped(self, grader):
        sample = _make_sample("hello world")
        trajectory = _make_trajectory(["  hello world  "])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_exact_match_case_sensitive(self, grader):
        sample = _make_sample("Hello World")
        trajectory = _make_trajectory(["hello world"])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 0.0


class TestToolGraderContains:
    @pytest.fixture
    def grader(self):
        return ToolGrader(function="contains")

    @pytest.mark.asyncio
    async def test_contains_pass(self, grader):
        sample = _make_sample("paris")
        trajectory = _make_trajectory(["The capital of France is Paris."])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_contains_case_insensitive(self, grader):
        sample = _make_sample("PARIS")
        trajectory = _make_trajectory(["The capital of France is paris."])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_contains_fail(self, grader):
        sample = _make_sample("berlin")
        trajectory = _make_trajectory(["The capital of France is Paris."])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_ground_truth(self, grader):
        sample = Sample(id=0, input="test")
        trajectory = _make_trajectory(["some response"])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 0.0


class TestToolGraderRegexMatch:
    @pytest.fixture
    def grader(self):
        return ToolGrader(function="regex_match")

    @pytest.mark.asyncio
    async def test_regex_match_pass(self, grader):
        sample = _make_sample(r"\d{4}-\d{2}-\d{2}")
        trajectory = _make_trajectory(["The date is 2026-02-27."])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_regex_match_fail(self, grader):
        sample = _make_sample(r"\d{4}-\d{2}-\d{2}")
        trajectory = _make_trajectory(["No date here."])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_invalid_regex(self, grader):
        sample = _make_sample("[invalid")
        trajectory = _make_trajectory(["some text"])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 0.0
        assert "Invalid regex" in result.rationale


# ── Edge cases: empty trajectories and submissions ──


class TestToolGraderEdgeCases:
    @pytest.fixture
    def grader(self):
        return ToolGrader(function="contains")

    @pytest.mark.asyncio
    async def test_empty_trajectory(self, grader):
        sample = _make_sample("answer")
        result, submission = await grader.grade(sample, [])
        assert result.score == 0.0
        assert "Empty trajectory" in result.rationale

    @pytest.mark.asyncio
    async def test_empty_turn(self, grader):
        sample = _make_sample("answer")
        result, submission = await grader.grade(sample, [[]])
        assert result.score == 0.0
        assert "Empty trajectory" in result.rationale

    @pytest.mark.asyncio
    async def test_extraction_time_in_metadata(self, grader):
        sample = _make_sample("hello")
        trajectory = _make_trajectory(["hello world"])
        result, submission = await grader.grade(sample, trajectory)
        assert "extraction_time" in result.metadata
        assert result.metadata["extraction_time"] >= 0

    @pytest.mark.asyncio
    async def test_multiple_messages_uses_last(self, grader):
        """last_assistant extractor picks the last assistant message."""
        sample = _make_sample("second")
        trajectory = _make_trajectory(["first message", "second message"])
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 1.0
        assert submission == "second message"

    @pytest.mark.asyncio
    async def test_trajectory_with_no_assistant_messages(self, grader):
        """Trajectory exists but has no assistant messages — extractor returns empty."""
        from letta_client.types.agents import UserMessage

        sample = _make_sample("answer")
        trajectory = [[UserMessage(id="msg-0", message_type="user_message", date=_FAKE_DATE, content="hello")]]
        result, submission = await grader.grade(sample, trajectory)
        assert result.score == 0.0
        assert "Empty submission" in result.rationale


class TestToolGraderConstructor:
    def test_invalid_function_name_raises(self):
        with pytest.raises(ValueError, match="not found in registry"):
            ToolGrader(function="nonexistent_function")

    def test_invalid_module_path_raises(self):
        with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
            ToolGrader(function="fake_module:fake_func")
