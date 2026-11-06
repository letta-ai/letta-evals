"""Unit tests for cleanup behavior."""

from letta_evals.models import (
    LettaCodeTargetSpec,
    MetricRewardSpec,
    SuiteSpec,
    ToolGraderSpec,
)
from letta_evals.types import GraderKind, RewardKind


def _make_suite(target_spec, cleanup: bool = False) -> SuiteSpec:
    """Create a minimal SuiteSpec for testing."""
    return SuiteSpec(
        name="test-cleanup",
        dataset="fake.jsonl",
        target=target_spec,
        graders={"accuracy": ToolGraderSpec(kind=GraderKind.TOOL, function="exact_match")},
        reward=MetricRewardSpec(kind=RewardKind.METRIC, metric_key="accuracy"),
        cleanup=cleanup,
    )


# ── SuiteSpec.cleanup field ──


class TestSuiteSpecCleanupField:
    def test_defaults_to_false(self):
        suite = _make_suite(
            LettaCodeTargetSpec(model_handles=["openai/gpt-4.1-mini"]),
        )
        assert suite.cleanup is False

    def test_explicit_true(self):
        suite = _make_suite(
            LettaCodeTargetSpec(model_handles=["openai/gpt-4.1-mini"]),
            cleanup=True,
        )
        assert suite.cleanup is True

    def test_explicit_false(self):
        suite = _make_suite(
            LettaCodeTargetSpec(model_handles=["openai/gpt-4.1-mini"]),
            cleanup=False,
        )
        assert suite.cleanup is False
