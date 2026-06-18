"""Unit tests for reward composition helpers."""

import pytest

from letta_evals import RewardOutput, reward_composer
from letta_evals.models import CustomRewardSpec, GradeResult, MetricRewardSpec, Sample, Timing
from letta_evals.rewards import RewardContext, coerce_reward_output, load_reward_composer
from letta_evals.types import RewardKind


def _ctx() -> RewardContext:
    return RewardContext(
        sample=Sample(id=0, input="hello"),
        grades={"quality": GradeResult(score=0.8), "valid": GradeResult(score=1.0)},
        submissions={"quality": "answer", "valid": "answer"},
        trajectory=[[]],
        agent_id="agent-1",
        model_handle="model-1",
        agent_state=None,
        usage=None,
        timing=Timing(total=1.0, target=0.9),
    )


class TestCoerceRewardOutput:
    def test_accepts_reward_output(self):
        output = RewardOutput(score=0.5)
        assert coerce_reward_output(output) is output

    def test_accepts_float(self):
        assert coerce_reward_output(0.75).score == 0.75

    def test_accepts_dict(self):
        output = coerce_reward_output({"score": 0.0, "metadata": {"reason": "x"}})
        assert output.score == 0.0
        assert output.metadata == {"reason": "x"}

    def test_rejects_invalid_type(self):
        with pytest.raises(TypeError):
            coerce_reward_output("bad")  # type: ignore[arg-type]

    def test_validates_range(self):
        with pytest.raises(ValueError):
            coerce_reward_output(2.0)


class TestMetricReward:
    @pytest.mark.asyncio
    async def test_metric_reward_uses_grade_score(self):
        composer = load_reward_composer(MetricRewardSpec(kind=RewardKind.METRIC, metric_key="quality"), None)
        output = await composer(_ctx())
        assert output == RewardOutput(score=0.8)

    @pytest.mark.asyncio
    async def test_missing_metric_raises(self):
        composer = load_reward_composer(MetricRewardSpec(kind=RewardKind.METRIC, metric_key="missing"), None)
        with pytest.raises(KeyError):
            await composer(_ctx())


def test_reward_composer_requires_ctx_param():
    with pytest.raises(TypeError):

        @reward_composer
        def bad(sample):
            return 1.0


class TestCustomReward:
    @pytest.mark.asyncio
    async def test_custom_reward_module(self, tmp_path):
        module = tmp_path / "rewards.py"
        module.write_text(
            "from letta_evals import RewardOutput, reward_composer\n"
            "@reward_composer\n"
            "def compose(ctx):\n"
            "    if ctx.grades['valid'].score < 1.0:\n"
            "        return RewardOutput(score=0.0, metadata={'reason': 'invalid'})\n"
            "    return ctx.grades['quality'].score\n"
        )
        composer = load_reward_composer(
            CustomRewardSpec(kind=RewardKind.CUSTOM, function="rewards.py:compose"), tmp_path
        )
        output = await composer(_ctx())
        assert output == RewardOutput(score=0.8)

    @pytest.mark.asyncio
    async def test_custom_reward_requires_decorator(self, tmp_path):
        module = tmp_path / "rewards.py"
        module.write_text("def compose(ctx):\n    return 1.0\n")
        with pytest.raises(ValueError, match="decorated"):
            load_reward_composer(CustomRewardSpec(kind=RewardKind.CUSTOM, function="rewards.py:compose"), tmp_path)
