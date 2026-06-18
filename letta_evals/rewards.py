"""Reward composition helpers."""

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from letta_client.types import AgentState

from letta_evals.models import (
    CustomRewardSpec,
    GradeResult,
    LettaMessageUnion,
    MetricRewardSpec,
    RewardOutput,
    RewardSpec,
    Sample,
    Timing,
    Usage,
)
from letta_evals.utils import load_object

RewardReturn = RewardOutput | float | int | dict[str, Any]
RewardComposer = Callable[["RewardContext"], RewardReturn | Awaitable[RewardReturn]]
LoadedRewardComposer = Callable[["RewardContext"], Awaitable[RewardOutput]]


@dataclass(frozen=True)
class RewardContext:
    """Inputs available to custom reward composers."""

    sample: Sample
    grades: Mapping[str, GradeResult]
    submissions: Mapping[str, str]
    trajectory: list[list[LettaMessageUnion]]
    agent_id: Optional[str]
    model_handle: Optional[str]
    agent_state: Optional[AgentState]
    usage: Optional[Usage]
    timing: Timing


def coerce_reward_output(value: RewardReturn) -> RewardOutput:
    """Normalize a composer return value into ``RewardOutput``."""
    if isinstance(value, RewardOutput):
        return value
    if isinstance(value, (float, int)):
        return RewardOutput(score=float(value))
    if isinstance(value, dict):
        return RewardOutput(**value)
    raise TypeError(f"Reward composer must return RewardOutput, float, or dict; got {type(value).__name__}")


def load_reward_composer(reward: RewardSpec, base_dir: Optional[Path]) -> LoadedRewardComposer:
    """Build the executable reward composer for a suite reward spec."""
    if isinstance(reward, MetricRewardSpec):

        async def metric_reward(ctx: RewardContext) -> RewardOutput:
            if reward.metric_key not in ctx.grades:
                raise KeyError(f"reward metric_key '{reward.metric_key}' not found in grades")
            return RewardOutput(score=ctx.grades[reward.metric_key].score)

        return metric_reward

    if isinstance(reward, CustomRewardSpec):
        func = load_object(reward.function, base_dir)
        if not getattr(func, "_is_reward_composer", False):
            raise ValueError(f"Reward composer must be decorated with @reward_composer: {reward.function}")

        async def custom_reward(ctx: RewardContext) -> RewardOutput:
            output = func(ctx)
            if inspect.isawaitable(output):
                output = await output
            return coerce_reward_output(output)

        return custom_reward

    raise ValueError(f"unknown reward spec type: {type(reward)}")
