from letta_evals import RewardOutput, reward_composer


@reward_composer
def quality_if_ascii(ctx):
    quality = ctx.grades["quality"].score
    ascii_only = ctx.grades["ascii_only"].score
    if ascii_only < 1.0:
        return RewardOutput(
            score=0.0,
            metadata={"reason": "ascii_check_failed", "failed_metric": "ascii_only"},
        )
    return RewardOutput(score=quality)


@reward_composer
def weighted_quality_ascii(ctx):
    quality = ctx.grades["quality"].score
    ascii_only = ctx.grades["ascii_only"].score
    return RewardOutput(score=(0.7 * quality) + (0.3 * ascii_only))
