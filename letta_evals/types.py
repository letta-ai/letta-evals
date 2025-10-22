from enum import Enum


class TargetKind(str, Enum):
    AGENT = "agent"


class GraderKind(str, Enum):
    TOOL = "tool"
    RUBRIC = "rubric"
    LETTA_JUDGE = "letta_judge"


class MetricOp(str, Enum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"


class GateMetric(str, Enum):
    """Supported aggregate metrics for gating."""

    AVG_SCORE = "avg_score"
    ACCURACY = "accuracy"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
