from enum import Enum


class TargetKind(str, Enum):
    LETTA_AGENT = "letta_agent"
    LETTA_CODE = "letta_code"


class GraderKind(str, Enum):
    TOOL = "tool"
    MODEL_JUDGE = "model_judge"
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
