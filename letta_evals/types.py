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


class Aggregation(str, Enum):
    """supported aggregation functions for gate metrics."""

    AVG_SCORE = "avg_score"
    ACCURACY = "accuracy"
    MIN = "min"
    MAX = "max"
    MEDIAN = "median"
    P50 = "p50"
    P95 = "p95"
    P99 = "p99"


class GateKind(str, Enum):
    """types of gates for multi-grader support."""

    SIMPLE = "simple"
    WEIGHTED_AVERAGE = "weighted_average"
    LOGICAL = "logical"


class LogicalOp(str, Enum):
    """logical operators for combining gate conditions."""

    AND = "and"
    OR = "or"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
