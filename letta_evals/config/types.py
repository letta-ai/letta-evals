from enum import Enum


class TargetKind(str, Enum):
    AGENT = "agent"


class GraderKind(str, Enum):
    TOOL = "tool"
    RUBRIC = "rubric"


class MetricOp(str, Enum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"


class LLMProvider(str, Enum):
    OPENAI = "openai"
