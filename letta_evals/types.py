from enum import Enum


class GraderKind(str, Enum):
    TOOL = "tool"
    MODEL_JUDGE = "model_judge"


class RewardKind(str, Enum):
    METRIC = "metric"
    CUSTOM = "custom"


class LLMProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ErrorCategory(str, Enum):
    """Categories of errors that can occur during evaluation."""

    TARGET = "target"  # Agent/target failed (timeout, API error, connection)
    EXTRACTION = "extraction"  # Empty trajectory or empty submission
    GRADING = "grading"  # Grader itself failed (LLM judge error, tool grader exception)
    REWARD = "reward"  # Reward composition failed
    UNKNOWN = "unknown"  # Catch-all
