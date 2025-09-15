from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from letta_client import LettaMessageUnion
from pydantic import BaseModel, Field, field_validator

from letta_evals.types import GraderKind, LLMProvider, MetricOp, TargetKind

# Dataset models


class Sample(BaseModel):
    """Single evaluation sample."""

    input: Union[str, List[str]] = Field(description="Input message(s) to send to the agent")
    ground_truth: Optional[str] = Field(default=None, description="Expected ground_truth response for grading")
    agent_args: Optional[Dict[str, Any]] = Field(default=None, description="Custom arguments for agent creation")

    submission: Optional[Union[str, List[str]]] = Field(default=None, description="Actual response(s) from the agent")
    trajectory: Optional[List[Dict[str, Any]]] = Field(default=None, description="Full conversation trajectory")


# Config models


class TargetSpec(BaseModel):
    """Target configuration for evaluation."""

    kind: TargetKind = Field(description="Type of target (agent)")
    base_url: str = Field(default="http://localhost:8283", description="Letta server URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: float = Field(default=300.0, description="Request timeout in seconds")

    agent_id: Optional[str] = Field(default=None, description="ID of existing agent to use")
    agent_file: Optional[Path] = Field(default=None, description="Path to .af agent file to upload")
    agent_script: Optional[str] = Field(
        default=None, description="Path to Python script with AgentFactory (e.g., script.py:FactoryClass)"
    )

    # model configs to test (names without .json extension)
    model_configs: Optional[List[str]] = Field(
        default=None, description="List of model config names from llm_model_configs directory"
    )

    # internal field for path resolution
    base_dir: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("agent_file")
    def validate_agent_file(cls, v: Optional[Path]) -> Optional[Path]:
        if v and not str(v).endswith(".af"):
            raise ValueError("Agent file must have .af extension")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if self.kind == TargetKind.AGENT:
            sources = [self.agent_id, self.agent_file, self.agent_script]
            provided = sum(1 for s in sources if s is not None)

            if provided == 0:
                raise ValueError("Agent target requires one of: agent_id, agent_file, or agent_script")
            if provided > 1:
                raise ValueError("Agent target can only have one of: agent_id, agent_file, or agent_script")


class GraderSpec(BaseModel):
    """Grader configuration for evaluation."""

    kind: GraderKind = Field(description="Type of grader (tool or rubric)")

    function: Optional[str] = Field(default=None, description="Name of grading function for tool grader")

    prompt: Optional[str] = Field(default=None, description="Rubric prompt for LLM judge")
    prompt_path: Optional[Path] = Field(default=None, description="Path to file containing rubric prompt")
    model: Optional[str] = Field(default="gpt-4o-mini", description="LLM model to use for rubric grading")
    temperature: Optional[float] = Field(default=0.0, description="Temperature for LLM judge")
    provider: Optional[LLMProvider] = Field(default=LLMProvider.OPENAI, description="LLM provider for rubric grading")

    extractor: str = Field(default="last_assistant", description="Strategy for extracting submission from trajectory")
    extractor_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for the extractor")

    base_dir: Optional[Path] = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        if self.kind == GraderKind.TOOL:
            if not self.function:
                raise ValueError("Tool grader requires function name")
        elif self.kind == GraderKind.RUBRIC:
            if not self.prompt and not self.prompt_path:
                raise ValueError("Rubric grader requires either prompt or prompt_path")
            if self.prompt and self.prompt_path:
                raise ValueError("Rubric grader cannot have both prompt and prompt_path")
            if self.prompt_path:
                with open(self.prompt_path, "r") as f:
                    self.prompt = f.read()


class GateSpec(BaseModel):
    """Gate configuration for pass/fail criteria."""

    op: MetricOp = Field(description="Comparison operator for avg_score")
    value: float = Field(description="Threshold value for avg_score")

    def check_score(self, score: float) -> bool:
        """Check if a score satisfies this gate."""
        if self.op == MetricOp.GT:
            return score > self.value
        elif self.op == MetricOp.GTE:
            return score >= self.value
        elif self.op == MetricOp.LT:
            return score < self.value
        elif self.op == MetricOp.LTE:
            return score <= self.value
        elif self.op == MetricOp.EQ:
            return score == self.value
        return False


class SuiteSpec(BaseModel):
    """Complete suite configuration."""

    name: str = Field(description="Name of the evaluation suite")
    description: Optional[str] = Field(default=None, description="Description of what this suite evaluates")
    dataset: Path = Field(description="Path to JSONL dataset file")
    target: TargetSpec = Field(description="Target configuration")
    grader: GraderSpec = Field(description="Grader configuration")
    gate: GateSpec = Field(description="Pass/fail criteria for avg_score (required)")

    max_samples: Optional[int] = Field(default=None, description="Maximum number of samples to evaluate")
    sample_tags: Optional[List[str]] = Field(default=None, description="Only evaluate samples with these tags")

    @classmethod
    def from_yaml(cls, yaml_data: Dict[str, Any], base_dir: Optional[Path] = None) -> "SuiteSpec":
        """Create from parsed YAML data."""
        if base_dir:
            # resolve dataset path
            if "dataset" in yaml_data and not Path(yaml_data["dataset"]).is_absolute():
                yaml_data["dataset"] = str((base_dir / yaml_data["dataset"]).resolve())

            # resolve target paths
            if "target" in yaml_data:
                if "agent_file" in yaml_data["target"] and yaml_data["target"]["agent_file"]:
                    if not Path(yaml_data["target"]["agent_file"]).is_absolute():
                        yaml_data["target"]["agent_file"] = str(
                            (base_dir / yaml_data["target"]["agent_file"]).resolve()
                        )

                # store base_dir in target for agent_script resolution
                yaml_data["target"]["base_dir"] = base_dir

            # resolve grader paths
            if "grader" in yaml_data:
                if "prompt_path" in yaml_data["grader"] and yaml_data["grader"]["prompt_path"]:
                    if not Path(yaml_data["grader"]["prompt_path"]).is_absolute():
                        yaml_data["grader"]["prompt_path"] = str(
                            (base_dir / yaml_data["grader"]["prompt_path"]).resolve()
                        )

                # store base_dir in grader for custom function resolution
                yaml_data["grader"]["base_dir"] = base_dir

        if "gate" in yaml_data and isinstance(yaml_data["gate"], dict):
            yaml_data["gate"] = GateSpec(**yaml_data["gate"])
        return cls(**yaml_data)


# Target/Grader result models


class TargetResult(BaseModel):
    """Result from running a target."""

    trajectory: List[List[LettaMessageUnion]] = Field(
        description="List of conversation turns, each containing Letta messages"
    )
    agent_id: Optional[str] = Field(default=None, description="ID of the agent that generated this trajectory")
    model_name: Optional[str] = Field(default=None, description="Model configuration name used for this target")


class GradeResult(BaseModel):
    """Grading result."""

    score: float = Field(description="Numeric score between 0.0 and 1.0")
    rationale: Optional[str] = Field(default=None, description="Explanation of the grading decision")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional grading metadata")

    @field_validator("score")
    def validate_score(cls, v: float) -> float:
        if v < 0.0 or v > 1.0:
            raise ValueError(f"Score must be between 0.0 and 1.0, got {v}")
        return v


# Runner models


class ModelMetrics(BaseModel):
    """Metrics for a specific model configuration."""

    model_name: str = Field(description="Model configuration name")
    total: int = Field(description="Total number of samples evaluated")
    avg_score: float = Field(description="Average score across all samples (0.0 to 1.0)")
    passed_samples: int = Field(description="Number of samples that passed the gate")
    failed_samples: int = Field(description="Number of samples that failed the gate")


class Metrics(BaseModel):
    """Evaluation metrics."""

    total: int = Field(description="Total number of samples evaluated")
    avg_score: float = Field(description="Average score across all samples (0.0 to 1.0)")
    per_model: Optional[List[ModelMetrics]] = Field(
        default=None, description="Metrics broken down by model configuration"
    )


class SampleResult(BaseModel):
    """Result for a single sample evaluation."""

    sample: Sample = Field(description="The original sample that was evaluated")
    trajectory: List[List[LettaMessageUnion]] = Field(description="Full conversation trajectory from the agent")
    agent_id: Optional[str] = Field(default=None, description="ID of the agent that generated this trajectory")
    grade: GradeResult = Field(description="Grading result for this sample")
    model_name: Optional[str] = Field(default=None, description="Model configuration name used for this sample")


class RunnerResult(BaseModel):
    """Complete evaluation run result."""

    suite: str = Field(description="Name of the evaluation suite")
    config: Dict[str, Any] = Field(description="Configuration used for this run (target config, grader config, etc.)")
    results: List[SampleResult] = Field(description="Results for each evaluated sample")
    metrics: Metrics = Field(description="Aggregate metrics across all samples")
    gates_passed: bool = Field(description="Whether all gate criteria were satisfied")
