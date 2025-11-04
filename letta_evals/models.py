from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from letta_client import AgentState, LettaMessageUnion
from pydantic import BaseModel, Field, field_validator, model_validator

from letta_evals.types import Aggregation, GateKind, GraderKind, LLMProvider, LogicalOp, MetricOp, TargetKind

# Dataset models


class Sample(BaseModel):
    """Single evaluation sample."""

    id: int = Field(description="Sample ID (0-based index from dataset)")
    input: Union[str, List[str]] = Field(description="Input message(s) to send to the agent")
    ground_truth: Optional[str] = Field(default=None, description="Expected ground_truth response for grading")
    agent_args: Optional[Dict[str, Any]] = Field(default=None, description="Custom arguments for agent creation")
    rubric_vars: Optional[Dict[str, Any]] = Field(
        default=None, description="Variables for prompt substitution in rubric graders"
    )
    extra_vars: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom user-supplied variables. Useful when writing custom extractors, graders, etc."
    )


# Config models


class BaseTargetSpec(BaseModel):
    """Base target configuration with common fields."""

    kind: TargetKind = Field(description="Type of target (agent)")
    base_url: str = Field(default="http://localhost:8283", description="Letta server URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: float = Field(default=300.0, description="Request timeout in seconds")
    project_id: Optional[str] = Field(default=None, description="Letta project ID")
    max_retries: int = Field(default=0, description="Maximum number of retries for failed create_stream calls")

    # model configs to test (names without .json extension)
    model_configs: Optional[List[str]] = Field(
        default=None, description="List of model config names from llm_model_configs directory"
    )

    # model handles to test (cloud-compatible model identifiers)
    model_handles: Optional[List[str]] = Field(
        default=None, description="List of model handles (e.g., 'openai/gpt-4.1') for cloud deployments"
    )

    # internal field for path resolution
    base_dir: Optional[Path] = Field(default=None, exclude=True)


class LettaAgentTargetSpec(BaseTargetSpec):
    """Letta agent target configuration."""

    kind: Literal[TargetKind.LETTA_AGENT] = TargetKind.LETTA_AGENT

    agent_id: Optional[str] = Field(default=None, description="ID of existing agent to use")
    agent_file: Optional[Path] = Field(default=None, description="Path to .af agent file to upload")
    agent_script: Optional[str] = Field(
        default=None, description="Path to Python script with AgentFactory (e.g., script.py:FactoryClass)"
    )

    @field_validator("agent_file")
    @classmethod
    def validate_agent_file(cls, v: Optional[Path]) -> Optional[Path]:
        if v and not str(v).endswith(".af"):
            raise ValueError("Agent file must have .af extension")
        return v

    @model_validator(mode="after")
    def validate_agent_source(self):
        sources = [self.agent_id, self.agent_file, self.agent_script]
        provided = sum(1 for s in sources if s is not None)

        if provided == 0:
            raise ValueError("Agent target requires one of: agent_id, agent_file, or agent_script")
        if provided > 1:
            raise ValueError("Agent target can only have one of: agent_id, agent_file, or agent_script")

        return self


class LettaCodeTargetSpec(BaseTargetSpec):
    """Letta code target configuration."""

    kind: Literal[TargetKind.LETTA_CODE] = TargetKind.LETTA_CODE

    working_dir: Optional[Path] = Field(default=None, description="Working directory for letta code execution")
    allowed_tools: Optional[List[str]] = Field(
        default=None, description="List of allowed tools for letta code (e.g., ['Bash', 'Read'])"
    )
    disallowed_tools: Optional[List[str]] = Field(default=None, description="List of disallowed tools for letta code")


TargetSpec = Annotated[
    Union[LettaAgentTargetSpec, LettaCodeTargetSpec],
    Field(discriminator="kind"),
]


class BaseGraderSpec(BaseModel):
    """Base grader configuration with common fields."""

    kind: GraderKind = Field(description="Type of grader (tool, model_judge, or letta_judge)")
    display_name: Optional[str] = Field(default=None, description="Human-friendly name for this metric")
    extractor: str = Field(default="last_assistant", description="Strategy for extracting submission from trajectory")
    extractor_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for the extractor")
    base_dir: Optional[Path] = Field(default=None, exclude=True)


class ToolGraderSpec(BaseGraderSpec):
    """Tool grader configuration."""

    kind: Literal[GraderKind.TOOL] = GraderKind.TOOL
    function: str = Field(description="Name of grading function for tool grader")


class ModelJudgeGraderSpec(BaseGraderSpec):
    """Model judge grader configuration."""

    kind: Literal[GraderKind.MODEL_JUDGE] = GraderKind.MODEL_JUDGE
    prompt: Optional[str] = Field(default=None, description="Prompt for model judge")
    prompt_path: Optional[Path] = Field(default=None, description="Path to file containing prompt")
    model: str = Field(default="gpt-4o-mini", description="LLM model to use for model judge")
    temperature: float = Field(default=0.0, description="Temperature for model judge")
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider for model judge")
    max_retries: int = Field(default=5, description="Maximum number of retries for model judge")
    timeout: float = Field(default=120.0, description="Timeout for model judge in seconds")
    rubric_vars: Optional[List[str]] = Field(
        default=None, description="List of required custom variables for prompt substitution"
    )

    @model_validator(mode="after")
    def validate_prompt_config(self):
        if not self.prompt and not self.prompt_path:
            raise ValueError("Model judge requires either prompt or prompt_path")
        if self.prompt and self.prompt_path:
            raise ValueError("Model judge cannot have both prompt and prompt_path")

        # load prompt from file if needed
        if self.prompt_path:
            with open(self.prompt_path, "r") as f:
                self.prompt = f.read()

        return self


class LettaJudgeGraderSpec(BaseGraderSpec):
    """Letta judge grader configuration."""

    kind: Literal[GraderKind.LETTA_JUDGE] = GraderKind.LETTA_JUDGE
    prompt: Optional[str] = Field(default=None, description="Prompt for letta judge")
    prompt_path: Optional[Path] = Field(default=None, description="Path to file containing prompt")
    agent_file: Optional[Path] = Field(default=None, description="Path to .af agent file to use as judge")
    judge_tool_name: str = Field(
        default="submit_grade", description="Name of tool that agent uses to submit score/rationale"
    )
    rubric_vars: Optional[List[str]] = Field(
        default=None, description="List of required custom variables for prompt substitution"
    )

    @field_validator("agent_file")
    @classmethod
    def validate_agent_file(cls, v: Optional[Path]) -> Optional[Path]:
        if v and not str(v).endswith(".af"):
            raise ValueError("Agent file must have .af extension")
        return v

    @model_validator(mode="after")
    def validate_letta_judge_config(self):
        if not self.prompt and not self.prompt_path:
            raise ValueError("Letta judge requires either prompt or prompt_path")
        if self.prompt and self.prompt_path:
            raise ValueError("Letta judge cannot have both prompt and prompt_path")

        # if using default agent (agent_file is None), cannot specify judge_tool_name
        if self.agent_file is None and self.judge_tool_name != "submit_grade":
            raise ValueError(
                "Cannot specify judge_tool_name when using default Letta judge (agent_file is None). "
                "To use a custom judge_tool_name, provide a custom agent_file."
            )

        # load prompt from file if needed
        if self.prompt_path:
            with open(self.prompt_path, "r") as f:
                self.prompt = f.read()

        return self


GraderSpec = Annotated[
    Union[ToolGraderSpec, ModelJudgeGraderSpec, LettaJudgeGraderSpec],
    Field(discriminator="kind"),
]


# gate helper functions


def _compare(a: float, op: MetricOp, b: float) -> bool:
    """compare two values using the given operator."""
    if op == MetricOp.GT:
        return a > b
    elif op == MetricOp.GTE:
        return a >= b
    elif op == MetricOp.LT:
        return a < b
    elif op == MetricOp.LTE:
        return a <= b
    elif op == MetricOp.EQ:
        return a == b
    return False


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """normalize weights to sum to 1.0."""
    total = sum(weights.values())
    if total == 0:
        raise ValueError("weights must sum to a non-zero value")
    return {k: v / total for k, v in weights.items()}


# gate models


class SimpleCondition(BaseModel):
    """simple condition for logical gates (leaf node)."""

    metric_key: str = Field(description="grader name to evaluate")
    aggregation: Aggregation = Field(description="aggregation function to apply")
    op: MetricOp = Field(description="comparison operator")
    value: float = Field(description="threshold value")
    pass_threshold: Optional[float] = Field(
        default=None, description="per-sample pass threshold for accuracy aggregation (defaults to 1.0)"
    )

    def __hash__(self):
        """make simple condition hashable for set operations."""
        return hash((self.metric_key, self.aggregation, self.op, self.value, self.pass_threshold))


class SimpleGateSpec(BaseModel):
    """single-metric gate."""

    kind: Literal[GateKind.SIMPLE] = Field(description="gate type")
    metric_key: str = Field(description="grader name to gate on")
    aggregation: Aggregation = Field(default=Aggregation.AVG_SCORE, description="aggregation function")
    op: MetricOp = Field(description="comparison operator")
    value: float = Field(description="threshold value")
    pass_threshold: Optional[float] = Field(
        default=None, description="per-sample pass threshold for accuracy aggregation (defaults to 1.0)"
    )


class WeightedAverageGateSpec(BaseModel):
    """weighted average of multiple grader metrics."""

    kind: Literal[GateKind.WEIGHTED_AVERAGE] = Field(description="gate type")
    aggregation: Aggregation = Field(description="aggregation function applied to each metric before weighting")
    weights: Dict[str, float] = Field(description="weights for each metric_key (grader name)")
    op: MetricOp = Field(description="comparison operator")
    value: float = Field(description="threshold value")

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, v: Dict[str, float]) -> Dict[str, float]:
        if not v:
            raise ValueError("weights dict cannot be empty")
        if any(w < 0 for w in v.values()):
            raise ValueError("weights must be non-negative")
        if sum(v.values()) == 0:
            raise ValueError("weights must sum to a non-zero value")
        return v


class LogicalGateSpec(BaseModel):
    """logical combination of conditions."""

    kind: Literal[GateKind.LOGICAL] = Field(description="gate type")
    operator: LogicalOp = Field(description="logical operator (and/or)")
    conditions: List[Union["SimpleCondition", "LogicalGateSpec"]] = Field(
        description="list of conditions (can be simple or nested logical)"
    )

    @field_validator("conditions")
    @classmethod
    def validate_conditions(cls, v: List) -> List:
        if not v:
            raise ValueError("conditions list cannot be empty")
        return v


GateSpec = Annotated[
    Union[SimpleGateSpec, WeightedAverageGateSpec, LogicalGateSpec],
    Field(discriminator="kind"),
]


class SuiteSpec(BaseModel):
    """Complete suite configuration."""

    name: str = Field(description="Name of the evaluation suite")
    description: Optional[str] = Field(default=None, description="Description of what this suite evaluates")
    dataset: Path = Field(description="Path to JSONL dataset file")
    target: TargetSpec = Field(description="Target configuration")
    graders: Optional[Dict[str, GraderSpec]] = Field(default=None, description="Multiple graders keyed by metric name")
    gate: GateSpec = Field(description="Pass/fail criteria for avg_score (required)")

    max_samples: Optional[int] = Field(default=None, description="Maximum number of samples to evaluate")
    sample_tags: Optional[List[str]] = Field(default=None, description="Only evaluate samples with these tags")
    num_runs: Optional[int] = Field(default=1, description="Number of times to run the evaluation suite")

    setup_script: Optional[str] = Field(
        default=None, description="Path to Python script with setup function (e.g., setup.py:prepare_evaluation)"
    )

    # internal field for path resolution
    base_dir: Optional[Path] = Field(default=None, exclude=True)

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

                if "working_dir" in yaml_data["target"] and yaml_data["target"]["working_dir"]:
                    if not Path(yaml_data["target"]["working_dir"]).is_absolute():
                        yaml_data["target"]["working_dir"] = str(
                            (base_dir / yaml_data["target"]["working_dir"]).resolve()
                        )

                # store base_dir in target for agent_script resolution
                yaml_data["target"]["base_dir"] = base_dir

            # resolve multi-graders (required)
            if "graders" in yaml_data and isinstance(yaml_data["graders"], dict):
                resolved_graders: Dict[str, Any] = {}
                for key, gspec in yaml_data["graders"].items():
                    if "prompt_path" in gspec and gspec["prompt_path"]:
                        if not Path(gspec["prompt_path"]).is_absolute():
                            gspec["prompt_path"] = str((base_dir / gspec["prompt_path"]).resolve())
                    if "agent_file" in gspec and gspec["agent_file"]:
                        if not Path(gspec["agent_file"]).is_absolute():
                            gspec["agent_file"] = str((base_dir / gspec["agent_file"]).resolve())
                    gspec["base_dir"] = base_dir
                    resolved_graders[key] = gspec
                yaml_data["graders"] = resolved_graders

            yaml_data["base_dir"] = base_dir

        return cls(**yaml_data)


# Target/Grader result models


class TargetResult(BaseModel):
    """Result from running a target."""

    trajectory: List[List[LettaMessageUnion]] = Field(
        description="List of conversation turns, each containing Letta messages"
    )
    agent_id: str = Field(description="ID of the agent that generated this trajectory")
    model_name: str = Field(description="Model configuration name used for this target")
    agent_usage: Optional[List[dict]] = Field(
        default=None, description="Usage statistics emitted by the agent during the run"
    )
    agent_state: Optional[AgentState] = Field(
        default=None, description="Agent state after running the target (includes memory blocks)"
    )


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
    """metrics for a specific model configuration."""

    model_name: str = Field(description="model configuration name")
    total: int = Field(description="total results (success + error)")
    total_attempted: int = Field(description="total successfully attempted (completed without error)")
    avg_score_attempted: float = Field(description="average score across attempted results (0.0 to 1.0)")
    avg_score_total: float = Field(description="average score across all results (0.0 to 1.0)")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="per-metric scores (metric_key -> average score percentage)"
    )


class MetricAggregate(BaseModel):
    """aggregate metrics for a single metric key (grader)."""

    avg_score_attempted: float = Field(
        description="average score for this metric across attempted results (0.0 to 1.0)"
    )
    avg_score_total: float = Field(description="average score for this metric across all results (0.0 to 1.0)")
    pass_rate: float = Field(description="average score as percentage")


class Metrics(BaseModel):
    """evaluation metrics."""

    total: int = Field(description="total results (success + error)")
    total_attempted: int = Field(description="total successfully attempted (completed without error)")
    avg_score_attempted: float = Field(description="average score across attempted results (0.0 to 1.0)")
    avg_score_total: float = Field(description="average score across all results (0.0 to 1.0)")
    per_model: Optional[List[ModelMetrics]] = Field(
        default=None, description="metrics broken down by model configuration"
    )
    by_metric: Optional[Dict[str, MetricAggregate]] = Field(default=None, description="aggregates for each metric key")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="per-metric scores (metric_key -> average score percentage)"
    )


class RunStatistics(BaseModel):
    """Aggregate statistics across multiple evaluation runs."""

    num_runs: int = Field(description="Total number of runs executed")
    runs_passed: int = Field(description="Number of runs that passed the gate")
    mean_avg_score_attempted: float = Field(description="Mean of avg_score_attempted across all runs")
    std_avg_score_attempted: float = Field(description="Standard deviation of avg_score_attempted across all runs")
    mean_avg_score_total: float = Field(description="Mean of avg_score_total across all runs")
    std_avg_score_total: float = Field(description="Standard deviation of avg_score_total across all runs")
    mean_scores: Dict[str, float] = Field(
        default_factory=dict, description="Mean score for each metric across all runs"
    )
    std_scores: Dict[str, float] = Field(
        default_factory=dict, description="Standard deviation for each metric across all runs"
    )
    individual_run_metrics: List[Metrics] = Field(description="Metrics from each individual run")


class SampleResult(BaseModel):
    """Result for a single sample evaluation."""

    sample: Sample = Field(description="The original sample that was evaluated")
    submission: str = Field(description="Extracted response from the trajectory")
    submissions: Optional[Dict[str, str]] = Field(default=None, description="Per-metric extracted submissions")
    trajectory: List[List[LettaMessageUnion]] = Field(description="Full conversation trajectory from the agent")
    agent_id: Optional[str] = Field(default=None, description="ID of the agent that generated this trajectory")
    grade: GradeResult = Field(description="Grading result for this sample")
    grades: Optional[Dict[str, GradeResult]] = Field(default=None, description="Per-metric grading results")
    model_name: Optional[str] = Field(description="Model configuration name used for this sample")
    agent_usage: Optional[List[dict]] = Field(
        default=None, description="Usage statistics emitted by the agent during the run"
    )


class RunnerResult(BaseModel):
    """Complete evaluation run result."""

    suite: str = Field(description="Name of the evaluation suite")
    config: Dict[str, Any] = Field(description="Configuration used for this run (target config, grader config, etc.)")
    results: List[SampleResult] = Field(description="Results for each evaluated sample")
    metrics: Metrics = Field(description="Aggregate metrics across all samples")
    gates_passed: bool = Field(description="Whether all gate criteria were satisfied")
    run_statistics: Optional[RunStatistics] = Field(
        default=None, description="Aggregate statistics across multiple runs (if num_runs > 1)"
    )
