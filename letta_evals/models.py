from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from letta_client import AgentState, LettaMessageUnion
from pydantic import BaseModel, Field, field_validator

from letta_evals.types import GateMetric, GraderKind, LLMProvider, MetricOp, TargetKind

# Dataset models


class Sample(BaseModel):
    """Single evaluation sample."""

    id: int = Field(description="Sample ID (0-based index from dataset)")
    input: Union[str, List[str]] = Field(description="Input message(s) to send to the agent")
    ground_truth: Optional[str] = Field(default=None, description="Expected ground_truth response for grading")
    agent_args: Optional[Dict[str, Any]] = Field(default=None, description="Custom arguments for agent creation")
    rubric_vars: Optional[Dict[str, Any]] = Field(
        default=None, description="Custom variables to substitute in rubric prompts"
    )


# Config models


class TargetSpec(BaseModel):
    """Target configuration for evaluation."""

    kind: TargetKind = Field(description="Type of target (agent)")
    base_url: str = Field(default="http://localhost:8283", description="Letta server URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: float = Field(default=300.0, description="Request timeout in seconds")
    project_id: Optional[str] = Field(default=None, description="Letta project ID")
    max_retries: int = Field(default=0, description="Maximum number of retries for failed create_stream calls")

    agent_id: Optional[str] = Field(default=None, description="ID of existing agent to use")
    agent_file: Optional[Path] = Field(default=None, description="Path to .af agent file to upload")
    agent_script: Optional[str] = Field(
        default=None, description="Path to Python script with AgentFactory (e.g., script.py:FactoryClass)"
    )

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

    # Optional display name for UI/CLI output
    display_name: Optional[str] = Field(default=None, description="Human-friendly name for this metric")

    function: Optional[str] = Field(default=None, description="Name of grading function for tool grader")

    prompt: Optional[str] = Field(default=None, description="Rubric prompt for LLM judge")
    prompt_path: Optional[Path] = Field(default=None, description="Path to file containing rubric prompt")
    model: Optional[str] = Field(default="gpt-4o-mini", description="LLM model to use for rubric grading")
    temperature: Optional[float] = Field(default=0.0, description="Temperature for LLM judge")
    provider: Optional[LLMProvider] = Field(default=LLMProvider.OPENAI, description="LLM provider for rubric grading")
    max_retries: Optional[int] = Field(default=5, description="Maximum number of retries for rubric grading")
    timeout: Optional[float] = Field(default=120.0, description="Timeout for rubric grading in seconds")
    rubric_vars: Optional[List[str]] = Field(
        default=None, description="List of required custom variables for rubric substitution"
    )

    # Agent-based judge fields
    agent_file: Optional[Path] = Field(default=None, description="Path to .af agent file to use as judge")
    judge_tool_name: Optional[str] = Field(
        default="submit_grade", description="Name of tool that agent uses to submit score/rationale"
    )

    extractor: str = Field(default="last_assistant", description="Strategy for extracting submission from trajectory")
    extractor_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for the extractor")

    base_dir: Optional[Path] = Field(default=None, exclude=True)

    @field_validator("agent_file")
    def validate_agent_file(cls, v: Optional[Path]) -> Optional[Path]:
        if v and not str(v).endswith(".af"):
            raise ValueError("Agent file must have .af extension")
        return v

    def __init__(self, **data):
        super().__init__(**data)
        if self.kind == GraderKind.TOOL:
            if not self.function:
                raise ValueError("Tool grader requires function name")
            if self.rubric_vars:
                raise ValueError("Tool grader cannot use rubric_vars (only available for rubric graders)")
        elif self.kind == GraderKind.RUBRIC:
            # check if agent-based or LLM-based judge
            if self.agent_file:
                # agent-based judge validation
                if not self.prompt and not self.prompt_path:
                    raise ValueError("Agent judge requires either prompt or prompt_path for rubric text")
                if self.prompt and self.prompt_path:
                    raise ValueError("Agent judge cannot have both prompt and prompt_path")
                if self.model != "gpt-4o-mini" or self.temperature != 0.0 or self.provider != LLMProvider.OPENAI:
                    raise ValueError(
                        "Agent judge should not specify model/temperature/provider (those are only for LLM judges)"
                    )
            else:
                # LLM-based judge validation
                if not self.prompt and not self.prompt_path:
                    raise ValueError("Rubric grader requires either prompt or prompt_path")
                if self.prompt and self.prompt_path:
                    raise ValueError("Rubric grader cannot have both prompt and prompt_path")

            # load prompt from file if needed
            if self.prompt_path:
                with open(self.prompt_path, "r") as f:
                    self.prompt = f.read()
        elif self.kind == GraderKind.LETTA_JUDGE:
            # letta judge validation
            if not self.prompt and not self.prompt_path:
                raise ValueError("Letta judge requires either prompt or prompt_path for rubric text")
            if self.prompt and self.prompt_path:
                raise ValueError("Letta judge cannot have both prompt and prompt_path")

            # if using default agent (agent_file is None), cannot specify judge_tool_name
            if self.agent_file is None and self.judge_tool_name != "submit_grade":
                raise ValueError(
                    "Cannot specify judge_tool_name when using default Letta judge (agent_file is None). "
                    "To use a custom judge_tool_name, provide a custom agent_file."
                )

            # disallow LLM-specific fields for letta judge
            if self.model != "gpt-4o-mini" or self.temperature != 0.0 or self.provider != LLMProvider.OPENAI:
                raise ValueError(
                    "Letta judge should not specify model/temperature/provider (those are only for LLM judges)"
                )

            # load prompt from file if needed
            if self.prompt_path:
                with open(self.prompt_path, "r") as f:
                    self.prompt = f.read()


class GateSpec(BaseModel):
    """Gate configuration for pass/fail criteria."""

    # Which aggregate metric kind to compare (e.g., avg_score or accuracy)
    metric: GateMetric = Field(default=GateMetric.AVG_SCORE, description="Aggregate kind to apply gate on")

    # Which metric key (grader name) to evaluate; if None, uses the single configured grader
    metric_key: Optional[str] = Field(default=None, description="Metric key (grader name) to gate on")

    # Gate comparison for the selected aggregate metric
    op: MetricOp = Field(description="Comparison operator for the selected metric")
    value: float = Field(description="Threshold value for the selected metric")

    # Optional, separate per-sample pass criteria (used for accuracy computation)
    pass_op: Optional[MetricOp] = Field(
        default=None, description="Comparison operator for per-sample pass (defaults to op)"
    )
    pass_value: Optional[float] = Field(
        default=None, description="Threshold value for per-sample pass (defaults to value)"
    )

    def _compare(self, a: float, op: MetricOp, b: float) -> bool:
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

    def check_sample(self, score: float) -> bool:
        """Check if an individual sample score passes.

        Uses pass_op/pass_value if provided; otherwise falls back to op/value.
        """
        # If gate is on accuracy aggregate and no explicit per-sample threshold set,
        # default per-sample pass to score >= 1.0 (perfect) using GTE.
        if self.pass_value is None and self.metric == GateMetric.ACCURACY:
            op = MetricOp.GTE
            value = 1.0
        else:
            op = self.pass_op or self.op
            value = self.pass_value if self.pass_value is not None else self.value
        return self._compare(score, op, value)

    # Back-compat alias
    def check_score(self, score: float) -> bool:
        return self.check_sample(score)


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

            # store base_dir in SuiteSpec for setup_script resolution
            yaml_data["base_dir"] = base_dir

        if "gate" in yaml_data and isinstance(yaml_data["gate"], dict):
            yaml_data["gate"] = GateSpec(**yaml_data["gate"])
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
    """Metrics for a specific model configuration."""

    model_name: str = Field(description="Model configuration name")
    total: int = Field(description="Total results (success + error)")
    total_attempted: int = Field(description="Total successfully attempted (completed without error)")
    avg_score_attempted: float = Field(description="Average score across attempted results (0.0 to 1.0)")
    avg_score_total: float = Field(description="Average score across all results (0.0 to 1.0)")
    passed_samples: int = Field(description="Number of attempted samples that passed the gate")
    failed_samples: int = Field(description="Number of attempted samples that failed the gate")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Per-metric pass rates (metric_key -> percentage)"
    )


class MetricAggregate(BaseModel):
    """Aggregate metrics for a single metric key (grader)."""

    avg_score_attempted: float = Field(
        description="Average score for this metric across attempted results (0.0 to 1.0)"
    )
    avg_score_total: float = Field(description="Average score for this metric across all results (0.0 to 1.0)")
    pass_rate: float = Field(description="Pass rate for this metric (percent)")
    passed_attempts: int = Field(description="Number of attempted samples that passed for this metric")
    failed_attempts: int = Field(description="Number of attempted samples that failed for this metric")


class Metrics(BaseModel):
    """Evaluation metrics."""

    total: int = Field(description="Total results (success + error)")
    total_attempted: int = Field(description="Total successfully attempted (completed without error)")
    avg_score_attempted: float = Field(description="Average score across attempted results (0.0 to 1.0)")
    avg_score_total: float = Field(description="Average score across all results (0.0 to 1.0)")
    passed_attempts: int = Field(default=0, description="Number of attempted samples that passed")
    failed_attempts: int = Field(default=0, description="Number of attempted samples that failed")
    per_model: Optional[List[ModelMetrics]] = Field(
        default=None, description="Metrics broken down by model configuration"
    )
    by_metric: Optional[Dict[str, MetricAggregate]] = Field(default=None, description="Aggregates for each metric key")
    metrics: Dict[str, float] = Field(
        default_factory=dict, description="Per-metric pass rates (metric_key -> percentage)"
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
