"""Configuration specs.

Pydantic models for the suite YAML: target (agent), graders (tool / model
judge), gates (simple / weighted_average / logical), and the
top-level :class:`SuiteSpec`. Also includes small gate helper functions used
by metrics computation.
"""

import shlex
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from letta_evals.types import (
    Aggregation,
    GateKind,
    GraderKind,
    LLMProvider,
    LogicalOp,
    MetricOp,
    TargetKind,
)

# Target specs


class BaseTargetSpec(BaseModel):
    """Base target configuration with common fields."""

    model_config = ConfigDict(extra="forbid")

    kind: TargetKind = Field(description="Type of target")
    base_url: str = Field(default="http://localhost:8283", description="Letta server URL")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    timeout: float = Field(default=300.0, description="Request timeout in seconds")
    project_id: Optional[str] = Field(default=None, description="Letta project ID")
    max_retries: int = Field(default=0, description="Maximum number of retries for failed target runs")

    # model handles to test (cloud-compatible model identifiers)
    model_handles: Optional[List[str]] = Field(
        default=None, description="List of model handles (e.g., 'openai/gpt-4.1') for cloud deployments"
    )

    agent_script: Optional[str] = Field(
        default=None, description="Path to Python script with AgentFactory (e.g., script.py:factory_fn)"
    )

    # internal field for path resolution
    base_dir: Optional[Path] = Field(default=None, exclude=True)


class LettaCodeTargetSpec(BaseTargetSpec):
    """Letta code target configuration."""

    kind: Literal[TargetKind.LETTA_CODE] = TargetKind.LETTA_CODE

    allowed_tools: Optional[List[str]] = Field(
        default=None, description="List of allowed tools for letta code (e.g., ['Bash', 'Read'])"
    )
    disallowed_tools: Optional[List[str]] = Field(default=None, description="List of disallowed tools for letta code")
    flags: Optional[str] = Field(
        default=None,
        description="Additional CLI flags to pass to letta code (e.g., '--memfs --context-window 8000'). "
        "Parsed with shell quoting rules so values with spaces can be quoted.",
    )
    permission_mode: Optional[str] = Field(
        default=None,
        description="Permission mode for letta code (e.g., 'memory' to scope writes to memory roots).",
    )


TargetSpec = LettaCodeTargetSpec


# Sandbox specs


class ModalSandboxSpec(BaseModel):
    """Modal sandbox execution configuration.

    When attached to a :class:`SuiteSpec` via the ``sandbox`` field, every
    sample executes inside a fresh Modal sandbox driven by the host runner.

    When ``image`` is unset (the default), the driver builds the sandbox
    image from the Dockerfile bundled at ``letta_evals/sandbox/Dockerfile``
    via Modal's ``Image.from_dockerfile``. The bundled recipe carries the
    ``letta-evals`` Python package and the ``@letta-ai/letta-code`` npm
    CLI, so no registry publishing is required for the common case.
    """

    kind: Literal["modal"] = "modal"
    image: Optional[str] = Field(
        default=None,
        description=(
            "Optional registry reference for the runtime image. When unset, "
            "the driver builds an image from the bundled Dockerfile "
            "(letta_evals/sandbox/Dockerfile) via Modal's Image.from_dockerfile. "
            "Set this to point at a pre-built registry image when you need "
            "extra runtime your derived image already ships with."
        ),
    )
    letta_evals_version: Optional[str] = Field(
        default=None,
        description=(
            "If set, the runner asserts the image's ``letta-evals --version`` "
            "matches at sandbox start to guard against SampleResult schema drift."
        ),
    )
    secrets: List[str] = Field(default_factory=list, description="Names of pre-uploaded Modal Secrets to attach")
    forward_env: List[str] = Field(
        default_factory=list,
        description=(
            "Extra host environment variable names to forward into the sandbox, "
            "in addition to a built-in allowlist (LETTA_API_KEY and common model-"
            "provider keys). Values are read from the host process environment at "
            "run time; only listed names are forwarded — never the whole environment."
        ),
    )
    volumes: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of in-sandbox mount path -> Modal Volume name",
    )
    cpu: int = Field(default=2, description="vCPU count for the sandbox")
    memory_mb: int = Field(default=2048, description="Memory in MiB for the sandbox")
    timeout_sec: int = Field(default=1800, description="Hard sandbox timeout in seconds")
    idle_timeout_sec: Optional[int] = Field(default=None, description="Idle timeout (seconds) before auto-termination")
    block_network: bool = Field(default=False, description="If True, the sandbox is created without network access")
    app_name: str = Field(default="letta-evals", description="Modal App name to attach sandboxes to")


SandboxSpec = Annotated[ModalSandboxSpec, Field(discriminator="kind")]


# Grader specs


class BaseGraderSpec(BaseModel):
    """Base grader configuration with common fields."""

    kind: GraderKind = Field(description="Type of grader (tool or model_judge)")
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
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model to use for model judge (e.g., gpt-4o-mini for OpenAI, claude-haiku-4-5-20251001 for Anthropic, gemini-3-flash-preview for Google)",
    )
    temperature: float = Field(default=0.0, description="Temperature for model judge")
    provider: LLMProvider = Field(default=LLMProvider.OPENAI, description="LLM provider for model judge")
    max_retries: int = Field(default=5, description="Maximum number of retries for model judge")
    timeout: float = Field(default=120.0, description="Timeout for model judge in seconds")
    system_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional system prompt sent to the judge. By default no system "
            "prompt is sent — the rubric is the entire instruction. Set this "
            "to opt back into a system message (e.g. role framing)."
        ),
    )

    @model_validator(mode="after")
    def validate_prompt_config(self):
        if self.prompt is None and self.prompt_path is None:
            raise ValueError(
                "Model judge requires either 'prompt' or 'prompt_path' "
                "(samples may also override these via Sample.rubric / rubric_path)."
            )
        if self.prompt is not None and self.prompt_path is not None:
            raise ValueError("Model judge cannot have both prompt and prompt_path")

        # load prompt from file if needed
        if self.prompt_path:
            with open(self.prompt_path, "r") as f:
                self.prompt = f.read()

        return self


GraderSpec = Annotated[
    Union[ToolGraderSpec, ModelJudgeGraderSpec],
    Field(discriminator="kind"),
]


# Gate helper functions


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


def compute_gate_score(gate: "GateSpec", scores: Dict[str, float]) -> float:
    """Compute a single reward score from per-grader scores using the gate config.

    For simple gates, returns the score of the gate's metric_key.
    For weighted_average gates, returns the normalized weighted sum.
    For logical gates or unknown, falls back to averaging all scores.
    """
    if not scores:
        return 0.0
    if hasattr(gate, "metric_key") and not hasattr(gate, "weights"):
        # SimpleGateSpec
        return scores.get(gate.metric_key, 0.0)
    elif hasattr(gate, "weights"):
        # WeightedAverageGateSpec
        normalized = normalize_weights(gate.weights)
        return sum(normalized.get(k, 0) * scores.get(k, 0) for k in normalized)
    else:
        # LogicalGateSpec or unknown — fall back to average
        return sum(scores.values()) / len(scores)


# Gate specs


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


# Suite spec (top-level)


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
    max_concurrent: Optional[int] = Field(default=None, description="Maximum concurrent evaluations")
    output: Optional[Path] = Field(default=None, description="Directory where evaluation results are written")
    cleanup: bool = Field(
        default=False, description="Delete agents created during evaluation after each sample completes"
    )

    setup_script: Optional[str] = Field(
        default=None, description="Path to Python script with setup function (e.g., setup.py:prepare_evaluation)"
    )

    sandbox: Optional[ModalSandboxSpec] = Field(
        default=None,
        description=(
            "Optional sandbox configuration. When set, every sample runs inside "
            "a fresh per-sample sandbox. Only Modal is supported in v1."
        ),
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

            # resolve output path
            if "output" in yaml_data and yaml_data["output"] and not Path(yaml_data["output"]).is_absolute():
                yaml_data["output"] = str((base_dir / yaml_data["output"]).resolve())

            # resolve target paths
            if "target" in yaml_data:
                # resolve path-valued flags (--skills, --import) relative to suite file
                if "flags" in yaml_data["target"] and yaml_data["target"]["flags"]:
                    PATH_FLAGS = {"--skills", "--import"}
                    tokens = shlex.split(yaml_data["target"]["flags"])
                    resolved = []
                    i = 0
                    while i < len(tokens):
                        token = tokens[i]
                        resolved.append(token)
                        if token in PATH_FLAGS and i + 1 < len(tokens):
                            i += 1
                            path_val = tokens[i]
                            if not Path(path_val).is_absolute():
                                path_val = str((base_dir / path_val).resolve())
                            resolved.append(path_val)
                        i += 1
                    yaml_data["target"]["flags"] = shlex.join(resolved)

                # store base_dir in target for agent_script resolution
                yaml_data["target"]["base_dir"] = base_dir

            # resolve multi-graders (required)
            if "graders" in yaml_data and isinstance(yaml_data["graders"], dict):
                import warnings as _warnings

                resolved_graders: Dict[str, Any] = {}
                for key, gspec in yaml_data["graders"].items():
                    if "prompt_path" in gspec and gspec["prompt_path"]:
                        if not Path(gspec["prompt_path"]).is_absolute():
                            gspec["prompt_path"] = str((base_dir / gspec["prompt_path"]).resolve())
                    # Deprecation: drop legacy grader-level rubric_vars allow-list.
                    # Variables are now auto-substituted from sample.rubric_vars.
                    if "rubric_vars" in gspec:
                        _warnings.warn(
                            f"Grader '{key}': 'rubric_vars' on the grader is "
                            "deprecated and ignored. Variables in sample.rubric_vars "
                            "are now substituted into the rubric automatically. "
                            "Remove this field from your suite YAML.",
                            DeprecationWarning,
                            stacklevel=3,
                        )
                        gspec.pop("rubric_vars", None)
                    gspec["base_dir"] = base_dir
                    resolved_graders[key] = gspec
                yaml_data["graders"] = resolved_graders

            yaml_data["base_dir"] = base_dir

        return cls(**yaml_data)
