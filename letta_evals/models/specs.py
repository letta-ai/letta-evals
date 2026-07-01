"""Configuration specs.

Pydantic models for the suite YAML: target (agent), graders (tool / model
judge), reward composition, and the top-level :class:`SuiteSpec`.
"""

import shlex
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from letta_evals.types import (
    GraderKind,
    LLMProvider,
    RewardKind,
)


class LettaCodeTargetSpec(BaseModel):
    """Letta code target configuration."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["letta_code"] = "letta_code"
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

    flags: Optional[str] = Field(
        default=None,
        description="Additional CLI flags to pass to letta code (e.g., '--memfs --context-window 8000', "
        "or tool restrictions like '--allowed-tools Bash Read'). "
        "Parsed with shell quoting rules so values with spaces can be quoted.",
    )
    permission_mode: Optional[str] = Field(
        default=None,
        description=("Permission mode passed to letta code (e.g., 'unrestricted', 'standard', or 'acceptEdits')."),
    )
    memory_workspace: bool = Field(
        default=False,
        description=(
            "Configure MEMORY_DIR/LETTA_MEMORY_DIR and run the Letta Code subprocess from a memory "
            "workspace. This does not pass '--permission-mode memory' to letta code."
        ),
    )
    memory_dir: Optional[Path] = Field(
        default=None,
        description=(
            "Optional memory workspace root. Relative paths are resolved from the suite file. "
            "When unset, memory_workspace uses per-sample overrides or the factory-created agent's memory root."
        ),
    )

    @model_validator(mode="after")
    def reject_removed_memory_permission_mode(self):
        if self.permission_mode == "memory":
            raise ValueError(
                "permission_mode: memory was removed from Letta Code. "
                "Use memory_workspace: true with a current permission_mode such as 'unrestricted'."
            )
        return self


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
    letta_code_version: Optional[str] = Field(
        default=None,
        description=(
            "If set, pins the ``@letta-ai/letta-code`` npm version installed in "
            "the bundled Dockerfile image, passed through as the "
            "``LETTA_CODE_VERSION`` build arg (e.g. '0.27.17'). Defaults to the "
            "Dockerfile's ``latest``. Ignored when ``image`` is set, since a "
            "pre-built registry image already bakes in its own letta-code."
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
    project_root: Optional[Path] = Field(
        default=None,
        description=(
            "Optional directory (an ancestor of the suite file) uploaded to the "
            "sandbox and placed on PYTHONPATH, so a suite that lives inside a "
            "larger package tree can import shared modules "
            "(e.g. `from myproject.pkg import ...`) and reference files outside "
            "the suite folder. Relative paths resolve against the suite file's "
            "directory. When unset, only the suite directory is uploaded "
            "(self-contained-suite behavior, unchanged)."
        ),
    )
    respect_gitignore: bool = Field(
        default=True,
        description=(
            "When True (default), patterns from the uploaded root's .gitignore are "
            "excluded from the upload, so anything git ignores is never shipped to "
            "the sandbox. Reads the root-level .gitignore only; nested .gitignore "
            "files, the global gitignore, and .git/info/exclude are not consulted."
        ),
    )
    cpu: int = Field(default=2, description="vCPU count for the sandbox")
    memory_mb: int = Field(default=2048, description="Memory in MiB for the sandbox")
    timeout_sec: int = Field(default=1800, description="Hard sandbox timeout in seconds")
    idle_timeout_sec: Optional[int] = Field(default=None, description="Idle timeout (seconds) before auto-termination")
    block_network: bool = Field(default=False, description="If True, the sandbox is created without network access")
    app_name: str = Field(default="letta-evals", description="Modal App name to attach sandboxes to")


SandboxSpec = Annotated[ModalSandboxSpec, Field(discriminator="kind")]


class BaseGraderSpec(BaseModel):
    """Base grader configuration with common fields."""

    model_config = ConfigDict(extra="forbid")

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


class MetricRewardSpec(BaseModel):
    """Use one grader's score directly as the sample reward."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[RewardKind.METRIC] = Field(description="reward type")
    metric_key: str = Field(description="grader name whose score becomes the reward")


class CustomRewardSpec(BaseModel):
    """Call a user-defined Python reward composer."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[RewardKind.CUSTOM] = Field(description="reward type")
    function: str = Field(description="Path to reward composer function, e.g. rewards.py:compose_reward")


RewardSpec = Annotated[
    Union[MetricRewardSpec, CustomRewardSpec],
    Field(discriminator="kind"),
]


class SuiteSpec(BaseModel):
    """Complete suite configuration."""

    name: str = Field(description="Name of the evaluation suite")
    description: Optional[str] = Field(default=None, description="Description of what this suite evaluates")
    dataset: Path = Field(description="Path to JSONL dataset file")
    target: LettaCodeTargetSpec = Field(description="Target configuration")
    graders: Optional[Dict[str, GraderSpec]] = Field(default=None, description="Multiple graders keyed by metric name")
    reward: RewardSpec = Field(description="Per-sample reward composition contract")

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

    # internal fields for path resolution / dispatch
    base_dir: Optional[Path] = Field(default=None, exclude=True)
    suite_path: Optional[Path] = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def reject_gate_config(cls, data: Any) -> Any:
        if isinstance(data, dict) and "gate" in data:
            raise ValueError("Top-level 'gate' has been removed; use top-level 'reward' instead")
        return data

    @classmethod
    def from_yaml(
        cls,
        yaml_data: Dict[str, Any],
        base_dir: Optional[Path] = None,
        suite_path: Optional[Path] = None,
    ) -> "SuiteSpec":
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
                if "memory_dir" in yaml_data["target"] and yaml_data["target"]["memory_dir"]:
                    memory_dir = Path(yaml_data["target"]["memory_dir"])
                    if not memory_dir.is_absolute():
                        yaml_data["target"]["memory_dir"] = str((base_dir / memory_dir).resolve())

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
                resolved_graders: Dict[str, Any] = {}
                for key, gspec in yaml_data["graders"].items():
                    if "prompt_path" in gspec and gspec["prompt_path"]:
                        if not Path(gspec["prompt_path"]).is_absolute():
                            gspec["prompt_path"] = str((base_dir / gspec["prompt_path"]).resolve())
                    gspec["base_dir"] = base_dir
                    resolved_graders[key] = gspec
                yaml_data["graders"] = resolved_graders

            # resolve sandbox.project_root — the ancestor dir uploaded and made
            # the import root. Relative paths resolve against the suite file dir.
            if isinstance(yaml_data.get("sandbox"), dict) and yaml_data["sandbox"].get("project_root"):
                project_root = Path(yaml_data["sandbox"]["project_root"])
                if not project_root.is_absolute():
                    project_root = base_dir / project_root
                yaml_data["sandbox"]["project_root"] = str(project_root.resolve())

            yaml_data["base_dir"] = base_dir

        if suite_path is not None:
            yaml_data["suite_path"] = suite_path

        # Fail fast at load (so `validate` catches it once) when project_root
        # isn't an ancestor of the suite — otherwise every sample's sandbox
        # dispatch would fail identically. Deferred when the suite path is
        # unknown; sandbox_mount re-checks defensively.
        sandbox_cfg = yaml_data.get("sandbox")
        if isinstance(sandbox_cfg, dict) and sandbox_cfg.get("project_root") and suite_path is not None:
            project_root = Path(sandbox_cfg["project_root"])
            try:
                suite_path.resolve().relative_to(project_root.resolve())
            except ValueError as e:
                raise ValueError(
                    f"Suite file {suite_path} is not inside sandbox.project_root {project_root}; "
                    "project_root must be an ancestor of the suite."
                ) from e

        return cls(**yaml_data)
