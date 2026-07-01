import json
import logging
import os
import shlex
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import pathspec

from letta_evals.models import Error, ModalSandboxSpec, Sample, SampleResult, SuiteSpec, Timing
from letta_evals.sandbox.modal import ModalSandbox
from letta_evals.types import ErrorCategory

logger = logging.getLogger(__name__)

# Always dropped from the upload regardless of user config — VCS metadata,
# caches, virtualenvs, and editor cruft that would only bloat the per-sample
# tarball. Gitignore-style: a bare name matches at any depth.
DEFAULT_UPLOAD_EXCLUDES = (
    ".git",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".venv",
    "venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "*.egg-info",
    ".DS_Store",
)

# Built-in host env vars forwarded to Modal sandboxes. Only explicit names are
# forwarded; suite authors can extend this with `sandbox.forward_env`.
DEFAULT_SANDBOX_FORWARD_ENV = (
    "LETTA_API_KEY",
    "LETTA_BASE_URL",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
    "TINKER_API_KEY",
)


def suite_yaml_remote_path(suite: SuiteSpec) -> str:
    """Return the in-sandbox path for the suite file the host loaded."""
    if suite.suite_path is None:
        return "/mnt/suite/suite.yaml"

    base = suite.base_dir
    if base is None:
        return f"/mnt/suite/{suite.suite_path.name}"

    try:
        relative_suite_path = suite.suite_path.relative_to(base)
    except ValueError:
        relative_suite_path = suite.suite_path.resolve().relative_to(base.resolve())

    return f"/mnt/suite/{relative_suite_path.as_posix()}"


@dataclass(frozen=True)
class SandboxMount:
    """Where the suite's code lands in the sandbox and how the CLI is invoked.

    ``project_root`` mode uploads an ancestor package tree so absolute imports
    resolve; the default (``project_root`` unset) uploads just the suite
    directory, preserving self-contained-suite behavior byte-for-byte.
    """

    local_root: Path  # host directory tarred and uploaded
    remote_root: str  # where it is extracted in the sandbox
    cwd: str  # in-sandbox working directory for the CLI
    pythonpath: Optional[str]  # dir prepended to PYTHONPATH, or None
    suite_yaml_remote: str  # in-sandbox path to the suite YAML


def sandbox_mount(suite: SuiteSpec) -> SandboxMount:
    """Compute the upload/exec layout for a suite's sandbox run.

    Raises SuiteConfigurationError-style ValueErrors with actionable messages
    when the layout can't be resolved (e.g. suite not under project_root).
    """
    project_root = suite.sandbox.project_root if suite.sandbox else None
    if project_root is None:
        return SandboxMount(
            local_root=suite.base_dir,
            remote_root="/mnt/suite",
            cwd="/mnt/suite",
            pythonpath=None,
            suite_yaml_remote=suite_yaml_remote_path(suite),
        )

    if suite.suite_path is None:
        raise ValueError("sandbox.project_root is set but the suite file path is unknown — cannot place the suite YAML.")

    try:
        relative_suite_path = suite.suite_path.resolve().relative_to(project_root.resolve())
    except ValueError as e:
        raise ValueError(
            f"Suite file {suite.suite_path} is not inside sandbox.project_root {project_root}; "
            "project_root must be an ancestor of the suite."
        ) from e

    return SandboxMount(
        local_root=project_root,
        remote_root="/mnt/project",
        cwd="/mnt/project",
        pythonpath="/mnt/project",
        suite_yaml_remote=f"/mnt/project/{relative_suite_path.as_posix()}",
    )


def build_upload_filter(
    spec: Optional[ModalSandboxSpec], root: Optional[Path] = None
) -> Callable[[str, bool], bool]:
    """Return ``keep(relpath, is_dir)`` deciding what enters the upload tarball.

    Built-in junk excludes always apply. When ``spec.respect_gitignore`` is set
    (the default), the patterns from ``root/.gitignore`` are folded in too — the
    file is just more gitignore-syntax lines, which is exactly what the exclude
    spec consumes. ``exclude`` then adds to that; a non-empty ``include`` turns
    file selection into an allowlist. Directories are always kept (descended)
    unless they themselves match an exclude, so nested allowlisted files stay
    reachable.
    """
    exclude_patterns = list(DEFAULT_UPLOAD_EXCLUDES)
    if spec and spec.respect_gitignore and root is not None:
        gitignore = root / ".gitignore"
        if gitignore.is_file():
            exclude_patterns += gitignore.read_text().splitlines()
    exclude_patterns += list(spec.exclude if spec else [])
    exclude_spec = pathspec.GitIgnoreSpec.from_lines(exclude_patterns)
    include_lines = list(spec.include) if spec and spec.include else []
    include_spec = pathspec.GitIgnoreSpec.from_lines(include_lines) if include_lines else None

    def keep(relpath: str, is_dir: bool) -> bool:
        if exclude_spec.match_file(relpath):
            return False
        if is_dir:
            return True
        if include_spec is not None and not include_spec.match_file(relpath):
            return False
        return True

    return keep


def sandbox_error_result(
    sample_id, t_sample_start: float, category: ErrorCategory, exception_type: str, message: str
) -> SampleResult:
    return SampleResult(
        sample_id=sample_id,
        trajectory=[],
        submissions={},
        grades={},
        timing=Timing(total=time.perf_counter() - t_sample_start, target=0.0),
        error=Error(
            category=category,
            exception_type=exception_type,
            message=message,
        ),
    )


def build_sandbox_command(mount: SandboxMount, model_handle: Optional[str]) -> str:
    cmd_parts = [
        "letta-evals",
        "run",
        mount.suite_yaml_remote,
        "--sample",
        "/mnt/sample.json",
        "--output-json",
        "/mnt/result.json",
    ]
    if isinstance(model_handle, str):
        cmd_parts += ["--model-handle", model_handle]

    inner_command = " ".join(shlex.quote(p) for p in cmd_parts)
    prefix = f"cd {shlex.quote(mount.cwd)}"
    if mount.pythonpath:
        # Prepend the import root, preserving any PYTHONPATH the image already set.
        prefix += f" && export PYTHONPATH={shlex.quote(mount.pythonpath)}${{PYTHONPATH:+:$PYTHONPATH}}"
    return f"{prefix} && {inner_command}"


def forwarded_sandbox_env(suite: SuiteSpec) -> Dict[str, str]:
    exec_env: Dict[str, str] = {}
    forward_names = list(DEFAULT_SANDBOX_FORWARD_ENV) + list(suite.sandbox.forward_env or [])
    for name in forward_names:
        value = os.environ.get(name)
        if value is not None:
            exec_env[name] = value
    return exec_env


async def run_sample_in_sandbox(
    suite: SuiteSpec,
    sample: Sample,
    model_handle: Optional[str],
    t_sample_start: float,
) -> SampleResult:
    """Dispatch a single sample to a fresh Modal sandbox.

    Creates a sandbox, uploads the whole suite directory tree to /mnt/suite/,
    execs the in-sandbox ``letta-evals run --sample`` invocation, and
    round-trips the resulting SampleResult JSON. The sandbox is torn down in a
    finally block.

    v1 contract: the host only sees the final SampleResult per sample.
    Mid-sample progress callbacks are not emitted. Sandbox-internal stdout and
    stderr surface only on nonzero return code (wrapped into ``Error``).
    """
    sample_id = sample.id
    session_id = f"{suite.name}-{sample_id}-{uuid.uuid4().hex[:8]}"

    if suite.base_dir is None:
        return sandbox_error_result(
            sample_id,
            t_sample_start,
            ErrorCategory.UNKNOWN,
            "SuiteConfigurationError",
            "SuiteSpec.base_dir is unset — required for sandbox execution.",
        )

    try:
        mount = sandbox_mount(suite)
    except ValueError as e:
        return sandbox_error_result(
            sample_id, t_sample_start, ErrorCategory.UNKNOWN, "SuiteConfigurationError", str(e)
        )

    sandbox = ModalSandbox(suite.sandbox, session_id=session_id)
    try:
        try:
            await sandbox.start()
        except Exception as e:
            logger.error(f"Sandbox start failed for sample {sample_id}: {e}")
            return sandbox_error_result(
                sample_id,
                t_sample_start,
                ErrorCategory.TARGET,
                type(e).__name__,
                str(e) or type(e).__name__,
            )

        logger.info("Sandbox %s started for sample %s", sandbox.sandbox_id, sample_id)

        if suite.sandbox.letta_evals_version:
            version_check = await sandbox.exec("letta-evals --version")
            expected = suite.sandbox.letta_evals_version
            if version_check.return_code != 0 or expected not in (version_check.stdout + version_check.stderr):
                message = (
                    f"Sandbox image's letta-evals version does not match "
                    f"pinned '{expected}': {version_check.stdout.strip() or version_check.stderr.strip()}"
                )
                return sandbox_error_result(
                    sample_id, t_sample_start, ErrorCategory.UNKNOWN, "VersionMismatch", message
                )

        upload_filter = build_upload_filter(suite.sandbox, mount.local_root)
        await sandbox.upload_dir(mount.local_root, mount.remote_root, path_filter=upload_filter)

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(Sample.model_validate(sample.model_dump()).model_dump_json())
            tmp_path = Path(tmp.name)
        try:
            await sandbox.upload_file(tmp_path, "/mnt/sample.json")
        finally:
            tmp_path.unlink(missing_ok=True)

        command = build_sandbox_command(mount, model_handle)
        exec_env = forwarded_sandbox_env(suite)
        if exec_env:
            logger.debug("Forwarding env vars to sandbox %s: %s", sandbox.sandbox_id, sorted(exec_env))

        exec_timeout = suite.sandbox.timeout_sec
        result_exec = await sandbox.exec(command, env=exec_env, timeout_sec=exec_timeout)

        if result_exec.return_code == -1:
            msg = f"Sandbox exec exceeded timeout_sec={exec_timeout}"
            logger.error(
                "Sandbox %s timed out for sample %s after %ss",
                sandbox.sandbox_id,
                sample_id,
                exec_timeout,
            )
            return sandbox_error_result(sample_id, t_sample_start, ErrorCategory.TARGET, "SandboxTimeout", msg)

        if result_exec.return_code != 0:
            msg = (result_exec.stderr or result_exec.stdout or "sandbox exec returned non-zero").strip()
            category = (
                ErrorCategory.GRADING if "grading" in msg.lower() or "rubric" in msg.lower() else ErrorCategory.TARGET
            )
            logger.error(
                "Sandbox %s exec failed for sample %s rc=%d: %s",
                sandbox.sandbox_id,
                sample_id,
                result_exec.return_code,
                msg[:1000],
            )
            return sandbox_error_result(sample_id, t_sample_start, category, "SandboxExecError", msg)

        with tempfile.NamedTemporaryFile("rb", suffix=".json", delete=False) as tmp_r:
            local_result_path = Path(tmp_r.name)
        try:
            try:
                await sandbox.download_file("/mnt/result.json", local_result_path)
            except Exception as e:
                return sandbox_error_result(
                    sample_id,
                    t_sample_start,
                    ErrorCategory.UNKNOWN,
                    "ResultDeserializationError",
                    f"Failed to download result: {e}",
                )

            try:
                with open(local_result_path, "r") as f:
                    result_data = json.load(f)
                # Be tolerant of older/custom sandbox images that still wrote
                # agent_state; the host does not compute on it post-grading.
                result_data.pop("agent_state", None)
                sample_result = SampleResult.model_validate(result_data)
            except Exception as e:
                return sandbox_error_result(
                    sample_id,
                    t_sample_start,
                    ErrorCategory.UNKNOWN,
                    "ResultDeserializationError",
                    f"Failed to parse result JSON: {e}",
                )
        finally:
            local_result_path.unlink(missing_ok=True)

        logger.info(
            "Sandbox %s completed sample %s in %.2fs",
            sandbox.sandbox_id,
            sample_id,
            time.perf_counter() - t_sample_start,
        )
        return sample_result
    finally:
        try:
            await sandbox.stop()
        except Exception as e:
            logger.warning("Failed to terminate sandbox %s: %s", sandbox.sandbox_id, e)
